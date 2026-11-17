"""Verify that a two-operator submission cannot change one task or its visibility."""

from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "integrations/cannbench/ci"))
import prepared_l1 as l1


class PreparedL1Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / "prepared"
        self.run.mkdir()
        self.ledger = self.run.parent / "ledger"
        self.ledger.mkdir()
        self.q = Mock(benchmark_slug="official-tasks", target_hardware="950pr")
        self.q._client.list_jobs.return_value = {"jobs": []}
        self.q._submit_streaming.return_value = {
            "job_id": "job_123",
            "submission_id": "sub_abc",
        }

        def task(queue, op, fn):
            return {
                "benchmark_version": "2.1.0",
                "benchmark_slug": "official-tasks",
                "operators": [{"operator": op, "function_name": fn, "case_count": 20}],
            }, {"contract": fn}

        self.snapshot = patch.object(l1.smoke, "task_snapshot", side_effect=task)
        self.snapshot.start()
        self.addCleanup(self.snapshot.stop)
        catalog, self.tasks = l1.smoke.group_snapshot(self.q, l1.OPERATORS)
        (self.run / "submission.zip").write_bytes(b"opaque, qualified source archive")
        self.save(self.tasks, catalog)

    def save(self, tasks, catalog):
        l1.smoke.save(self.run / "task.json", tasks)
        self.manifest = {
            "catalog": catalog,
            "visibility": "private",
            "target_hardware": "950pr",
            "aggregation_token": "example",
            "files": {
                f: l1.smoke.sha(self.run / f) for f in ("task.json", "submission.zip")
            },
        }
        l1.smoke.save(self.run / "manifest.json", self.manifest)

    def test_changed_second_task_stops_before_intent(self):
        self.save(
            {**self.tasks, "swi_glu": {"contract": "old"}}, self.manifest["catalog"]
        )
        with self.assertRaisesRegex(ValueError, "changed"):
            l1.smoke.submit_once(self.q, self.run, self.ledger, selections=l1.OPERATORS)
        self.q._submit_streaming.assert_not_called()
        self.assertEqual(list(self.ledger.iterdir()), [])

    def test_source_proof_must_match_the_submitted_operator_module(self):
        sources = {"exp": b"retained Exp", "swi_glu": b"retained SwiGLU"}
        with ZipFile(self.run / "submission.zip", "w") as z:
            z.writestr("build.sh", b"build")
            z.writestr("notes/old_exp.py", sources["exp"])
            for fn, content in sources.items():
                z.writestr(f"cann_bench/{fn}.py", content)
        q = {
            "build_passed": True,
            "package_sha256": l1.smoke.sha(self.run / "submission.zip"),
            "wheel_sha256": "3" * 64,
            "operators": {
                fn: {
                    "package_source": f"cann_bench/{fn}.py",
                    "source_sha256": sha256(content).hexdigest(),
                    "passed_cases": 20,
                    "skills_revision": "a" * 40,
                }
                for fn, content in sources.items()
            },
        }

        def pin():
            l1.smoke.save(self.run / "qualification.json", q)
            for name in ("qualification.json", "submission.zip"):
                self.manifest["files"][name] = l1.smoke.sha(self.run / name)
            l1.smoke.save(self.run / "manifest.json", self.manifest)
            return l1.smoke.sha(self.run / "manifest.json")

        self.assertEqual(l1.verify_package(self.run, pin()), self.manifest)
        q["operators"]["exp"]["package_source"] = "notes/old_exp.py"
        with self.assertRaisesRegex(ValueError, "source for Exp"):
            l1.verify_package(self.run, pin())

    def test_camodel_can_qualify_for_first_hardware_measurement(self):
        source_sha = sha256(b"generated source").hexdigest()
        model = {
            "source_sha256": source_sha,
            "evidence_sha256": sha256(b"simulator evidence").hexdigest(),
            "compiled_case_ids": list(range(1, 21)),
            "numerical_pairs": [
                {"seed": s, "launch_matches_golden": True, "suppressed_rejected": True}
                for s in (19, 23)
            ],
        }
        source = {"source_sha256": source_sha, "camodel": model}
        self.assertTrue(l1.qualified_source(source))
        for change in (
            {"source_sha256": "0" * 64}, {"compiled_case_ids": [1] * 20},
            {"numerical_pairs": [model["numerical_pairs"][0]]},
            {"numerical_pairs": [*model["numerical_pairs"], {"seed": 29, "suppressed_rejected": False}]},
        ):
            with self.subTest(change=change):
                self.assertFalse(l1.qualified_source({**source, "camodel": {**model, **change}}))

    def test_private_qualification_never_becomes_public_by_dispatch(self):
        with self.assertRaisesRegex(ValueError, "visibility"):
            l1.smoke.submit_once(
                self.q, self.run, self.ledger, selections=l1.OPERATORS, public=True
            )
        self.q._submit_streaming.assert_not_called()

    def test_replaying_retains_the_two_operator_job(self):
        for _ in range(2):
            state = l1.smoke.submit_once(
                self.q, self.run, self.ledger, selections=l1.OPERATORS
            )
            self.assertEqual(state["job_id"], "job_123")
        self.q._check_credit.assert_called_once_with(2)
        self.q._submit_streaming.assert_called_once()
        args, kwargs = self.q._submit_streaming.call_args
        self.assertEqual(args[1], ["exp", "swi_glu"])
        self.assertEqual(kwargs["benchmark_version"], "2.1.0")
        self.assertIs(kwargs["public"], False)

    def test_public_receipt_requires_public_service_visibility(self):
        self.manifest["visibility"] = "public"
        job = {
            "selected_operators": ["exp", "swi_glu"],
            "case_set": "standard",
            "is_private": True,
            "is_independent_environment": False,
            "result_score": 166,
        }
        l1.smoke.save(self.run / "job.json", {"job": job})
        with self.assertRaisesRegex(ValueError, "visibility"):
            l1.result_receipt(self.run, self.manifest, {}, self.q)

    def test_published_score_matches_downloaded_package_and_anonymous_job(self):
        job = {
            "id": "job_abc", "submission_id": "sub_def", "case_set": "standard",
            "selected_operators": ["exp", "swi_glu"], "target_hardware": "950pr",
            "benchmark_slug": "official-tasks", "benchmark_version": "2.1.0", "is_private": False,
            "is_independent_environment": False, "result_score": 161.5, "status": "succeeded",
            "queued_at": "2026-11-17", "claimed_at": "2026-11-17", "finished_at": "2026-11-17",
            "submission_download_url": "/api/submissions/sub_def/download",
        }
        self.manifest["visibility"] = "public"
        self.manifest["files"]["qualification.json"] = "b" * 64
        self.q._client._url.return_value = "https://cannbench.example/api/jobs/job_abc"
        l1.smoke.save(self.run / "job.json", {"job": job})
        with patch.object(l1, "download", return_value=b"different source"), \
                self.assertRaisesRegex(ValueError, "prepared ZIP"):
            l1.result_receipt(self.run, self.manifest, {}, self.q)
        qualified = (self.run / "submission.zip").read_bytes()
        for field, value in [("result_score", 160), ("is_private", True), ("submission_id", "sub_abc"),
                             ("selected_operators", ["exp"])]:
            public = io.BytesIO(json.dumps({"job": {**job, field: value}}).encode())
            with self.subTest(field=field), patch.object(l1, "download", return_value=qualified), \
                    patch.object(l1.urllib.request, "urlopen", return_value=public), \
                    self.assertRaisesRegex(ValueError, f"differs in {field}"):
                l1.result_receipt(self.run, self.manifest, {}, self.q)
        with patch.object(l1, "download", return_value=qualified), \
                patch.object(l1.urllib.request, "urlopen",
                             return_value=io.BytesIO(json.dumps({"job": job}).encode())) as anonymous:
            receipt = l1.result_receipt(self.run, self.manifest, {"operators": []}, self.q)
        anonymous.assert_called_once_with(self.q._client._url.return_value, timeout=30)
        self.assertEqual(receipt["score"], job["result_score"])
        self.assertEqual(receipt["package_sha256"], sha256(qualified).hexdigest())
