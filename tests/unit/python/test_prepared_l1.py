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

    def test_upload_size_boundary_and_get_only_recovery(self):
        digest = "a" * 64
        home = self.run.parent
        staged = home / ".local/state/pyasc-cannbench-l1" / digest
        staged.mkdir(parents=True)
        archive = staged / "submission.zip"
        limit = 10 * 1024 * 1024
        with archive.open("wb") as f:
            f.truncate(limit)

        def invoke(action, value=digest):
            with patch.object(l1.sys, "argv", ["prepared_l1", action, value]):
                l1.main()

        with (patch.object(l1.Path, "home", return_value=home),
              patch.object(l1, "ci_client", return_value=self.q) as client):
            invoke("check-size")
            with archive.open("ab") as f:
                f.write(b"x")
            for action in ("check-size", "stage-private", "stage-public", "submit-private",
                           "submit-public", "retry-partial"):
                with self.subTest(action=action), self.assertRaisesRegex(
                    ValueError, "WEB-ZIP-001.*10485761.*10485760"
                ):
                    invoke(action, str(staged) if action.startswith("stage-") else digest)
            client.assert_not_called()
            self.assertFalse((home / ".local/state/pyasc-cannbench-smoke").exists())
            self.assertEqual(list(staged.iterdir()), [archive])

            with (patch.object(l1, "verify_package", return_value=self.manifest) as verify,
                  patch.object(l1.smoke, "collect", return_value=(0, {})) as collect,
                  patch.object(l1.smoke, "submit_once") as submit,
                  patch.object(l1, "result_receipt", return_value={"score": 161, "job": "job_a"})):
                invoke("collect")
            verify.assert_called_once_with(staged, digest)
            collect.assert_called_once()
            submit.assert_not_called()

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

    def add_active_job(self, number):
        digest = sha256(str(number).encode()).hexdigest()
        job = {"id": f"job_{number}", "submission_id": f"sub_{number}",
               "job_tag": "pyasc-smoke-" + digest[:24], "status": "queued"}
        state = {"phase": "submitted", "archive_sha256": digest, "job_id": job["id"],
                 "submission_id": job["submission_id"], "job_tag": job["job_tag"]}
        l1.smoke.save(self.ledger / (digest + ".json"), state)
        return job

    def test_accepts_different_package_alongside_verified_job(self):
        current = self.add_active_job(1)
        self.q._client.get_job.return_value = {"job": current}
        self.q._client.list_jobs.return_value = {"jobs": [{"id": "job_1", "status": "queued"}]}
        with self.assertRaisesRegex(ValueError, "Reconcile"):
            l1.smoke.submit_once(self.q, self.run, self.ledger, selections=l1.OPERATORS)
        for _ in range(2):
            state = l1.smoke.submit_once(self.q, self.run, self.ledger,
                selections=l1.OPERATORS, allow_confirmed_jobs=True)
            self.assertEqual(state["other_submitted_jobs"], ["job_1"])
        self.q._submit_streaming.assert_called_once()
        self.q._client.get_job.assert_called_once_with("job_1")

    def test_unknown_delivery_untracked_job_and_capacity_stop_new_upload(self):
        def submit():
            l1.smoke.submit_once(self.q, self.run, self.ledger,
                selections=l1.OPERATORS, allow_confirmed_jobs=True)
        untracked = {"id": "job_external", "status": "queued"}
        self.q._client.list_jobs.return_value = {"jobs": [untracked]}
        with self.assertRaisesRegex(ValueError, "active or unknown"):
            submit()
        first = self.add_active_job(1)
        originals = {first["id"]: first}
        self.q._client.get_job.side_effect = lambda ident: {"job": originals[ident]}
        self.q._client.list_jobs.return_value = {"jobs": list(originals.values())}
        for field in ("id", "submission_id", "job_tag"):
            with self.subTest(field=field):
                originals["job_1"] = {**first, field: "different"}
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    submit()
        originals["job_1"] = first
        path = self.ledger / "unknown.json"
        l1.smoke.save(path, {"phase": "ambiguous"})
        with self.assertRaisesRegex(ValueError, "Reconcile"):
            submit()
        path.unlink()
        for i in (2, 3):
            job = self.add_active_job(i)
            originals[job["id"]] = job
        with self.assertRaisesRegex(ValueError, "three active jobs"):
            submit()
        self.q._submit_streaming.assert_not_called()

    def test_overlapping_upload_returns_accepted_id_without_holding_runner(self):
        manifest = "f" * 64
        staged = self.run.parent / ".local/state/pyasc-cannbench-l1" / manifest
        staged.mkdir(parents=True)
        (staged / "submission.zip").write_bytes(b"already verified")
        state = {"phase": "submitted", "job_id": "job_a", "other_submitted_jobs": ["job_b"]}
        summary = self.run / "summary.md"
        self.q._client._get_json.return_value = {"maintenance": {"enabled": False}}
        with (
            patch.dict(l1.os.environ, GITHUB_STEP_SUMMARY=str(summary)),
            patch.object(l1.sys, "argv", ["prepared_l1", "submit-public", manifest]),
            patch.object(l1.Path, "home", return_value=self.run.parent),
            patch.object(l1, "ci_client", return_value=self.q),
            patch.object(l1, "verify_package", return_value={"visibility": "public"}),
            patch.object(l1.smoke, "submit_once", return_value=state),
            patch.object(l1.smoke, "collect") as collect,
        ):
            l1.main()
        collect.assert_not_called()
        for expected in ("job_a", "pending", "l1-collect", manifest):
            self.assertIn(expected, summary.read_text())

    def test_retry_partial_preserves_original_attempt_and_reconciles_tag_first(self):
        from evalqueue import RemoteError
        size = (self.run / "submission.zip").stat().st_size
        def submit(retry_partial=False):
            return l1.smoke.submit_once(self.q, self.run, self.ledger,
                selections=l1.OPERATORS, allow_confirmed_jobs=True, retry_partial=retry_partial)
        self.q._client.token = "test-token"
        self.q._submit_streaming.side_effect = RemoteError(
            "CANNBench upload failed (HTTP 000, curl rc=28, 8 multipart bytes in 3600s): timeout")
        with self.assertRaises(RemoteError):
            submit()
        state = submit()
        self.assertEqual(state["phase"], "ambiguous")
        self.q._client.list_jobs.return_value = {"jobs": [{"job_tag": state["job_tag"]}]}
        self.assertEqual(submit(True), state)
        self.q._submit_streaming.assert_called_once()
        self.q._client.list_jobs.return_value = {"jobs": []}
        error = self.run / "upload-error.json"
        original = json.loads(error.read_bytes())
        l1.smoke.save(error, dict(original, detail=original["detail"].replace("8 multipart", f"{size} multipart")))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            submit(True)
        l1.smoke.save(error, dict(original, intent_created_at="previous transfer"))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            submit(True)
        l1.smoke.save(error, original)
        self.q._submit_streaming.side_effect = None
        accepted = submit(True)
        self.assertEqual(accepted["phase"], "submitted")
        prior, = accepted["previous_intents"]
        self.assertEqual(prior["partial_upload"]["error"], original)
        self.assertEqual(prior["created_at"], state["created_at"])
        self.assertEqual(submit(True), accepted)
        self.assertEqual(self.q._submit_streaming.call_count, 2)

    def test_legacy_incomplete_upload_requires_bound_final_read_counters(self):
        run, ledger = self.run, self.ledger
        record = ledger / "original.json"
        state = {"phase": "ambiguous", "created_at": "2026-11-17T13:27:00+00:00"}
        l1.smoke.save(record, state)
        error = run / "upload-error.json"
        l1.smoke.save(error, {"type": "RemoteError", "at": "2026-11-17T14:27:00+00:00",
                              "detail": "CANNBench upload failed (HTTP 000, curl rc=28): timeout"})
        size = (run / "submission.zip").stat().st_size
        start = dict(event="before", at="2026-11-17T14:20:00+00:00", parent_rchar=1000,
                     expected_zip_bytes=size, zip_offset=2, child_rchar=5)
        trace = run / "transfer-trace.jsonl"
        def bind(end):
            tail = dict(event="parent_after_wait", at="2026-11-17T14:26:59+00:00", parent_rchar=1000+end,
                        upper_bound_child_read_bytes=end, expected_zip_bytes=size)
            trace.write_text(json.dumps(start)+"\n"+json.dumps(tail)+"\n")
            l1.smoke.save(run/"partial-upload.json", dict(intent_sha256=l1.smoke.sha(record),
                error_sha256=l1.smoke.sha(error), trace_sha256=l1.smoke.sha(trace)))
        bind(size)
        with self.assertRaisesRegex(ValueError, "incomplete ZIP"):
            l1.smoke.verify_partial_upload(run, record, state)
        bind(9)
        self.assertEqual(l1.smoke.verify_partial_upload(run, record, state)["trace_sha256"], l1.smoke.sha(trace))
        l1.smoke.save(record, {"phase": "another attempt"})
        with self.assertRaisesRegex(ValueError, "original attempt"):
            l1.smoke.verify_partial_upload(run, record, state)

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
