"""Verify that a two-operator submission cannot change one task or its visibility."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

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
            l1.result_receipt(self.run, self.manifest, {})
