from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations/cannbench/ci"))
sys.path.insert(1, str(ROOT / "integrations/cannbench/workers"))
submission = importlib.import_module("submission")
evalqueue = importlib.import_module("evalqueue")
campaign = importlib.import_module("run")


class FakeClient:
    def __init__(self, listed):
        self.listed = listed
        self.queue = None

    def list_jobs(self, **kwargs):
        if self.queue and self.queue.tag:
            return {"jobs": [{"id": "job_test", "job_tag": self.queue.tag, "status": "succeeded"}]} if self.listed else {"jobs": []}
        return {"jobs": []}

    def get_job(self, job_id):
        return {"job": {"id": job_id, "status": "succeeded"}}


class FakeQueue:
    def __init__(self, ambiguous=False, listed=True):
        self._client = FakeClient(listed); self._client.queue = self
        self.tag = None; self.posts = 0; self.ambiguous = ambiguous

    def _check_credit(self):
        return None

    def _submit_streaming(self, archive, operators, tag):
        self.posts += 1; self.tag = tag
        if self.ambiguous:
            raise TimeoutError("Response lost after upload")
        return {"job_id": "job_test", "submission_id": "submission_test"}


def fixture(root):
    bundle = root / "bundle"; bundle.mkdir()
    manifest = {"wheel_sha256": "wheel-sha", "catalog_sha256": "catalog-sha", "operators": {"exp": {}}}
    catalog = {"catalog_sha256": "catalog-sha", "benchmark_slug": "official-tasks", "benchmark_version": "1.1.2",
               "operators": [{"operator": "Exp", "function_name": "exp"}]}
    (bundle / "BUNDLE.json").write_text(json.dumps(manifest))
    (bundle / "qualification.json").write_text(json.dumps({"status": "passed", "wheel_sha256": "wheel-sha", "catalog_sha256": "catalog-sha"}))
    (bundle / "build.sh").write_text("#!/bin/bash\nexit 0\n")
    return bundle, catalog


class DurableSubmissionTests(unittest.TestCase):
    def test_known_job_reuses_get_and_never_posts_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle, catalog = fixture(root); queue = FakeQueue()
            with patch.object(evalqueue, "EvalQueue", return_value=queue), patch.object(submission, "shared_locks", return_value=nullcontext()), patch.object(submission, "STATE_ROOT", root / "state"):
                result, state = submission.submit(bundle, catalog, ROOT)
                self.assertEqual(state["job_id"], "job_test")
                result, state = submission.submit(bundle, catalog, ROOT)
            self.assertEqual(queue.posts, 1)
            self.assertEqual(state["phase"], "finished")

    def test_ambiguous_post_reconciles_by_get_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle, catalog = fixture(root); queue = FakeQueue(ambiguous=True)
            with patch.object(evalqueue, "EvalQueue", return_value=queue), patch.object(submission, "shared_locks", return_value=nullcontext()), patch.object(submission, "STATE_ROOT", root / "state"):
                result, state = submission.submit(bundle, catalog, ROOT)
            self.assertEqual(queue.posts, 1)
            self.assertEqual(state["job_id"], "job_test")

    def test_missing_get_result_never_authorizes_second_post(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle, catalog = fixture(root); queue = FakeQueue(ambiguous=True, listed=False)
            with patch.object(evalqueue, "EvalQueue", return_value=queue), patch.object(submission, "shared_locks", return_value=nullcontext()), patch.object(submission, "STATE_ROOT", root / "state"):
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, "POST will not be repeated"):
                        submission.submit(bundle, catalog, ROOT)
            self.assertEqual(queue.posts, 1)
            states = list((root / "state").glob("*.json"))
            self.assertEqual(json.loads(states[0].read_text())["phase"], "ambiguous")

    def test_changed_bundle_cannot_reuse_existing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle, catalog = fixture(root); queue = FakeQueue()
            with patch.object(evalqueue, "EvalQueue", return_value=queue), patch.object(submission, "shared_locks", return_value=nullcontext()), patch.object(submission, "STATE_ROOT", root / "state"):
                submission.submit(bundle, catalog, ROOT)
                (bundle / "build.sh").write_text("modified")
                with self.assertRaisesRegex(ValueError, "Existing archive differs"):
                    submission.submit(bundle, catalog, ROOT)
            self.assertEqual(queue.posts, 1)

    def test_hardware_identity_without_generated_binding_is_unverified(self):
        self.assertFalse(campaign.verify_binding({}, None, ["job_test"], ROOT))


if __name__ == "__main__":
    unittest.main()
