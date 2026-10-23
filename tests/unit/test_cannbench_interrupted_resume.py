"""Integrity boundaries for immutable interrupted native generation backups."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_cannbench_resume import (TASK_DIR, build_recorded_iteration, driver,
                                   generation_gate, make_catalog)


class InterruptedSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "snapshot"
        self.generation = self.root / "generation"
        self.generation.mkdir(parents=True)
        self.catalog = make_catalog(["exp", "gelu"])
        (self.generation / "catalog-manifest.json").write_text(json.dumps(self.catalog))
        self.iteration, _ = build_recorded_iteration(
            self.generation / "exp-generate", self.catalog["operators"][0])
        self.seal()

    def seal(self):
        self.pause = {"source_run_id": 36878888437, "source_changed_during_copy": [],
                      "file_sha256": {str(p.relative_to(self.root)): driver._sha256(p)
                                      for p in self.root.rglob("*")
                                      if p.is_file() and p.name != "PAUSE-SNAPSHOT.json"}}
        self.write_pause()

    def write_pause(self):
        (self.root / "PAUSE-SNAPSHOT.json").write_text(json.dumps(self.pause))

    def load(self):
        return driver.load_resume_records(self.root, self.catalog, driver.SKILLS_ROOT)

    def test_partial_native_records_with_full_catalog_are_valid(self):
        result = self.load()
        self.assertEqual(set(result), {"exp"})
        binding = result["exp"][0]["resume_binding"]
        self.assertEqual(binding["source_run_id"], 36878888437)
        self.assertEqual(binding["snapshot_manifest_sha256"],
                         driver._sha256(self.root / "PAUSE-SNAPSHOT.json"))
        self.assertFalse((self.generation / "summary.json").exists())

    def test_partial_catalog_with_same_sha_string_is_rejected(self):
        partial = {**self.catalog, "operators": self.catalog["operators"][:1]}
        (self.generation / "catalog-manifest.json").write_text(json.dumps(partial))
        self.seal()
        with self.assertRaisesRegex(ValueError, "complete official catalog"):
            self.load()

    def test_identical_catalog_refetched_at_different_time_is_accepted(self):
        self.catalog["fetched_at"] = "2026-10-02T00:00:00Z"
        previous = {**self.catalog, "fetched_at": "2026-10-01T00:00:00Z"}
        (self.generation / "catalog-manifest.json").write_text(json.dumps(previous))
        self.seal()
        self.assertEqual(set(self.load()), {"exp"})

    def test_unbound_native_trace_is_rejected(self):
        del self.pause["file_sha256"]["generation/exp-generate/iter1/review.skill-trace.attempt1.json"]
        self.write_pause()
        with self.assertRaisesRegex(ValueError, "unbound"):
            self.load()

    def test_changed_candidate_is_rejected(self):
        (self.iteration / "candidate.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed on disk"):
            self.load()

    def test_changed_during_copy_is_rejected(self):
        self.pause["source_changed_during_copy"] = ["generation.log"]
        self.write_pause()
        with self.assertRaisesRegex(ValueError, "unchanged source"):
            self.load()

    def test_parent_traversal_and_absolute_paths_are_rejected(self):
        for relative in ("../outside", "/tmp/outside"):
            with self.subTest(relative=relative):
                self.seal()
                self.pause["file_sha256"][relative] = "0" * 64
                self.write_pause()
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    self.load()

    def test_symlink_to_same_bytes_is_rejected(self):
        path = self.iteration / "candidate.py"
        external = self.root.parent / "candidate.py"
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.load()

    def test_unlisted_iteration_cannot_be_discovered(self):
        build_recorded_iteration(self.generation / "gelu-generate", self.catalog["operators"][1])
        with self.assertRaisesRegex(ValueError, "unbound"):
            self.load()


class InterruptedReplayOrderTests(unittest.TestCase):
    """Recorded-pass-first ordering and multi-iteration replay attempts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "snapshot"
        self.generation = self.root / "generation"
        self.generation.mkdir(parents=True)
        self.catalog = make_catalog(["exp"])
        self.spec = self.catalog["operators"][0]
        (self.generation / "catalog-manifest.json").write_text(json.dumps(self.catalog))
        # Newest iteration recorded a compiler failure; the older one passes.
        item_dir = self.generation / "exp-generate"
        self.iter1, _ = build_recorded_iteration(item_dir, self.spec)
        self.iter2 = item_dir / "iter2"
        self.iter2.mkdir(parents=True)
        for path in self.iter1.iterdir():
            if path.is_file():
                (self.iter2 / path.name).write_bytes(path.read_bytes())
        (self.iter2 / "local_compile.json").write_text(json.dumps(
            {"status": "failed", "compile_passed": 0, "cases": 20}))
        provenance = json.loads((self.iter2 / "provenance.json").read_text())
        provenance["validation"]["passed"] = 0
        provenance["validation"]["report_sha256"] = generation_gate.sha256(
            self.iter2 / "local_compile.json")
        (self.iter2 / "provenance.json").write_text(json.dumps(provenance))
        pause = {"source_run_id": 36878888437, "source_changed_during_copy": [],
                 "file_sha256": {str(p.relative_to(self.root)): driver._sha256(p)
                                 for p in self.root.rglob("*")
                                 if p.is_file() and p.name != "PAUSE-SNAPSHOT.json"}}
        (self.root / "PAUSE-SNAPSHOT.json").write_text(json.dumps(pause))
        self.records = driver.load_resume_records(
            self.root, self.catalog, driver.SKILLS_ROOT)

    def test_passing_older_iteration_replayed_before_newer_failure(self):
        records = self.records["exp"]
        self.assertEqual([r["iteration"].name for r in records],
                         ["iter1", "iter2"])
        self.assertEqual(records[0]["recorded_status"], "passed")
        self.assertEqual(records[1]["recorded_status"], "failed")

    def test_second_replay_attempt_qualifies_with_unique_stage_dirs(self):
        def evaluate(op, candidate, iter_dir, task_dir=None):
            replay2 = "reuse-attempt-2" in iter_dir.parts
            passed = replay2
            (iter_dir / "local_compile.json").write_text(json.dumps(
                {"status": "passed" if passed else "failed",
                 "compile_passed": 20 if passed else 0, "cases": 20}))
            return {"hard_failure": not passed,
                    "status": "passed" if passed else "failed",
                    "passed": 20 if passed else 0, "total": 20}

        item = driver.WorkItem("exp", "generate", task_dir=TASK_DIR,
                               contract=self.spec)
        new_run = Path(self.tmp.name) / "newrun"
        new_run.mkdir()
        patches = [
            patch.object(driver, "local_evaluate", side_effect=evaluate),
            patch.object(driver, "SCRATCH_ROOT", Path(self.tmp.name) / "scratch"),
            patch.object(driver, "incumbent_score", return_value=None),
            patch.object(driver.subprocess, "run",
                         return_value=SimpleNamespace(stdout="1.18.33")),
        ]
        with patches[0], patches[1], patches[2], patches[3]:
            result = driver.process_item(
                item, None, new_run, 1, False, "local",
                ("gateway/glm-5.3", "gateway/qwen3.8-max"), 1,
                reuse_record=self.records["exp"])
        self.assertTrue(result.status.startswith("locally qualified"))
        item_dir = new_run / "exp-generate"
        self.assertTrue((item_dir / "reuse-attempt/local_compile.json").is_file())
        self.assertTrue((item_dir / "reuse-attempt-2/local_compile.json").is_file())
        self.assertEqual([h["result"] for h in result.history],
                         ["reuse-replay-fail", "locally-qualified-reuse"])
        qualified = json.loads(
            (new_run / "locally_qualified/provenance/exp.json").read_text())
        self.assertEqual(qualified["reuse"]["original_provenance_sha256"],
                         self.records["exp"][1]["provenance_sha256"])
        self.assertEqual(qualified["reuse"]["interrupted_snapshot"]["source_run_id"],
                         36878888437)


class ResumeArgumentTests(unittest.TestCase):
    def test_resume_and_reuse_are_mutually_exclusive(self):
        argv = ["driver.py", "--items", "all", "--catalog", "missing/catalog",
                "--reuse-generation", "some/path",
                "--resume-interrupted", "other/path"]
        with patch.object(__import__("sys"), "argv", argv):
            with self.assertRaises(SystemExit):
                driver.main()

    def test_resume_requires_catalog(self):
        argv = ["driver.py", "--items", "exp:generate",
                "--resume-interrupted", "some/path"]
        with patch.object(__import__("sys"), "argv", argv):
            with self.assertRaises(SystemExit):
                driver.main()


if __name__ == "__main__":
    unittest.main()
