"""Reuse/requalification of recorded native OpenCode CANNBench candidates."""
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations/cannbench/workers"))
sys.path.insert(0, str(ROOT / "integrations/cannbench/ci"))
driver = importlib.import_module("driver")
generation_gate = importlib.import_module("generation_gate")

TASK_DIR = ROOT / "integrations/cannbench/catalogs/official-tasks-1.1.2-20261001/tasks/exp"
CANDIDATE_SOURCE = "import asctile\nfrom ._pyasc_runtime import ensure_npu_platform\n@asctile.jit\ndef kernel(x):\n    pass\ndef exp(x, base=-1.0, scale=1.0, shift=0.0):\n    ensure_npu_platform()\n    return x\n"
MODELS = {"design": "gateway/glm-5.3", "implement": "gateway/glm-5.3",
          "review": "gateway/qwen3.8-max"}
SESSIONS = {"design": "sess-design", "implement": "sess-implement", "review": "sess-review"}


def make_catalog(names):
    return {
        "catalog_sha256": "catalog-sha-1",
        "operators": [{"function_name": name, "case_count": 20} for name in names],
        "required_operators": len(names),
        "required_cases": 20 * len(names),
    }


def build_recorded_iteration(item_dir, spec, *, status="passed", compile_passed=20):
    iteration = item_dir / "iter1"
    iteration.mkdir(parents=True)
    candidate = iteration / "candidate.py"
    candidate.write_text(CANDIDATE_SOURCE)
    for phase in ("design", "implement", "review"):
        trace = {"skill_gate_passed": True, "session_id": SESSIONS[phase],
                 "model": MODELS[phase]}
        (iteration / (phase + ".skill-trace.attempt1.json")).write_text(json.dumps(trace))
    (iteration / "pre_review_compile.json").write_text(
        json.dumps({"status": status, "compile_passed": compile_passed, "cases": 20}))
    (iteration / "local_compile.json").write_text(
        json.dumps({"status": status, "compile_passed": compile_passed, "cases": 20}))
    provenance = {
        "generator": "opencode",
        "evaluation_mode": "local",
        "skill_gate_passed": True,
        "candidate_sha256": generation_gate.sha256(candidate),
        "task_contract": spec,
        "models": dict(MODELS),
        "sessions": dict(SESSIONS),
        "loaded_skills": list(driver.REQUIRED_SKILLS),
        "skill_sources": {
            name: {"path": str(driver.SKILLS_ROOT / name),
                   "files": driver._skill_source_manifest(name)}
            for name in driver.REQUIRED_SKILLS
        },
        "reviewed_compile": {
            "candidate_sha256": generation_gate.sha256(candidate),
            "report_sha256": generation_gate.sha256(iteration / "pre_review_compile.json"),
        },
        "validation": {
            "label": "verified-local-compile",
            "report_sha256": generation_gate.sha256(iteration / "local_compile.json"),
            "passed": compile_passed,
            "total": 20,
        },
    }
    (iteration / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return iteration, provenance


class ReuseLoadingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.reuse_root = self.root / "previous"
        self.reuse_root.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def write_summary(self, requested, catalog_sha="catalog-sha-1"):
        (self.reuse_root / "summary.json").write_text(json.dumps({
            "requested_operators": requested,
            "locally_qualified_operators": [],
            "catalog_sha256": catalog_sha,
        }))

    def load(self, catalog):
        return driver.load_reuse_records(self.reuse_root, catalog, driver.SKILLS_ROOT)

    def test_historical_subset_parent_summary_rejected(self):
        self.write_summary(["exp"])  # catalog covers exp + gelu
        with self.assertRaises(ValueError):
            self.load(make_catalog(["exp", "gelu"]))

    def test_parent_catalog_sha_mismatch_rejected(self):
        self.write_summary(["exp"], catalog_sha="other-catalog")
        with self.assertRaises(ValueError):
            self.load(make_catalog(["exp"]))

    def test_hand_authored_tree_without_summary_rejected(self):
        with self.assertRaises(ValueError):
            self.load(make_catalog(["exp"]))

    def test_tampered_candidate_source_skipped(self):
        catalog = make_catalog(["exp"])
        self.write_summary(["exp"])
        iteration, _ = build_recorded_iteration(
            self.reuse_root / "exp-generate", catalog["operators"][0])
        (iteration / "candidate.py").write_text("tampered source")
        self.assertEqual(self.load(catalog), {})

    def test_stale_skill_source_hash_skipped(self):
        catalog = make_catalog(["exp"])
        self.write_summary(["exp"])
        iteration, provenance = build_recorded_iteration(
            self.reuse_root / "exp-generate", catalog["operators"][0])
        provenance["skill_sources"]["pyasc-cannbench-kernel"]["files"]["SKILL.md"] = "0" * 64
        (iteration / "provenance.json").write_text(json.dumps(provenance))
        self.assertEqual(self.load(catalog), {})

    def test_missing_native_phase_trace_skipped(self):
        catalog = make_catalog(["exp"])
        self.write_summary(["exp"])
        iteration, _ = build_recorded_iteration(
            self.reuse_root / "exp-generate", catalog["operators"][0])
        (iteration / "review.skill-trace.attempt1.json").unlink()
        self.assertEqual(self.load(catalog), {})

    def test_static_contract_failure_without_report_skipped(self):
        catalog = make_catalog(["exp"])
        self.write_summary(["exp"])
        iteration, provenance = build_recorded_iteration(
            self.reuse_root / "exp-generate", catalog["operators"][0])
        provenance.pop("validation")
        (iteration / "provenance.json").write_text(json.dumps(provenance))
        (iteration / "local_compile.json").unlink()
        self.assertEqual(self.load(catalog), {})

    def test_recorded_compiler_failure_replayable_but_not_qualified(self):
        catalog = make_catalog(["exp"])
        spec = catalog["operators"][0]
        self.write_summary(["exp"])
        iteration, _ = build_recorded_iteration(
            self.reuse_root / "exp-generate", spec,
            status="failed", compile_passed=3)
        records = self.load(catalog)
        self.assertIn("exp", records)
        with self.assertRaises(ValueError):
            generation_gate.validate_iteration_record(spec, iteration, driver.SKILLS_ROOT)


class RequalificationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.catalog = make_catalog(["exp"])
        self.spec = self.catalog["operators"][0]
        self.reuse_root = self.root / "previous"
        self.reuse_root.mkdir()
        (self.reuse_root / "summary.json").write_text(json.dumps({
            "requested_operators": ["exp"],
            "locally_qualified_operators": ["exp"],
            "catalog_sha256": "catalog-sha-1",
        }))
        self.recorded_iteration, _ = build_recorded_iteration(
            self.reuse_root / "exp-generate", self.spec)
        self.records = driver.load_reuse_records(
            self.reuse_root, self.catalog, driver.SKILLS_ROOT)

    def tearDown(self):
        self._tmp.cleanup()

    def run_item(self, evaluate, **overrides):
        item = driver.WorkItem("exp", "generate", task_dir=TASK_DIR,
                               contract=self.spec)
        new_run = self.root / "newrun"
        new_run.mkdir()
        patches = [
            patch.object(driver, "local_evaluate", side_effect=evaluate),
            patch.object(driver, "SCRATCH_ROOT", self.root / "scratch"),
            patch.object(driver, "incumbent_score", return_value=None),
            patch.object(driver.subprocess, "run",
                         return_value=SimpleNamespace(stdout="1.18.33")),
        ]
        patches.extend(patch.object(driver, name, value) for name, value in overrides.items())
        with patches[0], patches[1], patches[2], patches[3]:
            for extra in patches[4:]:
                extra.start()
            try:
                result = driver.process_item(
                    item, None, new_run, 1, False, "local",
                    ("gateway/glm-5.3", "gateway/qwen3.8-max"), 1,
                    reuse_record=self.records.get("exp"))
            finally:
                for extra in patches[4:]:
                    extra.stop()
        return result, new_run

    def test_successful_unchanged_candidate_requalified(self):
        def evaluate(op, candidate, iter_dir, task_dir=None):
            (iter_dir / "local_compile.json").write_text(json.dumps(
                {"status": "passed", "compile_passed": 20, "cases": 20}))
            return {"hard_failure": False, "status": "passed",
                    "passed": 20, "total": 20}

        result, new_run = self.run_item(evaluate)
        self.assertTrue(result.status.startswith("locally qualified"))
        qualified = new_run / "locally_qualified/cann_bench/exp.py"
        self.assertEqual(generation_gate.sha256(qualified),
                         self.records["exp"]["candidate_sha256"])
        self.assertEqual(qualified.read_text(), CANDIDATE_SOURCE)
        # Source campaign remains immutable.
        self.assertEqual(
            generation_gate.sha256(self.recorded_iteration / "provenance.json"),
            self.records["exp"]["provenance_sha256"])
        provenance = json.loads(
            (new_run / "locally_qualified/provenance/exp.json").read_text())
        reuse = provenance["reuse"]
        self.assertEqual(reuse["mode"], "requalified-replay")
        self.assertEqual(reuse["original_provenance_sha256"],
                         self.records["exp"]["provenance_sha256"])
        self.assertEqual(reuse["original_compiler_report_sha256"],
                         self.records["exp"]["compiler_report_sha256"])
        self.assertEqual(reuse["original_validation"]["report_sha256"],
                         self.records["exp"]["compiler_report_sha256"])
        self.assertEqual(provenance["models"], self.records["exp"]["models"])
        self.assertEqual(provenance["sessions"], SESSIONS)
        new_report = new_run / "exp-generate/iter1/local_compile.json"
        self.assertEqual(provenance["validation"]["report_sha256"],
                         generation_gate.sha256(new_report))
        self.assertTrue(provenance["validation"]["replay"])
        # The new iteration itself satisfies the strict full gate.
        record = generation_gate.validate_iteration_record(
            self.spec, new_run / "exp-generate/iter1", driver.SKILLS_ROOT)
        self.assertEqual(record["candidate_sha256"],
                         self.records["exp"]["candidate_sha256"])
        (new_run / "summary.json").write_text(json.dumps({
            "requested_operators": ["exp"],
            "locally_qualified_operators": ["exp"],
            "catalog_sha256": "catalog-sha-1",
        }))
        self.assertIn("exp", generation_gate.validate_generation(
            self.catalog, new_run, driver.SKILLS_ROOT))
        original = new_run / "exp-generate/iter1/original-record/provenance.json"
        original.write_text(original.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "derivation mismatch"):
            generation_gate.validate_generation(self.catalog, new_run, driver.SKILLS_ROOT)

    def test_replay_failure_falls_back_to_fresh_generation(self):
        def evaluate(op, candidate, iter_dir, task_dir=None):
            replay = "reuse-attempt" in iter_dir.parts
            passed = not replay
            (iter_dir / "local_compile.json").write_text(json.dumps(
                {"status": "passed" if passed else "failed",
                 "compile_passed": 20 if passed else 3, "cases": 20}))
            return {"hard_failure": not passed,
                    "status": "passed" if passed else "failed",
                    "passed": 20 if passed else 3, "total": 20}

        def phase(scratch, name, iteration, models, index, attempts):
            self.assertIn("Recorded candidate replay failed", (scratch / "task.md").read_text())
            if name == "design":
                (scratch / "design.md").write_text("design")
            else:
                (scratch / "candidate.py").write_text("fresh " + name)
            return (driver.WorkerResult(0, name, driver.PHASE_SKILLS[name], (),
                                        name.upper() + "_DONE"), models[index])

        result, new_run = self.run_item(
            evaluate, run_phase=phase, static_check=lambda *a: [])
        self.assertTrue(result.status.startswith("locally qualified"))
        self.assertEqual([h["result"] for h in result.history],
                         ["reuse-replay-fail", "locally-qualified"])
        fresh = (new_run / "locally_qualified/cann_bench/exp.py").read_text()
        self.assertEqual(fresh, "fresh review")
        provenance = json.loads(
            (new_run / "locally_qualified/provenance/exp.json").read_text())
        self.assertNotIn("reuse", provenance)
        self.assertTrue(
            (new_run / "exp-generate/reuse-attempt/local_compile.json").is_file())


class ReuseArgumentTests(unittest.TestCase):
    def test_reuse_requires_catalog(self):
        with patch.object(sys, "argv", ["driver.py", "--items", "exp:generate",
                                        "--reuse-generation", "some/path"]):
            with self.assertRaises(SystemExit):
                driver.main()

    def test_reuse_requires_local_evaluation(self):
        with patch.object(sys, "argv", ["driver.py", "--items", "all",
                                        "--catalog", "missing/catalog",
                                        "--evaluation", "remote",
                                        "--reuse-generation", "some/path"]):
            with self.assertRaises(SystemExit):
                driver.main()


if __name__ == "__main__":
    unittest.main()
