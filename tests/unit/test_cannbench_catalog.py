import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("catalog", ROOT / "integrations/cannbench/ci/catalog.py")
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)
spec = importlib.util.spec_from_file_location("validate_catalog", ROOT / "integrations/cannbench/ci/validate_catalog.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class Client:
    def __init__(self, drift=False):
        self.calls = 0
        self.drift = drift

    def get(self, route):
        if route == "/api/benchmarks":
            self.calls += 1
            return {"benchmarks": [{"slug": "official-tasks", "submission_version":
                "2" if self.drift and self.calls > 1 else "1", "accepted_submission_versions": ["1", "2"],
                "operator_count": 1, "total_case_count": 2, "multi_operators": [{
                "operator": "Exp", "function_name": "exp", "rel_path": "level1/exp", "case_count": 2}]}]}
        if route != "/api/benchmarks/official-tasks/operators/Exp":
            raise AssertionError(route)
        return {"benchmark_slug": "official-tasks", "operator": "Exp", "function_name": "exp",
            "proto": {"schema": "exp(Tensor x) -> Tensor"}, "cases_yaml": [{"case_id": 1}, {"case_id": 2}],
            "golden_source": "def exp(x): return x\n", "desc_md": "Exp"}


class CatalogTests(unittest.TestCase):
    def test_full_contract_hashes_and_case_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "snapshot"
            result = catalog.snapshot(Client(), "official-tasks", destination)
            self.assertEqual(result["operators"][0]["case_ids"], ["level1/exp_1", "level1/exp_2"])
            for name, digest in result["operators"][0]["task_files"].items():
                self.assertEqual(hashlib.sha256((destination / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(json.loads((destination / "manifest.json").read_text()), result)

    def test_catalog_drift_is_not_qualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "snapshot"
            with self.assertRaisesRegex(ValueError, "changed"):
                catalog.snapshot(Client(drift=True), "official-tasks", destination)
            self.assertFalse((destination / "manifest.json").exists())

    def test_rehashed_edited_golden_cannot_replace_official_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "snapshot"
            manifest = catalog.snapshot(Client(), "official-tasks", destination)
            validator.validate(destination)
            name = "tasks/exp/golden.py"
            (destination / name).write_text("def exp(x): return None\n")
            manifest["operators"][0]["task_files"][name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
            (destination / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "raw official task response"):
                validator.validate(destination)

    def test_count_and_duplicate_inventory_rejected(self):
        doc = Client().get("/api/benchmarks")
        doc["benchmarks"][0]["operator_count"] = 2
        with self.assertRaisesRegex(ValueError, "operator count"):
            catalog.select_catalog(doc, "official-tasks")
        doc["benchmarks"][0]["multi_operators"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            catalog.select_catalog(doc, "official-tasks")

    def test_historical_snapshot_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "must be new"):
                catalog.snapshot(Client(), "official-tasks", Path(tmp))


if __name__ == "__main__":
    unittest.main()
