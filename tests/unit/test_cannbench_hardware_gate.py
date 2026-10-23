import copy
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hardware_gate", ROOT / "integrations/cannbench/ci/hardware_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def catalog():
    return {"benchmark_slug": "official-tasks", "benchmark_version": "1.1.2", "required_operators": 2, "required_cases": 4,
        "operators": [{"operator": n, "function_name": n.lower(), "rel_path": "level1/" + n.lower(), "case_count": 2,
                       "case_ids": ["level1/" + n.lower() + "_1", "level1/" + n.lower() + "_2"]} for n in ("Exp", "Sigmoid")]}


def job(name, index):
    path = "level1/" + name.lower()
    return {"job": {"id": "job_" + str(index), "status": "succeeded", "benchmark_slug": "official-tasks",
        "benchmark_version": "1.1.2", "target_hardware": "950pr", "runner_hardware": "950pr",
        "results": {"result_stage": "final", "contains_performance": True, "anti_cheat_failed_cases": 0,
            "summary": {"geometric_mean_speedup": 999}, "operators": [{"operator": name, "rel_path": path,
                "total_cases": 2, "passed_cases": 2, "cases": [
                    {"case_id": path + "_1", "status": "success", "elapsed_us": 4, "baseline_perf_us": 1, "speedup": 999},
                    {"case_id": path + "_2", "status": "success", "elapsed_us": 1, "baseline_perf_us": 4, "speedup": 999}]}]}}}


class HardwareGateTests(unittest.TestCase):
    def test_complete_jobs_true_gm_from_timings(self):
        report = gate.evaluate(catalog(), [job("Exp", 1), job("Sigmoid", 2)])
        self.assertTrue(report["gate_passed"])
        self.assertAlmostEqual(report["true_geometric_mean_speedup"], 1)
        self.assertEqual(report["correct_cases"], 4)

    def test_missing_operator_keeps_full_denominator(self):
        report = gate.evaluate(catalog(), [job("Exp", 1)])
        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["required_cases"], 4)
        self.assertEqual(report["correct_cases"], 2)
        self.assertIsNone(report["true_geometric_mean_speedup"])

    def test_duplicate_measurements_not_cherry_picked(self):
        report = gate.evaluate(catalog(), [job("Exp", 1), job("Exp", 3), job("Sigmoid", 2)])
        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["measured_cases"], 2)

    def test_bad_timing_and_booleans_fail(self):
        for value in (0, -1, math.inf, math.nan, True, None):
            a = job("Exp", 1)
            a["job"]["results"]["operators"][0]["cases"][0]["elapsed_us"] = value
            with self.subTest(value=value):
                self.assertFalse(gate.evaluate(catalog(), [a, job("Sigmoid", 2)])["gate_passed"])

    def test_wrong_version_and_hardware_do_not_count(self):
        for key, value in (("benchmark_version", "1.1.1"), ("runner_hardware", "910b")):
            a = job("Exp", 1)
            a["job"][key] = value
            report = gate.evaluate(catalog(), [a, job("Sigmoid", 2)])
            self.assertFalse(report["gate_passed"])
            self.assertEqual(report["correct_cases"], 2)

    def test_pending_failed_and_compile_only_not_accepted(self):
        for status in ("queued", "running", "correctness_failed", "compile_failed"):
            a = job("Exp", 1)
            a["job"]["status"] = status
            self.assertFalse(gate.evaluate(catalog(), [a, job("Sigmoid", 2)])["gate_passed"])
        a = job("Exp", 1)
        a["job"]["results"]["contains_performance"] = False
        self.assertFalse(gate.evaluate(catalog(), [a, job("Sigmoid", 2)])["gate_passed"])

    def test_anti_cheat_counts_and_exact_case_set(self):
        mutations = [
            lambda op, result: result.update(anti_cheat_failed_cases=1),
            lambda op, result: op.update(passed_cases=1),
            lambda op, result: op["cases"][0].update(case_id="level1/exp_99"),
            lambda op, result: op["cases"][0].update(case_id="level1/exp_2"),
            lambda op, result: op["cases"].pop(),
            lambda op, result: op["cases"][0].update(status="failed", error_msg="NaN mismatch"),
        ]
        for mutate in mutations:
            a = job("Exp", 1)
            result = a["job"]["results"]
            mutate(result["operators"][0], result)
            self.assertFalse(gate.evaluate(catalog(), [a, job("Sigmoid", 2)])["gate_passed"])

    def test_official_benchmark_wrapper_supported(self):
        a = job("Exp", 1)["job"]
        a["benchmark"] = {"slug": a.pop("benchmark_slug"), "version": a.pop("benchmark_version")}
        self.assertTrue(gate.evaluate(catalog(), [a, job("Sigmoid", 2)])["gate_passed"])

    def test_conflicting_anti_cheat_and_summary_counts_fail_closed(self):
        for key, value in (("anti_cheat_failed_cases", 1), ("passed_cases", 1), ("total_cases", 99)):
            a = job("Exp", 1)
            a["job"]["results"]["summary"][key] = value
            report = gate.evaluate(catalog(), [a, job("Sigmoid", 2)])
            self.assertFalse(report["gate_passed"])
            self.assertEqual(report["operators"][0]["status"], "failed")

    def test_malformed_json_structures_fail_without_crashing(self):
        for payload in ([], {"job": []}, {"job": None}, {"job": {"id": "job_1", "benchmark": []}}):
            self.assertFalse(gate.evaluate(catalog(), [payload])["gate_passed"])
        for value in ("not-an-object", [1]):
            a = job("Exp", 1)
            a["job"]["results"]["operators"][0]["cases"] = value
            self.assertFalse(gate.evaluate(catalog(), [a])["gate_passed"])

    def test_real_catalog_cannot_be_satisfied_by_nine_operator_subset(self):
        import json
        full = json.loads((ROOT / "integrations/cannbench/catalogs/official-tasks-1.1.2-20261001/manifest.json").read_text())
        report = gate.evaluate(full, [])
        self.assertEqual((report["required_operators"], report["required_cases"]), (53, 1060))
        self.assertEqual(len(report["operators"]), 53)
        self.assertFalse(report["gate_passed"])


if __name__ == "__main__":
    unittest.main()
