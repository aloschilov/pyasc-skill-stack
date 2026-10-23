#!/usr/bin/env python3
"""Unit tests for run_camodel_cases pure glue (no torch/asc required)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]
        / "integrations" / "cannbench" / "workers"),
)

import run_camodel_cases as runner  # noqa: E402


class FakeTensor:
    def __init__(self, dtype="torch.float32", shape=(4,), data=None):
        self.dtype = dtype
        self.shape = shape
        self.data = list(data) if data is not None else [0.0]
        self.doubled = False

    # clone_inputs protocol -------------------------------------------------
    def is_floating_point(self):
        return self.dtype in ("torch.float16", "torch.float32",
                              "torch.float64", "torch.bfloat16")

    def detach(self):
        return self

    def clone(self):
        copied = FakeTensor(self.dtype, self.shape, self.data)
        return copied

    def double(self):
        copied = FakeTensor("torch.float64", self.shape, self.data)
        copied.doubled = True
        return copied

    # nan_inf_report protocol ------------------------------------------------
    def numel(self):
        return len(self.data)


class FakeBoolMask:
    def __init__(self, values):
        self.values = list(values)

    def sum(self):
        return sum(1 for value in self.values if value)

    def __ne__(self, other):
        return FakeBoolMask([a != b for a, b in zip(self.values,
                                                    other.values)])


class FakeTorch:
    Tensor = FakeTensor

    @staticmethod
    def isnan(tensor):
        import math
        return FakeBoolMask([math.isnan(v) for v in tensor.data])

    @staticmethod
    def isinf(tensor):
        import math
        return FakeBoolMask([math.isinf(v) for v in tensor.data])


def _golden(query: "torch.Tensor", key: "torch.Tensor",
            layout: int = 0, mode: str = "half"):
    return query


def _golden_with_attr(query: "torch.Tensor", scale: float):
    return query


class PositionalParamsTest(unittest.TestCase):
    def test_tensor_order_defaults_and_none_fill(self):
        tensors = [FakeTensor(), FakeTensor()]
        params = runner.positional_params_from_signature(
            _golden, tensors, {"layout": 1})
        self.assertIs(params["query"], tensors[0])
        self.assertIs(params["key"], tensors[1])
        self.assertEqual(params["layout"], 1)  # attr wins over default
        self.assertEqual(params["mode"], "half")  # default fills rest

    def test_exhausted_tensors_become_none(self):
        params = runner.positional_params_from_signature(
            _golden, [FakeTensor()], {})
        self.assertIsNone(params["key"])
        self.assertEqual(params["layout"], 0)

    def test_attr_without_default_must_be_present(self):
        params = runner.positional_params_from_signature(
            _golden_with_attr, [FakeTensor()], {"scale": 0.5})
        self.assertEqual(params["scale"], 0.5)
        # Missing attr without default is omitted (official behaviour: the
        # golden call then fails loudly, which the child records).
        params = runner.positional_params_from_signature(
            _golden_with_attr, [FakeTensor()], {})
        self.assertNotIn("scale", params)


class UpdateParamsIdentityTest(unittest.TestCase):
    def test_single_tensor_positional_swap_skips_none(self):
        original = [FakeTensor()]
        params = {"a": original[0], "b": 3}
        tensors = [None, FakeTensor(), FakeTensor()]
        updated = runner.update_params_identity(params, tensors, FakeTensor)
        self.assertIs(updated["a"], tensors[1])
        self.assertEqual(updated["b"], 3)

    def test_exhausted_single_tensor_keeps_value(self):
        value = FakeTensor()
        updated = runner.update_params_identity(
            {"a": value}, [], FakeTensor)
        self.assertIs(updated["a"], value)

    def test_list_param_wrap_and_replace(self):
        item = FakeTensor()
        updated = runner.update_params_identity(
            {"xs": [FakeTensor(), FakeTensor()]}, [item], FakeTensor)
        self.assertEqual(updated["xs"], [item])
        listed = [FakeTensor(), FakeTensor()]
        updated = runner.update_params_identity(
            {"xs": [FakeTensor()]}, [listed], FakeTensor)
        self.assertIs(updated["xs"], listed)

    def test_list_exhausted_keeps_value_and_empty_list_untouched(self):
        original = [FakeTensor()]
        updated = runner.update_params_identity(
            {"xs": original, "ys": []}, [], FakeTensor)
        self.assertIs(updated["xs"], original)
        self.assertEqual(updated["ys"], [])


class DetermineDtypeTest(unittest.TestCase):
    def test_prefers_candidate_output_dtype(self):
        self.assertEqual(
            runner.determine_dtype(FakeTensor("torch.bfloat16"),
                                   [["float32"]], ["float16"]),
            "bfloat16")

    def test_list_outputs_use_first(self):
        self.assertEqual(
            runner.determine_dtype([FakeTensor("torch.float16")], [], []),
            "float16")

    def test_falls_back_to_op_then_case_dtype(self):
        self.assertEqual(runner.determine_dtype(None, [["int32"]], []),
                         "int32")
        self.assertEqual(runner.determine_dtype(None, [], ["float64"]),
                         "float64")
        self.assertEqual(runner.determine_dtype(None, [], []), "float32")


class ThresholdAndIgnoreTest(unittest.TestCase):
    def test_merge_prefers_operator_values(self):
        base = {"float32": 0.1, "float16": 0.2}
        self.assertEqual(
            runner.merged_thresholds(base, {"float32": 0.05}),
            {"float32": 0.05, "float16": 0.2})
        self.assertEqual(runner.merged_thresholds(base, {}), base)
        self.assertEqual(runner.merged_thresholds(base, None), base)

    def test_ignore_indices_from_compare_flag(self):
        outputs = [
            type("Out", (), {"compare": True})(),
            type("Out", (), {"compare": False})(),
            type("Out", (), {"compare": False})(),
        ]
        self.assertEqual(runner.ignore_output_indices(outputs), [1, 2])
        self.assertEqual(runner.ignore_output_indices(None), [])


class SerializationTest(unittest.TestCase):
    def test_sanitize_dataclass_and_containers(self):
        from dataclasses import dataclass

        @dataclass
        class Row:
            name: str
            values: list

        payload = runner._sanitize_json(
            {"row": Row("a", [1, None]), "odd": object()})
        self.assertEqual(payload["row"], {"name": "a", "values": [1, None]})
        self.assertIsInstance(payload["odd"], str)
        json.dumps(payload)  # must be JSON-serialisable

    def test_atomic_write_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "result.json"
            runner._atomic_write_json(path, {"status": "passed"})
            self.assertEqual(json.loads(path.read_text())["status"], "passed")
            self.assertFalse(list(path.parent.glob("*.tmp.*")))


class KernelEvalImportTest(unittest.TestCase):
    """The stub-package import must bypass torch_npu-heavy __init__ files."""

    def _write_package(self, root: Path) -> None:
        package = root / "kernel_eval"
        (package / "data").mkdir(parents=True)
        (package / "utils").mkdir(parents=True)
        (package / "eval").mkdir(parents=True)
        (package / "security").mkdir(parents=True)
        package.joinpath("__init__.py").write_text(
            "from .config import Config\n", encoding="utf-8")
        package.joinpath("config.py").write_text(
            "class Config:\n"
            "    def __init__(self):\n"
            "        self.precision_thresholds = {'float32': 2**-13}\n",
            encoding="utf-8")
        for sub in ("data", "utils", "eval", "security"):
            package.joinpath(sub, "__init__.py").write_text(
                "raise RuntimeError('heavy init must not execute')\n",
                encoding="utf-8")
        package.joinpath("utils", "naming.py").write_text(
            "def camel_to_snake(value):\n    return value.lower()\n",
            encoding="utf-8")
        package.joinpath("utils", "dtype_mapper.py").write_text(
            "DTYPES = {'float32': 'float32'}\n", encoding="utf-8")
        package.joinpath("data", "case_loader.py").write_text(
            "from ..utils.naming import camel_to_snake\n"
            "class CaseLoader:\n"
            "    def __init__(self, root):\n"
            "        self.root = camel_to_snake(root)\n",
            encoding="utf-8")

    def test_stub_import_executes_only_leaf_modules(self):
        leaves_backup = runner.KERNEL_EVAL_LEAVES
        runner.KERNEL_EVAL_LEAVES = (
            "kernel_eval.utils.naming",
            "kernel_eval.utils.dtype_mapper",
            "kernel_eval.data.case_loader",
            "kernel_eval.config",
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write_package(root)
                kem = runner.KernelEvalModules(root)
                loader_cls = kem.modules[
                    "kernel_eval.data.case_loader"].CaseLoader
                self.assertEqual(loader_cls("ABC").root, "abc")
                self.assertEqual(
                    kem.modules["kernel_eval.config"].Config()
                    .precision_thresholds,
                    {"float32": 2 ** -13})
                self.assertEqual(
                    len(kem.file_sha256), len(runner.KERNEL_EVAL_LEAVES) + 1)
        finally:
            runner.KERNEL_EVAL_LEAVES = leaves_backup
            for name in list(sys.modules):
                if name == "kernel_eval" or name.startswith("kernel_eval."):
                    del sys.modules[name]
            sys.path.remove(str(root.resolve()))


class CloneInputsTest(unittest.TestCase):
    def test_clones_are_storage_isolated_and_structure_preserved(self):
        float_tensor = FakeTensor("torch.float32", (2,), [1.0, 2.0])
        int_tensor = FakeTensor("torch.int64", (2,), [1, 2])
        nested = [FakeTensor("torch.float16", (1,), [3.0])]
        clones = runner.clone_inputs(
            [float_tensor, int_tensor, nested, None, "attr"],
            promote_fp64=False, torch=FakeTorch)
        self.assertIsNot(clones[0], float_tensor)
        self.assertIsNot(clones[1], int_tensor)  # ints must not alias
        self.assertIsNot(clones[2][0], nested[0])
        self.assertIsNone(clones[3])
        self.assertEqual(clones[4], "attr")
        self.assertEqual(clones[0].data, float_tensor.data)
        self.assertEqual(clones[1].data, int_tensor.data)

    def test_fp64_promotion_only_for_floats(self):
        clones = runner.clone_inputs(
            [FakeTensor("torch.float32", (1,), [1.5]),
             FakeTensor("torch.int32", (1,), [7])],
            promote_fp64=True, torch=FakeTorch)
        self.assertTrue(clones[0].doubled)
        self.assertEqual(clones[0].dtype, "torch.float64")
        self.assertFalse(clones[1].doubled)
        self.assertEqual(clones[1].dtype, "torch.int32")


class NanInfReportTest(unittest.TestCase):
    def test_reports_counts_and_position_mismatches(self):
        actual = FakeTensor("torch.float32", (4,),
                            [float("nan"), 1.0, float("inf"), 2.0])
        expected = FakeTensor("torch.float32", (4,),
                              [float("nan"), 1.0, 3.0, float("inf")])
        report = runner.nan_inf_report(actual, expected, torch=FakeTorch)
        self.assertEqual(len(report), 1)
        entry = report[0]
        self.assertEqual(entry["actual_nan_count"], 1)
        self.assertEqual(entry["expected_nan_count"], 1)
        self.assertEqual(entry["actual_inf_count"], 1)
        self.assertEqual(entry["expected_inf_count"], 1)
        self.assertEqual(entry["nan_position_mismatches"], 0)
        self.assertEqual(entry["inf_position_mismatches"], 2)

    def test_skips_shape_mismatch_and_large_tensors(self):
        small = FakeTensor("torch.float32", (1,), [1.0])
        other_shape = FakeTensor("torch.float32", (2, 2), [1.0] * 4)
        report = runner.nan_inf_report([small], [other_shape],
                                       torch=FakeTorch)
        self.assertEqual(report[0]["skipped"], "shape_mismatch")
        big = FakeTensor("torch.float32", (10,), [1.0] * 10)
        report = runner.nan_inf_report([big], [big], torch=FakeTorch,
                                       max_elements=4)
        self.assertEqual(report[0]["skipped"], "too_large")

    def test_non_tensor_outputs_recorded(self):
        report = runner.nan_inf_report([None], [None], torch=FakeTorch)
        self.assertEqual(report[0]["skipped"], "non_tensor_output")


class ManifestEnumerationTest(unittest.TestCase):
    """run_parent must match manifest function_name (snake_case dir), not
    the PascalCase "operator" display name."""

    def test_parent_uses_function_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "catalog"
            (catalog / "tasks").mkdir(parents=True)
            manifest = {"operators": [
                {"operator": "Sigmoid", "function_name": "sigmoid"},
                {"operator": "Exp", "function_name": "exp"},
            ]}
            (catalog / "manifest.json").write_text(json.dumps(manifest))

            class _FakeCaseLoader:
                def __init__(self, root):
                    self.root = root

                def scan_all_cases(self):
                    return []

            fake_kem = SimpleNamespace(
                modules={"kernel_eval.data.case_loader":
                         SimpleNamespace(CaseLoader=_FakeCaseLoader)},
                file_sha256={},
            )
            args = SimpleNamespace(
                catalog=catalog, candidate_root=root, kernel_eval_src=root,
                output_dir=root / "out", ops="sigmoid", timeout_seconds=1,
                max_rss_gb=1, platform="x", seed=None, force=False,
                expect_pyasc_version=None, expect_pyasc_source=None)
            with patch.object(runner, "KernelEvalModules",
                              return_value=fake_kem), \
                    patch.object(runner, "create_case_loader", side_effect=lambda kem, root: kem.modules["kernel_eval.data.case_loader"].CaseLoader(str(root))):
                rc = runner.run_parent(args)
            self.assertEqual(rc, 0)  # would have raised "not in manifest"
            self.assertTrue((root / "out" / "summary.json").exists())

    def test_parent_rejects_pascal_case_operator_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "catalog"
            (catalog / "tasks").mkdir(parents=True)
            (catalog / "manifest.json").write_text(json.dumps(
                {"operators": [{"operator": "Sigmoid",
                                "function_name": "sigmoid"}]}))

            fake_kem = SimpleNamespace(
                modules={"kernel_eval.data.case_loader":
                         lambda root: SimpleNamespace(
                             scan_all_cases=lambda: [])},
                file_sha256={},
            )
            args = SimpleNamespace(
                catalog=catalog, candidate_root=root, kernel_eval_src=root,
                output_dir=root / "out", ops="Sigmoid", timeout_seconds=1,
                max_rss_gb=1, platform="x", seed=None, force=False,
                expect_pyasc_version=None, expect_pyasc_source=None)
            with patch.object(runner, "KernelEvalModules",
                              return_value=fake_kem), \
                    patch.object(runner, "create_case_loader", side_effect=lambda kem, root: kem.modules["kernel_eval.data.case_loader"].CaseLoader(str(root))):
                with self.assertRaises(ValueError):
                    runner.run_parent(args)


class RuntimeProvenanceTest(unittest.TestCase):
    def test_records_fields_without_runtime(self):
        record = runner.runtime_provenance()
        self.assertIn("asc_version", record)
        self.assertIn("extension_sha256", record)
        self.assertNotIn("pin_mismatch", record)

    def test_expected_pin_mismatch_raises(self):
        with self.assertRaises(RuntimeError):
            runner.runtime_provenance(expect_version="definitely-absent")


class EnforcePinsTest(unittest.TestCase):
    PIN = "0a631f70968c3cb7c33ce45330a85768dd5a6f06"

    def _record(self, asc="1.1.1+g0a631f70", asctile="1.1.1+g0a631f70",
                path="/home/u/native-runtime-0a631f70/venv/lib/python3.11/"
                     "site-packages/asctile/__init__.py"):
        return {"asc_version": asc, "asctile_version": asctile,
                "asctile_file": path}

    def test_full_source_pin_matches_via_short_hash(self):
        record = self._record()
        runner._enforce_pins(record, None, self.PIN)
        self.assertNotIn("pin_mismatch", record)
        self.assertIn("0a631f70", record["observed_version_stamp"])

    def test_version_pin_matches(self):
        record = self._record()
        runner._enforce_pins(record, "1.1.1+g0a631f70", None)
        self.assertNotIn("pin_mismatch", record)

    def test_source_mismatch_raises(self):
        with self.assertRaises(RuntimeError):
            runner._enforce_pins(
                self._record(), None,
                "deadbeefcafebabe1234567890deadbeefcafeba")

    def test_install_path_alone_does_not_satisfy_pin(self):
        # Hash appears only in the install path, not in any version string:
        # the check must still fail (no path-substring tautology).
        record = self._record(asc="1.1.1", asctile="1.1.1")
        with self.assertRaises(RuntimeError):
            runner._enforce_pins(record, None, self.PIN)

    def test_no_pins_is_noop(self):
        record = self._record(asc="1.1.1", asctile="1.1.1")
        runner._enforce_pins(record, None, None)
        self.assertNotIn("pin_mismatch", record)


class ClassifyTimeoutTest(unittest.TestCase):
    def test_lock_wait_is_rerun_pending_not_numerical_failure(self):
        status, klass, rerun = runner.classify_timeout("lock_wait")
        self.assertEqual(status, "lock_wait_timeout")
        self.assertEqual(klass, "lock_wait_timeout")
        self.assertTrue(rerun)

    def test_compute_phases_are_compute_timeout(self):
        for phase in ("contract", "generate", "get_input", "params",
                      "golden", "cpu_native"):
            self.assertEqual(
                runner.classify_timeout(phase),
                ("timeout", "compute_timeout", False), phase)

    def test_candidate_phases_are_candidate_timeout(self):
        for phase in ("candidate", "compare", "done"):
            self.assertEqual(
                runner.classify_timeout(phase),
                ("timeout", "candidate_timeout", False), phase)

    def test_missing_marker_is_generic_timeout(self):
        self.assertEqual(
            runner.classify_timeout(None),
            ("timeout", "timeout", False))


class PhaseMarkerTest(unittest.TestCase):
    def test_marker_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "case_0001.json"
            runner._write_phase_marker(result, "lock_wait")
            self.assertEqual(runner._read_phase_marker(result), "lock_wait")
            runner._write_phase_marker(result, "candidate")
            self.assertEqual(runner._read_phase_marker(result), "candidate")

    def test_phase_marker_creates_fresh_operator_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "results" / "arg_max" / "case_0013.json"
            runner._write_phase_marker(result, "candidate")
            self.assertEqual(runner._read_phase_marker(result), "candidate")
            self.assertFalse(result.exists())
            self.assertEqual(runner.classify_timeout(runner._read_phase_marker(result))[1], "candidate_timeout")

    def test_missing_marker_reads_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                runner._read_phase_marker(Path(tmp) / "case.json"))


class FdLimitTest(unittest.TestCase):
    def test_raises_soft_limit_to_hard(self):
        import resource

        original = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            resource.setrlimit(
                resource.RLIMIT_NOFILE, (min(64, original[0]), original[1]))
            runner._set_fd_limit()
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            self.assertIn(soft, (hard, 1 << 20))
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, original)


class StaleResultRecycleTest(unittest.TestCase):
    """A child that dies before writing (e.g. simulator SIGABRT) must never
    be reported from a stale pre-existing result file."""

    def test_crashed_child_does_not_recycle_stale_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "catalog"
            (catalog / "tasks").mkdir(parents=True)
            (catalog / "manifest.json").write_text(json.dumps(
                {"operators": [{"operator": "Sigmoid",
                                "function_name": "sigmoid"}]}))
            op_dir = root / "out" / "sigmoid"
            op_dir.mkdir(parents=True)
            (op_dir / "case_0007.json").write_text(
                json.dumps({"status": "passed"}))
            (op_dir / "case_0007.json.phase").write_text(
                json.dumps({"phase": "done"}))

            class _FakeCaseLoader:
                def __init__(self, root):
                    pass

                def scan_all_cases(self):
                    return [SimpleNamespace(case_id=7, rel_path="sigmoid")]

            fake_kem = SimpleNamespace(
                modules={"kernel_eval.data.case_loader":
                         SimpleNamespace(CaseLoader=_FakeCaseLoader)},
                file_sha256={},
            )

            def _crashed_popen(command, **kwargs):
                return SimpleNamespace(
                    returncode=-6,
                    communicate=lambda timeout=None: ("", ""))

            args = SimpleNamespace(
                catalog=catalog, candidate_root=root, kernel_eval_src=root,
                output_dir=root / "out", ops="sigmoid", timeout_seconds=1,
                max_rss_gb=1, platform="x", seed=None, force=True,
                expect_pyasc_version=None, expect_pyasc_source=None)
            with patch.object(runner, "KernelEvalModules",
                              return_value=fake_kem), \
                    patch.object(runner, "create_case_loader", side_effect=lambda kem, root: kem.modules["kernel_eval.data.case_loader"].CaseLoader(str(root))), \
                    patch.object(runner, "run_bounded_case",
                                 return_value=(-6, "", "", False, None)):
                rc = runner.run_parent(args)
            record = json.loads((op_dir / "case_0007.json").read_text())
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["failure_class"], "harness_error")
            self.assertEqual(record["returncode"], -6)
            self.assertEqual(rc, 1)


class LoadCandidateFuncTest(unittest.TestCase):
    def _cleanup(self, *names):
        for name in names:
            sys.modules.pop(name, None)

    def test_flat_package_import_with_runtime_shim(self):
        # In-progress campaign layout: no __init__.py, no _pyasc_runtime.py.
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "cb_flat_test"
            pkg.mkdir()
            (pkg / "myop.py").write_text(
                "from ._pyasc_runtime import ensure_npu_platform\n"
                "def myop():\n    return ensure_npu_platform\n",
                encoding="utf-8")
            try:
                func = runner.load_candidate_func(pkg, "myop")
            finally:
                self._cleanup("cb_flat_test", "cb_flat_test.myop",
                              "cb_flat_test._pyasc_runtime")
        self.assertTrue(callable(func))
        # The imported reference is the inert shim (no-op).
        self.assertIsNone(func()())
        self.assertIn("ensure_npu_platform", func.__globals__)

    def test_completed_package_reexport_preferred(self):
        # Completed campaign layout: __init__.py re-exports, real runtime.
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "cb_full_test"
            pkg.mkdir()
            (pkg / "_pyasc_runtime.py").write_text(
                "def ensure_npu_platform():\n    return 'real'\n",
                encoding="utf-8")
            (pkg / "myop.py").write_text(
                "from ._pyasc_runtime import ensure_npu_platform\n"
                "def myop():\n    return ensure_npu_platform\n",
                encoding="utf-8")
            (pkg / "__init__.py").write_text(
                "from .myop import myop\n", encoding="utf-8")
            try:
                func = runner.load_candidate_func(pkg, "myop")
            finally:
                self._cleanup("cb_full_test", "cb_full_test.myop",
                              "cb_full_test._pyasc_runtime")
        self.assertEqual(func()(), "real")

    def test_missing_op_raises_attribute_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "cb_empty_test"
            pkg.mkdir()
            try:
                with self.assertRaises((AttributeError, ModuleNotFoundError)):
                    runner.load_candidate_func(pkg, "nosuchop")
            finally:
                self._cleanup("cb_empty_test", "cb_empty_test.nosuchop",
                              "cb_empty_test._pyasc_runtime")


class TerminateProcessGroupTest(unittest.TestCase):
    def test_descendants_do_not_survive(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "grandchild.pid"
            child = subprocess.Popen(
                [sys.executable, "-c",
                 "import subprocess, time, pathlib; "
                 f"g = subprocess.Popen(['sleep', '60']); "
                 f"pathlib.Path({str(pid_file)!r}).write_text(str(g.pid)); "
                 "time.sleep(60)"],
                start_new_session=True)
            deadline = 30.0
            while not pid_file.exists():
                time.sleep(0.05)
                deadline -= 0.05
                if deadline <= 0:
                    child.kill()
                    self.fail("grandchild did not start")
            grandchild_pid = int(pid_file.read_text())
            runner.terminate_process_group(child, grace_seconds=1.0)
            child.wait(timeout=10)
            self.assertIsNotNone(child.returncode)
            with self.assertRaises(ProcessLookupError):
                os.kill(grandchild_pid, 0)


if __name__ == "__main__":
    unittest.main()

class ParentScratchTest(unittest.TestCase):
    def test_timeout_reclaims_simulator_scratch(self):
        with tempfile.TemporaryDirectory() as root:
            command = [sys.executable, '-c',
                       'import os,pathlib,time; '
                       'pathlib.Path(os.environ["CAMODEL_LOG_PATH"],"trace.dump").write_bytes(b"x"*4096); '
                       'time.sleep(60)']
            rc, out, err, timed_out, failure = runner.run_bounded_case(
                command, 0.2, root, min_free_bytes=0, poll_seconds=0.05)
            self.assertTrue(timed_out)
            self.assertIsNone(failure)
            self.assertNotEqual(rc, 0)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_disk_reserve_prevents_launch_and_preserves_other_files(self):
        with tempfile.TemporaryDirectory() as root:
            sentinel = Path(root) / 'prior-evidence.json'
            sentinel.write_text('retained')
            with patch.object(runner.shutil, 'disk_usage',
                              return_value=SimpleNamespace(free=0)), \
                    patch.object(runner.subprocess, 'Popen') as spawn:
                result = runner.run_bounded_case(['never'], 10, root)
            spawn.assert_not_called()
            self.assertEqual(result[-1], 'disk_reserve')
            self.assertEqual(sentinel.read_text(), 'retained')
            self.assertEqual(list(Path(root).iterdir()), [sentinel])

class RunningDiskBudgetTest(unittest.TestCase):
    def test_low_space_stops_running_child_and_cleans(self):
        with tempfile.TemporaryDirectory() as root:
            command = [sys.executable, '-c', 'import time; time.sleep(60)']
            with patch.object(runner.shutil, 'disk_usage', side_effect=[
                    SimpleNamespace(free=20 << 30),
                    SimpleNamespace(free=20 << 30),
                    SimpleNamespace(free=1 << 30)]):
                rc, _, _, timed_out, failure = runner.run_bounded_case(
                    command, 60, root, poll_seconds=0.1)
            self.assertFalse(timed_out)
            self.assertEqual(failure, 'disk_reserve')
            self.assertNotEqual(rc, 0)
            self.assertEqual(list(Path(root).iterdir()), [])

class DescendantScratchTest(unittest.TestCase):
    def test_timeout_kills_descendant_that_ignores_term(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / 'survived'
            child_code = ('import signal,time,pathlib; '
                          'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                          'time.sleep(1); pathlib.Path(' + repr(str(marker)) + ').touch()')
            parent_code = ('import subprocess,sys,time; subprocess.Popen('
                           '[sys.executable,"-c",' + repr(child_code) + ']); time.sleep(60)')
            runner.run_bounded_case([sys.executable, '-c', parent_code],
                                    0.3, Path(root) / 'scratch', min_free_bytes=0)
            time.sleep(1)
            self.assertFalse(marker.exists())
            self.assertEqual(list((Path(root) / 'scratch').iterdir()), [])
