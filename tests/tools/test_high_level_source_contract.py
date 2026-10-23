#!/usr/bin/env python3
"""CPU-only source-contract and promotion-boundary regression tests.

Run: python3 -m unittest discover -s tests/tools -p test_high_level_source_contract.py
No candidate import, credentials, compiler, or remote submission is needed.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
WORKERS = ROOT / "integrations/cannbench/workers"
SUBMISSION = ROOT / "integrations/cannbench/submission"
sys.path[:0] = [str(WORKERS), str(SUBMISSION)]

from source_contract import check_high_level_source, check_runtime_helper, package_sources
import assemble_local_bundle
import driver
import evalqueue
import merge_wheels
import submit_skill_candidate


CLEAN = '''\
import asctile as tile
import torch as th
from ._pyasc_runtime import ensure_npu_platform as ensure

@tile.jit(vf_fusion=True)
def kernel(x: tile.GlobalAddress, out: tile.GlobalAddress, n: tile.ConstExpr):
    src = tile.global_tensor(x, [n])
    dst = tile.global_tensor(out, [n])
    value = tile.copy_in(src, [0], [128])
    result = tile.exp(value.to(tile.float32))
    tile.copy_out(result.to(value.dtype), dst, [0])

def gelu(x: th.Tensor, approximate: str = "none") -> th.Tensor:
    ensure()
    n = x.numel()
    y = th.empty_like(x)
    kernel[1](x, y, n)
    return y
'''

POISONED_HELPER = '''\
import asctile as tile
from asc.runtime.compiler import Compiler as BaseCompiler

class PatchedCompiler(BaseCompiler):
    pass

def custom_jit(fn):
    return fn

tile.jit = custom_jit

def ensure_npu_platform():
    pass
'''


class SourceContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.candidate = self.root / "gelu.py"

    def check(self, source):
        self.candidate.write_text(textwrap.dedent(source), encoding="utf-8")
        return check_high_level_source(self.candidate)

    def assert_rejected(self, source, message):
        problems = self.check(source)
        self.assertTrue(problems, source)
        self.assertIn(message, "\n".join(problems))

    def test_public_kernel_with_compiler_generated_vf_fusion_is_legal(self):
        self.assertEqual(self.check(CLEAN), [])

    def test_import_and_chained_assignment_aliases(self):
        source = CLEAN.replace("import asctile as tile", "from asctile import jit as decorate\nimport asctile as tile\nfirst = decorate\nsecond = first")
        self.assertEqual(self.check(source.replace("@tile.jit", "@second")), [])
        self.assertEqual(self.check(CLEAN.replace("import torch as th", "import torch\nth = torch")), [])

    def test_from_imported_public_compute_types_and_allocation(self):
        source = CLEAN.replace("import asctile as tile", "import asctile as tile\nfrom asctile import exp as exponential\nfrom asctile import GlobalAddress as Address")
        source = source.replace("tile.exp(value", "exponential(value").replace("tile.GlobalAddress", "Address")
        source = source.replace("import torch as th", "import torch as th\nfrom torch import empty_like as allocate").replace("th.empty_like(x)", "allocate(x)")
        self.assertEqual(self.check(source), [])

    def test_comments_strings_and_generated_register_text_are_not_code(self):
        self.assertEqual(self.check(CLEAN + '\n# asc2.RegTensor / inline_vf\nDIAGNOSTIC = "asc.RegTensor; exec; ctypes; inline"\n'), [])

    def test_low_level_imports_compute_types_and_aliases(self):
        cases = [
            "import asc", "import asc as low", "import asc2 as tile2",
            "from asc import GlobalAddress as Ptr", "from asc import float32 as f32",
            "from asc.language import RegTensor as R", "import asc.language as low",
            "from asc.runtime.jit import jit as decorate", "asc.add(a, b)",
            "asc2.LocalTensor(128)", "r = RegTensor()", "q = TQue()",
            "def hidden(x: asc.GlobalAddress): pass",
            'def hidden(x: "asc.GlobalAddress"): pass',
            'Ptr: "asc.GlobalAddress"',
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                self.assert_rejected(CLEAN + "\n" + extra + "\n", "low-level")

    def test_inline_direct_imported_and_assignment_aliases(self):
        cases = [
            "tile.inline('bad')", "tile.inline_vf('bad')", "inline('bad')",
            "inline_vf('bad')", "emit = tile.inline\nemit('bad')",
            "a, b = tile.inline, tile.exp\na('bad')",
            "from asctile import inline_vf as emit\nemit('bad')",
            "from asctile.language.debug_ops import inline as emit",
            "def hidden():\n    emit = tile.inline_vf\n    return emit('bad')",
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                self.assert_rejected(CLEAN + "\n" + extra + "\n", "inline")

    def test_imported_custom_helpers_fail_closed_without_reading_them(self):
        for statement in ("from .helper import run as safe", "import helper as safe", "from helpers import high_level", "from . import helper", "from package._pyasc_runtime import ensure_npu_platform"):
            with self.subTest(statement=statement):
                self.assert_rejected(CLEAN + "\n" + statement, "imported helper")
        self.assertFalse((self.root / "helper.py").exists())

    def test_internal_asctile_and_wildcard_imports(self):
        for extra in ("import asctile.runtime.jit as runtime", "from asctile import *", "tile.runtime.launch()", "tile.CompileOptions()", "tile.unknown_api()"):
            with self.subTest(extra=extra):
                self.assertTrue(self.check(CLEAN + "\n" + extra))

    def test_narrow_runtime_helper(self):
        self.assertEqual(self.check(CLEAN.replace("from ._pyasc_runtime", "from _pyasc_runtime")), [])
        for extra in ("from ._pyasc_runtime import custom_jit as decorate", "from ._pyasc_runtime import other", "import _pyasc_runtime as rt", "from ._pyasc_runtime import *", "ensure(1)"):
            with self.subTest(extra=extra):
                self.assertTrue(self.check(CLEAN + "\n" + extra))
        self.assert_rejected(CLEAN.replace("    src =", "    ensure()\n    src ="), "only on the host")

    def test_real_platform_helpers_are_accepted_without_importing_them(self):
        real_helpers = [SUBMISSION / "cann_bench/_pyasc_runtime.py",
                        ROOT / "integrations/cannbench/comparisons/gelu-asctile-jit-20260907/candidate/_pyasc_runtime.py"]
        self.check(CLEAN)
        for real in real_helpers:
            with self.subTest(helper=real):
                self.assertTrue(real.is_file(), real)
                self.assertEqual(check_runtime_helper(real), [])
                (self.root / "_pyasc_runtime.py").write_bytes(real.read_bytes())
                self.assertEqual(check_high_level_source(self.candidate), [])

    def test_missing_draft_runtime_is_allowed_but_present_invalid_runtime_is_rejected(self):
        self.assertEqual(self.check(CLEAN), [])
        helper = self.root / "_pyasc_runtime.py"
        for source in ("", "def broken(:", POISONED_HELPER):
            with self.subTest(source=source):
                helper.write_text(source)
                problems = check_high_level_source(self.candidate)
                self.assertTrue(problems)
                self.assertIn(str(helper), "\n".join(problems))
        # local_compile_gate loads adjacent runtime even if the operator has no
        # explicit helper import, so this audit must not depend on that import.
        self.assertTrue(self.check(CLEAN.replace("from ._pyasc_runtime import ensure_npu_platform as ensure\n", "").replace("    ensure()\n", "")))

    def test_runtime_audits_unused_definitions_numerics_inline_and_monkeypatches(self):
        safe = "import asctile as tile\ndef ensure_npu_platform():\n    tile.set_platform(tile.Backend.NPU)\n"
        cases = [
            "def custom_jit(fn):\n    return fn",
            "def unused():\n    return tile.exp(1)",
            "from asctile.runtime.jit import JITFunction as Base",
            "from asc.runtime.compiler import Compiler as C\nclass Custom(C):\n    pass",
            "import torch as th\nresult = th.exp(1)",
            "emit = tile.inline_vf\nemit('bad')",
            "from asctile import inline as emit",
            "tile.jit = ensure_npu_platform",
            "alias = tile\nalias.set_platform = ensure_npu_platform",
            "from asc.runtime.config import set_platform as setup\nsetup.compiler = 1",
            "import threading as threads\nthreads.Lock = ensure_npu_platform",
            "tile.__dict__['jit'] = ensure_npu_platform",
            "setattr(tile, 'jit', ensure_npu_platform)",
            "run = exec\nrun('bad')",
            "import ctypes as ffi", "import importlib as loader",
            "count = 2 * 3",
        ]
        self.check(CLEAN)
        helper = self.root / "_pyasc_runtime.py"
        for extra in cases:
            with self.subTest(extra=extra):
                helper.write_text(safe + "\n" + extra + "\n")
                self.assertTrue(check_high_level_source(self.candidate), extra)

    def test_runtime_platform_import_aliases_are_accepted(self):
        helper = self.root / "_pyasc_runtime.py"
        source = '''
from __future__ import annotations
from threading import Lock as Mutex
from os import getenv as platform_override
from asc.runtime.config import set_platform as setup, Backend as BackendKind, Platform as Target
_mutex = Mutex()
_ready = False
def ensure_npu_platform() -> None:
    global _ready
    with _mutex:
        if not _ready:
            override = platform_override("CANN_BENCH_PYASC_PLATFORM")
            if override:
                setup(BackendKind.NPU, Target(override))
            else:
                setup(BackendKind.NPU)
            _ready = True
'''
        helper.write_text(textwrap.dedent(source))
        self.assertEqual(self.check(CLEAN), [])

    def test_poisoned_runtime_rejected_before_local_import(self):
        import local_compile_gate
        self.check(CLEAN)
        (self.root / "_pyasc_runtime.py").write_text(POISONED_HELPER)
        with patch.object(local_compile_gate.importlib.util, "spec_from_file_location", side_effect=AssertionError("must not import helper or candidate")):
            with self.assertRaisesRegex(ValueError, "_pyasc_runtime.py"):
                local_compile_gate.load_candidate(self.candidate, "gelu")
            with self.assertRaisesRegex(ValueError, "_pyasc_runtime.py"):
                local_compile_gate.evaluate(self.candidate, "gelu", self.root / "missing.yaml")

    def test_dynamic_execution_imports_ctypes_and_reflection(self):
        cases = [
            "exec('bad')", "run = exec\nrun('bad')", "eval('bad')",
            "compile('bad', 'x', 'exec')", "__import__('asc')",
            "import importlib as loader", "from importlib import import_module as load",
            "import ctypes as ffi", "from ctypes import CDLL as load",
            "import builtins as b", "getattr(tile, 'inline')('bad')",
            "lookup = getattr\nlookup(tile, 'inline')('bad')",
            "tile.__dict__['inline']('bad')", "globals()['exec']('bad')",
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                self.assert_rejected(CLEAN + "\n" + extra, "dynamic")

    def test_torch_numerical_module_and_function_aliases(self):
        cases = [
            "th.exp(x)", "compute = th.exp", "th.nn.functional.gelu(x)",
            "th.Tensor.sin(x)", "th.ops.aten.gelu(x)",
            "from torch import exp as compute", "import torch.nn.functional as F",
            "from torch.nn.functional import gelu as compute", "import torch_npu",
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                self.assertTrue(self.check(CLEAN + "\n" + extra))

    def test_tensor_methods_operators_and_bound_aliases(self):
        cases = [
            "return x.exp()", "compute = x.sin\nreturn compute()",
            "return x + 1", "return -x", "return x > 0", "x *= 2\nreturn x",
            "return th.empty_like(x).add_(1)", "y = x\nreturn y.sum()",
            "return x.to(th.float32)", "return x.cpu().numpy()",
            "return abs(x)", "return float(x)", "return x.data_ptr()",
            "return [item.square() for item in x]",
            "for item in x:\n    item.add_(1)\nreturn x",
        ]
        for body in cases:
            with self.subTest(body=body):
                source = CLEAN + "\ndef hidden(x):\n" + textwrap.indent(body, "    ") + "\n"
                self.assert_rejected(source, "Tensor")

    def test_host_allocation_metadata_views_and_scalar_math(self):
        extra = '''
def host(xs, scale: float):
    out = []
    for index, x in enumerate(xs):
        n = x.numel()
        blocks = (n + 127) // 128
        shape = x.shape
        y = th.empty(shape, dtype=x.dtype, device=x.device)
        y = y.reshape(shape).contiguous()
        kernel[blocks](x, y, n)
        out.append(y)
        blocks += index
    return out
'''
        self.assertEqual(self.check(CLEAN + extra), [])

    def test_alias_scopes_do_not_leak_between_functions(self):
        source = CLEAN + '\ndef helper(th: int):\n    return th + 1\n'
        self.assertEqual(self.check(source), [])
        self.assert_rejected(source + "\ndef wrong(x):\n    return th.sin(x)\n", "torch")

    def test_tensor_branch_lambda_and_varargs_provenance(self):
        cases = [
            "def hidden(x):\n    if x is not None:\n        y = x\n    else:\n        y = None\n    return y.sin()",
            "hidden = lambda x: x.sin()",
            "def hidden(*args):\n    return args[0].sin()",
            "def hidden(**kwargs):\n    return kwargs['x'].sin()",
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                self.assert_rejected(CLEAN + "\n" + extra, "Tensor")

    def test_torch_allocation_cannot_move_into_kernel(self):
        self.assert_rejected(CLEAN.replace("    src =", "    th.empty(1)\n    src ="), "host allocation/metadata")

    def test_missing_jit_import_syntax_and_missing_file(self):
        for source in ("import asctile", "@asctile.jit\ndef kernel(): pass", "def broken(:"):
            with self.subTest(source=source):
                self.assertTrue(self.check(source))
        self.assertTrue(check_high_level_source(self.root / "missing.py"))

    def test_worker_signature_check_uses_alias_aware_contract(self):
        self.check(CLEAN)
        self.assertEqual(driver.static_check(self.candidate, "gelu"), [])
        self.assertIs(submit_skill_candidate.static_check, driver.static_check)
        self.check(CLEAN.replace("def gelu(x:", "def gelu(wrong:"))
        self.assertIn("public signature", "\n".join(driver.static_check(self.candidate, "gelu")))
        self.check(CLEAN + "\nimport asc as low\n")
        self.assertIn("low-level", "\n".join(driver.static_check(self.candidate, "gelu")))

    def test_local_compile_rejects_before_import_or_runtime_setup(self):
        # Loading the gate needs no yaml/pyasc/torch. No candidate is imported.
        import local_compile_gate
        self.check(CLEAN + "\nimport asc as low\n")
        with patch.object(local_compile_gate.importlib.util, "spec_from_file_location", side_effect=AssertionError("must not import")):
            with self.assertRaisesRegex(ValueError, "source contract"):
                local_compile_gate.load_candidate(self.candidate, "gelu")
            with self.assertRaisesRegex(ValueError, "source contract"):
                local_compile_gate.evaluate(self.candidate, "gelu", self.root / "missing.yaml")

    def test_submission_provenance_does_not_override_source_contract(self):
        source = CLEAN.replace('def gelu(x: th.Tensor, approximate: str = "none")',
                               'def rms_norm(x: th.Tensor, gamma: th.Tensor, epsilon: float = 1e-6)')
        candidate = self.root / "candidate.py"
        for invalid in (False, True):
            candidate.write_text(source + ("\nimport asc\n" if invalid else ""))
            digest = submit_skill_candidate.sha256(candidate)
            provenance = {"candidate_sha256": digest, "skill_gate_passed": True,
                          "loaded_skills": list(driver.REQUIRED_SKILLS)}
            validation = {"candidate_sha256": digest,
                          "llvm_lowering": {"passed_cases": 20, "total_cases": 20}}
            (self.root / "provenance.json").write_text(json.dumps(provenance))
            (self.root / "local_validation.json").write_text(json.dumps(validation))
            if invalid:
                with self.assertRaisesRegex(RuntimeError, "static candidate gate failed"):
                    submit_skill_candidate.validate_run(self.root)
            else:
                self.assertEqual(submit_skill_candidate.validate_run(self.root)[0], candidate)

    def test_staging_rejects_before_creating_output_or_using_client(self):
        self.check(CLEAN + "\nimport asc as low\n")
        with patch.object(evalqueue.tempfile, "mkdtemp", side_effect=AssertionError("must not stage")):
            with self.assertRaisesRegex(ValueError, "source contract"):
                evalqueue.EvalQueue._stage_submission(self.candidate, "", "gelu")
        # Bypass __init__: it loads credentials and is deliberately never called.
        queue = object.__new__(evalqueue.EvalQueue)
        with patch.object(queue, "_check_credit", side_effect=AssertionError("must not call client")):
            with self.assertRaisesRegex(ValueError, "source contract"):
                queue.evaluate("gelu", self.candidate, "", self.root)

    def test_staging_records_only_new_promotion_and_preserves_historical_source(self):
        self.check(CLEAN)
        base = self.root / "submission"
        (base / "cann_bench").mkdir(parents=True)
        historical = base / "cann_bench/exp.py"
        historical.write_text("import asc\n", encoding="utf-8")
        stage = self.root / "stage"
        stage.mkdir()
        with patch.object(evalqueue, "SUBMISSION_ROOT", base), patch.object(evalqueue.tempfile, "mkdtemp", return_value=str(stage)):
            evalqueue.EvalQueue._stage_submission(self.candidate, "", "gelu")
        self.assertEqual(json.loads((stage / "source-contract-promotions.json").read_text()), ["gelu"])
        self.assertEqual(historical.read_text(), "import asc\n")
        command = [sys.executable, str(SUBMISSION / "source_contract.py"), "--package", str(stage / "cann_bench")]
        direct = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(direct.returncode, 1, direct.stdout + direct.stderr)
        scoped = subprocess.run(command + ["--promotions", str(stage / "source-contract-promotions.json")], capture_output=True, text=True)
        self.assertEqual(scoped.returncode, 0, scoped.stdout + scoped.stderr)

    def test_scoped_package_gate_audits_poisoned_helper_even_when_not_selected(self):
        self.check(CLEAN)
        (self.root / "_pyasc_runtime.py").write_text(POISONED_HELPER)
        manifest = self.root / "promotions.json"
        manifest.write_text('["gelu"]')
        result = subprocess.run(
            [sys.executable, str(SUBMISSION / "source_contract.py"), "--package", str(self.root),
             "--promotions", str(manifest)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("_pyasc_runtime.py", result.stdout)

    def test_staging_audits_runtime_introduced_into_a_standalone_draft(self):
        self.check(CLEAN)
        base = self.root / "submission"
        (base / "cann_bench").mkdir(parents=True)
        helper = base / "cann_bench/_pyasc_runtime.py"
        helper.write_text(POISONED_HELPER)
        with patch.object(evalqueue, "SUBMISSION_ROOT", base), patch.object(evalqueue.tempfile, "mkdtemp", side_effect=AssertionError("must not stage")):
            with self.assertRaisesRegex(ValueError, "_pyasc_runtime.py"):
                evalqueue.EvalQueue._stage_submission(self.candidate, "", "gelu")
        self.assertEqual(helper.read_text(), POISONED_HELPER)

    def test_bundle_rejects_selected_base_before_any_writes(self):
        base = self.root / "base"
        (base / "cann_bench").mkdir(parents=True)
        (base / "cann_bench/gelu.py").write_text(CLEAN + "\nimport asc\n")
        output = self.root / "output"
        argv = ["assemble", "--base", str(base), "--output-dir", str(output)]
        with patch.object(sys, "argv", argv), patch.object(driver, "ALL_OPS", ("gelu",)):
            with self.assertRaisesRegex(ValueError, "source contract"):
                assemble_local_bundle.main()
        self.assertFalse(output.exists())

    def test_bundle_checks_qualified_replacement_not_superseded_base(self):
        base = self.root / "base"
        (base / "cann_bench").mkdir(parents=True)
        historical = base / "cann_bench/gelu.py"
        historical.write_text("import asc\n")
        (base / "cann_bench/_pyasc_runtime.py").write_text("def ensure_npu_platform(): pass\n")
        qualified = self.root / "qualified"
        qualified.mkdir()
        replacement = qualified / "gelu.py"
        replacement.write_text(CLEAN)
        (qualified / "provenance.json").write_text("{}")
        output = self.root / "output"
        argv = ["assemble", "--base", str(base), "--qualified", str(qualified), "--output-dir", str(output)]
        with patch.object(sys, "argv", argv), patch.object(driver, "ALL_OPS", ("gelu",)), patch.object(driver, "SUBMISSION_PKG", base / "cann_bench"), patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(assemble_local_bundle.main(), 0)
            self.assertEqual((output / "cann_bench/gelu.py").read_text(), CLEAN)
            replacement.write_text(CLEAN + "\nimport asc\n")
            with self.assertRaisesRegex(ValueError, "source contract"):
                assemble_local_bundle.main()
        self.assertEqual(historical.read_text(), "import asc\n")
        self.assertEqual((output / "cann_bench/gelu.py").read_text(), CLEAN)

    def test_bundle_audits_runtime_introduced_from_canonical_package(self):
        base = self.root / "base"
        (base / "cann_bench").mkdir(parents=True)
        (base / "cann_bench/gelu.py").write_text(CLEAN)
        canonical = self.root / "canonical"
        canonical.mkdir()
        (canonical / "_pyasc_runtime.py").write_text(POISONED_HELPER)
        output = self.root / "output"
        argv = ["assemble", "--base", str(base), "--output-dir", str(output)]
        with patch.object(sys, "argv", argv), patch.object(driver, "ALL_OPS", ("gelu",)), patch.object(driver, "SUBMISSION_PKG", canonical):
            with self.assertRaisesRegex(ValueError, "bundle runtime helper"):
                assemble_local_bundle.main()
        self.assertFalse(output.exists())

    def test_direct_wheel_merge_rejects_before_runtime_reads_and_output_writes(self):
        wheel = self.root / "base.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("cann_bench/gelu.py", CLEAN + "\nimport asc\n")
        output = self.root / "output"
        argv = ["merge", "--base-wheel", str(wheel), "--runtime-wheel", "missing-runtime.whl",
                "--pybind11-wheel", "missing-pybind11.whl", "--license", "missing-license",
                "--output-dir", str(output)]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "source contract"):
                merge_wheels.main()
        self.assertFalse(output.exists())

    def test_wheel_merge_audits_poisoned_helper_before_reading_vendor_runtime(self):
        wheel = self.root / "base.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("cann_bench/gelu.py", CLEAN)
            archive.writestr("cann_bench/_pyasc_runtime.py", POISONED_HELPER)
        output = self.root / "output"
        argv = ["merge", "--base-wheel", str(wheel), "--runtime-wheel", "missing-runtime.whl",
                "--pybind11-wheel", "missing-pybind11.whl", "--license", "missing-license",
                "--output-dir", str(output)]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "_pyasc_runtime.py"):
                merge_wheels.main()
        self.assertFalse(output.exists())

    def test_promotion_manifest_rejects_empty_missing_and_non_operator_paths(self):
        manifest = self.root / "promotions.json"
        for value in ([], {}, ["../escape"], ["_pyasc_runtime"], [1]):
            with self.subTest(value=value):
                manifest.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    package_sources(self.root, manifest)
        manifest.write_text('["missing"]')
        selected = package_sources(self.root, manifest)
        self.assertTrue(check_high_level_source(selected[0]))

    def test_build_entrypoint_rejects_before_replacing_outputs(self):
        # Only the rejection path is exercised; no wheel build or compiler runs.
        (self.root / "build.sh").write_bytes((SUBMISSION / "build.sh").read_bytes())
        (self.root / "source_contract.py").write_bytes((SUBMISSION / "source_contract.py").read_bytes())
        package = self.root / "cann_bench"
        package.mkdir()
        (package / "gelu.py").write_text(CLEAN + "\nimport asc\n")
        dist = self.root / "dist"
        dist.mkdir()
        historical = dist / "historical.whl"
        historical.write_bytes(b"preserve")
        result = subprocess.run(["bash", str(self.root / "build.sh")], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("low-level", result.stdout)
        self.assertEqual(historical.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
