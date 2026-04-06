#!/usr/bin/env python3.10
"""JIT kernel verification for pyasc using the same mock-launcher pattern as pyasc unit tests.

This mirrors the ``mock_launcher_run`` fixture in
``python/test/unit/conftest.py`` (``patch("asc.runtime.launcher.Launcher.run", ...)``)
and tests such as ``python/test/unit/language/basic/test_vector_unary.py``::

    def test_abs_kernel(mock_launcher_run):
        @asc.jit
        def abs_kernel(): ...
        abs_kernel[1]()
        assert mock_launcher_run.call_count == 1

This script is **not** a pytest test module: it runs standalone and applies the
patch directly so JIT codegen can be exercised without the simulator or device.

Loads ``kernel.py`` as a module, sets ``config.set_platform(config.Backend.Model,
check=False)``, patches ``Launcher.run``, then invokes each module-level
``@asc.jit`` function until a launch succeeds (``Launcher.run`` called).

Usage::

    python3.10 pytest_verify_kernel.py /path/to/kernel.py [--json]

Exit codes:
    0 — PASS (all JIT kernels reached ``Launcher.run`` after compile)
    1 — FAIL (bad path, syntax/import error, no JIT symbols, or compile/launch failure)
    2 — SKIP (``asc`` / pyasc not importable)
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import traceback
from pathlib import Path
from types import UnionType
from typing import Any, Callable, List, Optional, Tuple, Union, get_args, get_origin
from unittest.mock import patch

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 2


def _try_import_asc():
    """Return (asc_module, None) or (None, error_message)."""
    try:
        import asc  # noqa: WPS433 — intentional late import for exit code 2

        from asc.runtime import config  # noqa: F401
        import asc.runtime.launcher  # noqa: F401

        return asc, None
    except ImportError as e:
        return None, str(e)


def _unwrap_optional(annotation: Any) -> Any:
    if annotation is inspect.Parameter.empty:
        return annotation
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_global_address_type(asc_mod: Any, ann: Any) -> bool:
    if ann is inspect.Parameter.empty or ann is object:
        return False
    try:
        from asc.language.core.ir_value import GlobalAddress
    except ImportError:
        return False
    unwrapped = _unwrap_optional(ann)
    return unwrapped is GlobalAddress


def _annotation_implies_int(ann: Any) -> bool:
    if ann is inspect.Parameter.empty:
        return False
    u = _unwrap_optional(ann)
    return u is int


def _annotation_implies_float(ann: Any) -> bool:
    if ann is inspect.Parameter.empty:
        return False
    u = _unwrap_optional(ann)
    return u is float


def _annotation_implies_bool(ann: Any) -> bool:
    if ann is inspect.Parameter.empty:
        return False
    u = _unwrap_optional(ann)
    return u is bool


def _mock_tensor_dtype(asc_mod: Any, ann: Any):
    """Pick a DataType for MockTensor; default float16."""
    from asc.runtime.jit import MockTensor

    u = _unwrap_optional(ann)
    dtype = getattr(u, "dtype", None)
    if dtype is not None:
        return MockTensor(dtype)
    try:
        import torch

        if u is torch.Tensor:
            return MockTensor(asc_mod.float16)
    except ImportError:
        pass
    return MockTensor(asc_mod.float16)


def _build_call_args(asc_mod: Any, jit_fn: Any) -> Tuple[List[Any], List[str]]:
    """Positional args for ``jit_fn[...](*args)`` from the wrapped function signature."""
    from asc.common.compat import get_annotations
    from asc.language.core.constexpr import ConstExpr
    from asc.runtime.jit import MockTensor

    sig = inspect.signature(jit_fn.fn)
    annotations = get_annotations(jit_fn.fn)
    values: List[Any] = []
    notes: List[str] = []

    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            notes.append(f"skip variadic parameter {name!r}")
            continue

        ann = annotations.get(name, param.annotation)
        ann = _unwrap_optional(ann)

        if ann is not inspect.Parameter.empty:
            origin = get_origin(ann) or ann
            try:
                if isinstance(origin, type) and issubclass(origin, ConstExpr):
                    values.append(1)
                    notes.append(f"{name}: ConstExpr -> 1")
                    continue
            except TypeError:
                pass

        if _is_global_address_type(asc_mod, ann) or ann is inspect.Parameter.empty:
            if _is_global_address_type(asc_mod, ann):
                values.append(_mock_tensor_dtype(asc_mod, ann))
                notes.append(f"{name}: GlobalAddress -> MockTensor")
                continue
            lower = name.lower()
            if lower in {"x", "y", "a", "b", "c", "src", "dst", "input", "output"} or lower.endswith(
                "_gm"
            ):
                values.append(MockTensor(asc_mod.float16))
                notes.append(f"{name}: heuristic pointer -> MockTensor(float16)")
                continue

        if _annotation_implies_int(ann):
            values.append(512)
            notes.append(f"{name}: int -> 512")
            continue
        if _annotation_implies_float(ann):
            values.append(0.0)
            notes.append(f"{name}: float -> 0.0")
            continue
        if _annotation_implies_bool(ann):
            values.append(False)
            notes.append(f"{name}: bool -> False")
            continue

        if ann is not inspect.Parameter.empty and ann not in (object,):
            ann_name = getattr(ann, "__name__", str(ann))
            raise TypeError(f"Unsupported parameter {name!r} annotation {ann_name!r} for mock invocation")

        values.append(512)
        notes.append(f"{name}: default -> 512 (int)")

    return values, notes


def _launch_binders(jit_fn: Any) -> List[Tuple[str, Callable[[], Any]]]:
    """Return (label, lambda that returns the bound runner) for common launch patterns."""

    def make_int(n: int):
        return lambda: jit_fn[n]

    def make_tuple(t: Tuple[Any, ...]):
        return lambda: jit_fn[t]

    binders: List[Tuple[str, Callable[[], Any]]] = [
        ("[1]", make_int(1)),
        ("[8]", make_int(8)),
        ("[8, None]", make_tuple((8, None))),
        ("[1, None]", make_tuple((1, None))),
    ]

    try:
        from asc.lib import runtime as rt

        stream = rt.current_stream()
        binders.extend(
            [
                ("[8, current_stream()]", make_tuple((8, stream))),
                ("[1, current_stream()]", make_tuple((1, stream))),
            ]
        )
    except Exception:
        pass

    return binders


def _verify_single_jit(asc_mod: Any, jit_fn: Any) -> dict[str, Any]:
    """Run one JITFunction under Launcher.run patch; return result dict."""
    name = getattr(jit_fn, "__name__", getattr(jit_fn.fn, "__name__", "<unknown>"))
    result: dict[str, Any] = {
        "name": name,
        "passed": False,
        "launcher_calls": 0,
        "launch_pattern": None,
        "arg_notes": [],
        "error": None,
    }
    try:
        pos_args, notes = _build_call_args(asc_mod, jit_fn)
    except Exception as e:
        result["error"] = f"build args: {e}"
        return result

    result["arg_notes"] = notes

    last_err: Optional[str] = None
    with patch("asc.runtime.launcher.Launcher.run", return_value=None) as mock_run:
        for label, binder in _launch_binders(jit_fn):
            try:
                runner = binder()
                runner(*pos_args)
                calls = mock_run.call_count
                mock_run.reset_mock()
                if calls >= 1:
                    result["passed"] = True
                    result["launcher_calls"] = calls
                    result["launch_pattern"] = label
                    return result
                last_err = f"{label}: Launcher.run not invoked (call_count=0)"
            except Exception as e:
                mock_run.reset_mock()
                last_err = f"{label}: {e}"

    result["error"] = last_err or "Launcher.run was never called"
    return result


def _iter_jit_functions(module: Any) -> List[Any]:
    from asc.runtime.jit import JITFunction

    found: List[Any] = []
    for _attr_name, obj in inspect.getmembers(module):
        if isinstance(obj, JITFunction):
            found.append(obj)
    return found


def _load_kernel_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify pyasc JIT kernels with a mocked Launcher.run.")
    parser.add_argument("kernel_py", type=Path, help="Path to kernel.py")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON on stdout")
    args = parser.parse_args(argv)

    asc_mod, import_err = _try_import_asc()
    if asc_mod is None:
        payload = {
            "status": "skip",
            "exit_code": EXIT_SKIP,
            "reason": "pyasc (asc) not importable",
            "import_error": import_err,
            "kernels": [],
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"SKIP: pyasc not installed or not on PYTHONPATH: {import_err}", file=sys.stderr)
        return EXIT_SKIP

    from asc.runtime import config

    kernel_path = args.kernel_py.resolve()
    if not kernel_path.is_file():
        payload = {
            "status": "fail",
            "exit_code": EXIT_FAIL,
            "error": f"Not a file: {kernel_path}",
            "kernels": [],
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"FAIL: {payload['error']}", file=sys.stderr)
        return EXIT_FAIL

    config.set_platform(config.Backend.Model, check=False)

    try:
        mod = _load_kernel_module(kernel_path)
    except SyntaxError as e:
        payload = {
            "status": "fail",
            "exit_code": EXIT_FAIL,
            "error": f"SyntaxError: {e}",
            "kernels": [],
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"FAIL: {payload['error']}", file=sys.stderr)
        return EXIT_FAIL
    except Exception as e:
        payload = {
            "status": "fail",
            "exit_code": EXIT_FAIL,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "kernels": [],
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"FAIL: {payload['error']}", file=sys.stderr)
            traceback.print_exc()
        return EXIT_FAIL

    jit_fns = _iter_jit_functions(mod)
    if not jit_fns:
        payload = {
            "status": "fail",
            "exit_code": EXIT_FAIL,
            "error": "No @asc.jit functions found in module",
            "kernels": [],
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"FAIL: {payload['error']}", file=sys.stderr)
        return EXIT_FAIL

    results = [_verify_single_jit(asc_mod, fn) for fn in jit_fns]
    all_passed = all(r["passed"] for r in results)

    payload = {
        "status": "pass" if all_passed else "fail",
        "exit_code": EXIT_PASS if all_passed else EXIT_FAIL,
        "kernel_path": str(kernel_path),
        "kernels": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Kernel: {kernel_path}")
        for r in results:
            if r["passed"]:
                print(
                    f"  PASS  {r['name']}  (Launcher.run calls={r['launcher_calls']}, pattern={r['launch_pattern']})"
                )
            else:
                print(f"  FAIL  {r['name']}: {r['error']}")
        if not all_passed:
            print("FAIL: one or more JIT kernels did not reach Launcher.run", file=sys.stderr)

    return EXIT_PASS if all_passed else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
