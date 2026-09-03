#!/usr/bin/env python3
"""Compile every CANNBench case dispatch against the submitted pyasc v2 runtime.

The gate runs in the amd64/QEMU image built by ``run_local_compile_gate.sh``.
It imports an operator module with a tiny fake ``torch`` host API, records the
JIT launches selected by each benchmark case, and lowers every unique launch
specialization through pyasc to generated AscendC.  No CANNBench credit, NPU,
or torch installation is required.

This proves host dispatch, pyasc AST/codegen, compiler passes, translation, and
the 950PR UB budget.  It deliberately does not claim numerical correctness or
silicon performance; those remain camodel/NPU and CANNBench responsibilities.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PYASC_COMMIT = "030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d"
PLATFORM = "Ascend950PR_9599"


@dataclass(frozen=True)
class FakeDType:
    name: str
    itemsize: int

    def __str__(self) -> str:
        return f"torch.{self.name}"


DTYPES = {
    "float16": FakeDType("float16", 2),
    "bfloat16": FakeDType("bfloat16", 2),
    "float32": FakeDType("float32", 4),
    "int8": FakeDType("int8", 1),
    "uint8": FakeDType("uint8", 1),
    "int16": FakeDType("int16", 2),
    "int32": FakeDType("int32", 4),
    "int64": FakeDType("int64", 8),
    "bool": FakeDType("bool", 1),
}


class FakeTensor:
    """Shape/dtype-only tensor sufficient for CANNBench host dispatch."""

    def __init__(self, shape: Any, dtype: FakeDType, device: str = "npu:0"):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(int(v) for v in shape)
        self.dtype = dtype
        self.device = device

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return self.ndim

    def numel(self) -> int:
        return math.prod(self.shape)

    def element_size(self) -> int:
        return self.dtype.itemsize

    def is_contiguous(self) -> bool:
        return True

    def contiguous(self, *_: Any, **__: Any) -> "FakeTensor":
        return self

    def stride(self, dim: int | None = None):
        values = []
        running = 1
        for size in reversed(self.shape):
            values.append(running)
            running *= size
        strides = tuple(reversed(values))
        return strides if dim is None else strides[dim]

    def size(self, dim: int | None = None):
        return self.shape if dim is None else self.shape[dim]

    def narrow(self, dim: int, start: int, length: int) -> "FakeTensor":
        del start
        shape = list(self.shape)
        shape[dim] = int(length)
        return FakeTensor(shape, self.dtype, self.device)

    def view(self, *args: Any) -> "FakeTensor":
        if len(args) == 1 and isinstance(args[0], FakeDType):
            return FakeTensor(self.shape, args[0], self.device)
        return self.reshape(*args)

    def reshape(self, *args: Any) -> "FakeTensor":
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        shape = [int(v) for v in args]
        if shape.count(-1) > 1:
            raise ValueError("only one inferred dimension is supported")
        if -1 in shape:
            idx = shape.index(-1)
            known = math.prod(v for v in shape if v != -1)
            shape[idx] = self.numel() // known
        return FakeTensor(shape, self.dtype, self.device)

    def flatten(self) -> "FakeTensor":
        return FakeTensor((self.numel(),), self.dtype, self.device)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> "FakeTensor":
        if isinstance(key, int):
            return FakeTensor(self.shape[1:], self.dtype, self.device)
        return self


class FakeTorch(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.Tensor = FakeTensor
        for name, dtype in DTYPES.items():
            setattr(self, name, dtype)
        self.half = DTYPES["float16"]
        self.float = DTYPES["float32"]

    @staticmethod
    def _shape(args: tuple[Any, ...]) -> tuple[int, ...]:
        if len(args) == 1 and isinstance(args[0], int):
            return (args[0],)
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return tuple(args[0])
        return tuple(args)

    def empty(self, *shape: Any, dtype=None, device=None, **_: Any) -> FakeTensor:
        return FakeTensor(self._shape(shape), dtype or self.float32, device or "npu:0")

    def zeros(self, *shape: Any, dtype=None, device=None, **_: Any) -> FakeTensor:
        return self.empty(*shape, dtype=dtype, device=device)

    def tensor(self, value: Any, dtype=None, device=None, **_: Any) -> FakeTensor:
        shape = () if not isinstance(value, (list, tuple)) else (len(value),)
        return FakeTensor(shape, dtype or self.float32, device or "npu:0")

    @staticmethod
    def empty_like(tensor: FakeTensor, **_: Any) -> FakeTensor:
        return FakeTensor(tensor.shape, tensor.dtype, tensor.device)

    @staticmethod
    def zeros_like(tensor: FakeTensor, **_: Any) -> FakeTensor:
        return FakeTensor(tensor.shape, tensor.dtype, tensor.device)


@dataclass
class Launch:
    case_id: int
    kernel_name: str
    jit: Any
    cores: Any
    args: tuple[Any, ...]
    specialization: str | None = None


class CaptureKernel:
    def __init__(self, case_id: int, name: str, jit: Any,
                 launches: list[Launch], failed_specializations: set[str]):
        self.case_id = case_id
        self.name = name
        self.jit = jit
        self.launches = launches
        self.failed_specializations = failed_specializations

    def __getitem__(self, cores: Any):
        def record(*args: Any, **kwargs: Any) -> None:
            if kwargs:
                raise TypeError("keyword JIT launch arguments are not supported by the local gate")
            launch = Launch(self.case_id, self.name, self.jit, cores, args)
            try:
                launch.specialization = prepare_specialization(self.jit, args)[0]
            except Exception as exc:
                launch.specialization = (
                    f"prepare-error:{self.name}:{type(exc).__name__}:{exc}"
                )
            self.launches.append(launch)
            # On a repeated dispatch, reproduce a known compile-time exception.
            # This lets host-side try/fallback logic (for example GELU's wide
            # tile -> safe tile route) select the same path as a real launch.
            if launch.specialization in self.failed_specializations:
                raise RuntimeError(
                    f"local compile gate rejected {launch.specialization}"
                )
        return record


def tensor(shape: Any, dtype: str) -> FakeTensor:
    return FakeTensor(shape, DTYPES[dtype])


def case_call(op: str, fn: Any, case: dict[str, Any]) -> Any:
    shapes = case["input_shape"]
    dtypes = case["dtype"]
    attrs = case.get("attrs") or {}
    if op in {"sigmoid", "exp", "mish", "gelu"}:
        return fn(tensor(shapes[0], dtypes[0]), **attrs)
    if op == "masked_scale":
        return fn(tensor(shapes[0], dtypes[0]), tensor(shapes[1], dtypes[1]), **attrs)
    if op == "swi_glu":
        return fn(tensor(shapes[0], dtypes[0]), **attrs)
    if op == "rms_norm":
        return fn(tensor(shapes[0], dtypes[0]), tensor(shapes[1], dtypes[1]), **attrs)
    if op in {"softmax", "transpose"}:
        return fn(tensor(shapes[0], dtypes[0]), **attrs)
    if op == "foreach_norm":
        xs = [tensor(shape, dtypes[0]) for shape in shapes[0]]
        return fn(xs, **attrs)
    if op == "foreach_addcdiv_scalar":
        lists = [[tensor(shape, dtypes[0]) for shape in group] for group in shapes]
        return fn(*lists, **attrs)
    raise ValueError(f"unsupported operator {op!r}")


def load_candidate(path: Path, op: str):
    fake_torch = FakeTorch()
    sys.modules["torch"] = fake_torch
    package_name = "_local_cann_bench_candidate"
    package = types.ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    sys.modules[package_name] = package
    runtime = types.ModuleType(f"{package_name}._pyasc_runtime")
    runtime.ensure_npu_platform = lambda: None
    sys.modules[runtime.__name__] = runtime
    spec = importlib.util.spec_from_file_location(f"{package_name}.{op}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.torch = fake_torch
    return module


def mock_arg(value: Any):
    from asc import DataType
    from asc.runtime.jit import MockTensor

    if isinstance(value, FakeTensor):
        return MockTensor(DataType(value.dtype.name))
    return value


def prepare_specialization(jit: Any, args: tuple[Any, ...]):
    from asc.common.compat import get_annotations, merge_dict

    kwargs = merge_dict(jit.default_options, {})
    codegen_options = jit.extract_kwargs(jit.codegen.options_cls, kwargs)
    compile_options = jit.extract_kwargs(jit.compiler.options_cls, kwargs)
    mocked = tuple(mock_arg(arg) for arg in args)
    call_args = inspect.signature(jit.fn).bind(*mocked).arguments
    runtime_args, constexprs = jit.split_args(call_args, get_annotations(jit.fn))
    arg_types = {name: jit.get_arg_type(value) for name, value in runtime_args.items()}
    key_data = {
        "kernel": jit.fn.__name__,
        "arg_types": {name: f"{type(value).__name__}:{jit.get_arg_dtype(value)}"
                      for name, value in arg_types.items()},
        "constexprs": {name: repr(value) for name, value in constexprs.items()},
        "codegen": vars(codegen_options),
        "compile": vars(compile_options),
    }
    key = json.dumps(key_data, sort_keys=True, default=str)
    return key, arg_types, constexprs, codegen_options, compile_options


def compile_specialization(jit: Any, prepared: tuple[Any, ...]) -> dict[str, Any]:
    from asc._C import ir
    from asc.codegen.specialization import Specialization

    _, arg_types, constexprs, codegen_options, compile_options = prepared
    module = jit._run_codegen(Specialization(arg_types, constexprs), codegen_options)
    # Use the compiler class bound to this JIT object. asctile.JITFunction binds
    # asctile.runtime.compiler.Compiler; using asc.runtime.Compiler here silently
    # skips the AscTile lowering pipeline and leaves local tensors untranslated.
    compiler = jit.compiler(compile_options)
    compiler.preprocess_module(module)
    if compile_options.run_passes:
        compiler.run_passes(module)
    compiler.postprocess_module(module)
    source = compiler.run_translation(module)
    memory = module.op.get_dict_of_int_attr(ir.attr.memory_consumed)
    jit.launcher.check_memory_overflow(memory)
    return {
        "status": "passed",
        "memory_consumed": memory,
        "ascendc_bytes": len(source.encode("utf-8")),
    }


def evaluate(candidate: Path, op: str, cases_path: Path) -> dict[str, Any]:
    import asc.runtime.config as config
    from asc.runtime.jit import JITFunction

    os.environ.setdefault("PYASC_COMPILER", "/bin/true")
    os.environ.setdefault("PYASC_LINKER", "/bin/true")
    config.set_platform(config.Backend.Model, config.Platform(PLATFORM), check=False)

    module = load_candidate(candidate, op)
    fn = getattr(module, op)
    originals = {
        name: value for name, value in vars(module).items()
        if isinstance(value, JITFunction)
    }
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))["cases"]
    launches: list[Launch] = []
    case_results: list[dict[str, Any]] = []
    compiled: dict[str, dict[str, Any]] = {}
    failed_specializations: set[str] = set()

    for case in cases:
        case_id = int(case["case_id"])
        final_launches: list[Launch] = []
        dispatch_error: str | None = None
        # First pass discovers the optimistic route. If that route fails to
        # compile, subsequent passes inject the failure back into the wrapper
        # so explicit host fallback code can execute.
        for _attempt in range(3):
            selected: list[Launch] = []
            for name, jit in originals.items():
                setattr(
                    module, name,
                    CaptureKernel(
                        case_id, name, jit, selected, failed_specializations
                    ),
                )
            try:
                case_call(op, fn, case)
                dispatch_error = None
            except Exception as exc:
                dispatch_error = f"{type(exc).__name__}: {exc}"
            finally:
                for name, jit in originals.items():
                    setattr(module, name, jit)
            launches.extend(selected)
            final_launches = selected

            discovered_failure = False
            for launch in selected:
                key = launch.specialization
                if key in compiled:
                    continue
                try:
                    prepared = prepare_specialization(launch.jit, launch.args)
                    compiled[key] = compile_specialization(launch.jit, prepared)
                except Exception as exc:
                    compiled[key] = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failed_specializations.add(key)
                    discovered_failure = True
            if discovered_failure:
                continue
            break

        successful = [
            launch for launch in final_launches
            if compiled.get(launch.specialization, {}).get("status") == "passed"
        ]
        failed = [
            compiled[launch.specialization] for launch in final_launches
            if compiled.get(launch.specialization, {}).get("status") != "passed"
        ]
        if dispatch_error is not None or not successful:
            case_results.append({
                "case_id": case_id,
                "dispatch": "failed" if dispatch_error else "passed",
                "compile": "failed",
                "error": dispatch_error,
                "compile_errors": sorted({item.get("error", "unknown") for item in failed}),
                "kernels": [launch.kernel_name for launch in final_launches],
                "cores": [launch.cores for launch in final_launches],
            })
            continue
        case_results.append({
            "case_id": case_id,
            "dispatch": "passed",
            "compile": "passed",
            "kernels": [launch.kernel_name for launch in successful],
            "cores": [launch.cores for launch in successful],
            "handled_compile_failures": sorted({
                item.get("error", "unknown") for item in failed
            }),
        })

    dispatch_passed = sum(item["dispatch"] == "passed" for item in case_results)
    compile_passed = sum(item.get("compile") == "passed" for item in case_results)
    unique_passed = sum(item["status"] == "passed" for item in compiled.values())
    report = {
        "schema_version": 1,
        "operator": op,
        "candidate": str(candidate),
        "pyasc_source": "https://gitcode.com/compiler-team/pyasc/tree/v2",
        "pyasc_commit": PYASC_COMMIT,
        "runtime": "self-contained CANNBench wheel with repository overlays",
        "platform": PLATFORM,
        "method": "QEMU host dispatch + pyasc codegen/passes/AscendC translation",
        "limitations": [
            "does not execute numerical code",
            "does not estimate camodel ticks or NPU performance",
            "CANNBench remains the acceptance oracle when submissions are available",
        ],
        "cases": len(cases),
        "dispatch_passed": dispatch_passed,
        "compile_passed": compile_passed,
        "unique_specializations": len(compiled),
        "unique_specializations_passed": unique_passed,
        "status": "passed" if dispatch_passed == len(cases)
                  and compile_passed == len(cases) else "failed",
        "case_results": case_results,
        "specializations": list(compiled.values()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--op", required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = args.cases or Path("integrations/cannbench/tasks") / args.op / "cases.yaml"
    try:
        report = evaluate(args.candidate, args.op, cases)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "operator": args.op,
            "candidate": str(args.candidate),
            "pyasc_commit": PYASC_COMMIT,
            "platform": PLATFORM,
            "status": "failed",
            "fatal_error": f"{type(exc).__name__}: {exc}",
        }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
