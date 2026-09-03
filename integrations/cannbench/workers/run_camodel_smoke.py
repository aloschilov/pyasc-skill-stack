#!/usr/bin/env python3.10
"""Representative numerical/tick smoke for a generated CANNBench bundle.

Run only with a native build of the same pinned pyasc v2 source. This is not a
replacement for the 20 official cases per operator or for real-NPU profiling.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import asc.runtime.config as config
from asc.lib import runtime as rt


ALL_OPS = (
    "sigmoid", "exp", "mish", "gelu", "masked_scale", "swi_glu",
    "foreach_addcdiv_scalar", "foreach_norm", "rms_norm",
)


def close(actual, expected, *, rtol=2e-3, atol=2e-3) -> None:
    if isinstance(actual, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=rtol, atol=atol,
                                       equal_nan=True)
    else:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol,
                                   equal_nan=True)


def run_case(op: str, package) -> None:
    x = torch.linspace(-2.0, 2.0, 64, dtype=torch.float32)
    if op == "sigmoid":
        close(package.sigmoid(x), torch.sigmoid(x))
    elif op == "exp":
        close(package.exp(x, base=2.0, scale=0.5, shift=-0.25),
              torch.exp((x * 0.5 - 0.25) * math.log(2.0)))
    elif op == "mish":
        close(package.mish(x), F.mish(x), rtol=4e-3, atol=4e-3)
    elif op == "gelu":
        close(package.gelu(x, approximate="tanh"),
              F.gelu(x, approximate="tanh"), rtol=4e-3, atol=4e-3)
    elif op == "masked_scale":
        mask = (torch.arange(64) % 5).to(torch.float32)
        close(package.masked_scale(x, mask, scale=0.125), x * mask * 0.125)
    elif op == "swi_glu":
        value = torch.linspace(-2.0, 2.0, 128, dtype=torch.float32).reshape(2, 64)
        x0, x1 = value.float().chunk(2, dim=-1)
        close(package.swi_glu(value, dim=-1), F.silu(x0) * x1,
              rtol=4e-3, atol=4e-3)
    elif op == "foreach_addcdiv_scalar":
        x1 = [x.reshape(8, 8)]
        x2 = [(x + 0.25).reshape(8, 8)]
        x3 = [torch.linspace(0.5, 2.0, 64).reshape(8, 8)]
        close(package.foreach_addcdiv_scalar(x1, x2, x3, 0.5),
              [x1[0] + x2[0] / x3[0] * 0.5], rtol=4e-3, atol=4e-3)
    elif op == "foreach_norm":
        value = x.reshape(8, 8)
        close(package.foreach_norm([value], 2.0), [torch.norm(value, p=2)],
              rtol=4e-3, atol=4e-3)
    elif op == "rms_norm":
        value = x.reshape(4, 16)
        gamma = torch.linspace(0.5, 1.5, 16)
        expected = value * torch.rsqrt((value * value).mean(-1, keepdim=True) + 1e-6) * gamma
        close(package.rms_norm(value, gamma, 1e-6), expected,
              rtol=4e-3, atol=4e-3)
    else:
        raise ValueError(op)


def run_critical_case(op: str, package) -> None:
    """Exercise dtype/control-flow paths missed by the basic f32 smoke."""
    x = torch.linspace(-3.0, 3.0, 64, dtype=torch.float32)
    if op == "sigmoid":
        value = x.to(torch.float16)
        close(package.sigmoid(value), torch.sigmoid(value), rtol=1e-2, atol=1e-2)
    elif op == "exp":
        value = x.to(torch.bfloat16)
        close(package.exp(value, base=-1.0, scale=-0.25, shift=0.5),
              torch.exp(value.float() * -0.25 + 0.5).to(value.dtype),
              rtol=2e-2, atol=2e-2)
    elif op == "mish":
        value = x.to(torch.bfloat16)
        close(package.mish(value), F.mish(value.float()).to(value.dtype),
              rtol=2e-2, atol=2e-2)
    elif op == "gelu":
        value = x.to(torch.float16)
        close(package.gelu(value, approximate="none"),
              F.gelu(value, approximate="none"), rtol=1e-2, atol=1e-2)
    elif op == "masked_scale":
        signed = ((torch.arange(64) % 7) - 3).to(torch.int8)
        close(package.masked_scale(x, signed, scale=-0.25),
              (x * signed * -0.25).to(x.dtype), rtol=4e-3, atol=4e-3)
        unsigned = (torch.arange(64) * 5).to(torch.uint8)
        close(package.masked_scale(x, unsigned, scale=0.125),
              (x * unsigned * 0.125).to(x.dtype), rtol=4e-3, atol=4e-3)
    elif op == "swi_glu":
        value = torch.linspace(-4.0, 4.0, 128, dtype=torch.bfloat16).reshape(64, 2)
        x0, x1 = value.float().chunk(2, dim=1)
        close(package.swi_glu(value, dim=1), (F.silu(x0) * x1).to(value.dtype),
              rtol=2e-2, atol=2e-2)
    elif op == "foreach_addcdiv_scalar":
        x1 = [x.to(torch.bfloat16)]
        x2 = [(x + 0.5).to(torch.bfloat16)]
        x3 = [torch.linspace(0.5, 2.0, 64).to(torch.bfloat16)]
        expected = [(x1[0].float() + x2[0].float() / x3[0].float() * -0.5).to(x1[0].dtype)]
        close(package.foreach_addcdiv_scalar(x1, x2, x3, -0.5), expected,
              rtol=2e-2, atol=2e-2)
    elif op == "foreach_norm":
        value = x.reshape(8, 8)
        close(package.foreach_norm([value], math.inf),
              [torch.norm(value, p=math.inf)], rtol=4e-3, atol=4e-3)
        close(package.foreach_norm([value], 1.5),
              [torch.norm(value, p=1.5)], rtol=4e-3, atol=4e-3)
    elif op == "rms_norm":
        value = torch.linspace(-2.0, 2.0, 68, dtype=torch.bfloat16).reshape(4, 17)
        gamma = torch.linspace(0.5, 1.5, 17, dtype=torch.bfloat16)
        expected = (
            value.float()
            * torch.rsqrt((value.float() * value.float()).mean(-1, keepdim=True) + 1e-5)
            * gamma.float()
        ).to(value.dtype)
        close(package.rms_norm(value, gamma, 1e-5), expected,
              rtol=2e-2, atol=2e-2)
    else:
        raise ValueError(op)


def _strict_close(actual, expected, dtype: torch.dtype) -> None:
    """Use the task's dtype-scale tolerance and require IEEE positions."""
    tolerance = {
        torch.float32: 2.0 ** -13,
        torch.float16: 2.0 ** -10,
        torch.bfloat16: 2.0 ** -7,
    }[dtype]
    close(actual, expected, rtol=tolerance, atol=tolerance)
    actual_values = actual if isinstance(actual, list) else [actual]
    expected_values = expected if isinstance(expected, list) else [expected]
    for left, right in zip(actual_values, expected_values):
        assert torch.equal(torch.isnan(left), torch.isnan(right)), (
            "NaN positions differ")
        assert torch.equal(torch.isinf(left), torch.isinf(right)), (
            "Inf positions differ")


def run_adversarial_case(op: str, package) -> None:
    """Exercise CANNBench failure classes at simulator-sized shapes."""
    if op == "gelu":
        routes = (
            (torch.float32, "none", -0.5, 0.5),
            (torch.float32, "none", -88.0, 88.0),
            (torch.float32, "none", -20.0, 40.0),
            (torch.float32, "tanh", -5.0, 10.0),
            (torch.float32, "tanh", -100.0, 100.0),
            (torch.float16, "none", -1.0, 2.0),
            (torch.float16, "tanh", -1000.0, 1000.0),
            (torch.bfloat16, "none", -3.0, 3.0),
            (torch.bfloat16, "tanh", -3.0, 6.0),
        )
        for dtype, approximate, low, high in routes:
            value = torch.linspace(low, high, 4096, dtype=torch.float32).to(dtype)
            expected = F.gelu(value, approximate=approximate)
            _strict_close(package.gelu(value, approximate=approximate),
                          expected, dtype)
        special = torch.tensor(
            [float("-inf"), -100.0, -5.0, -0.0, 0.0, 5.0, 100.0,
             float("inf"), float("nan")], dtype=torch.bfloat16)
        _strict_close(package.gelu(special, approximate="tanh"),
                      F.gelu(special, approximate="tanh"), special.dtype)
    elif op == "foreach_addcdiv_scalar":
        routes = (
            (torch.float32, 1.0, -1.0, 1.0, 0.5, 1.0),
            (torch.float16, 0.0, -1000.0, 1000.0, 0.1, 1000.0),
            (torch.bfloat16, -0.5, -3.0, 6.0, 0.1, 6.0),
        )
        for dtype, scalar, low, high, div_low, div_high in routes:
            a = torch.linspace(low, high, 4096, dtype=torch.float32).to(dtype)
            b = torch.linspace(high, low, 4096, dtype=torch.float32).to(dtype)
            c = torch.linspace(div_low, div_high, 4096,
                               dtype=torch.float32).to(dtype)
            expected = [torch.addcdiv(a, b, c, value=scalar)]
            actual = package.foreach_addcdiv_scalar([a], [b], [c], scalar)
            _strict_close(actual, expected, dtype)
        nan = torch.full((64,), float("nan"), dtype=torch.float32)
        one = torch.ones(64, dtype=torch.float32)
        _strict_close(
            package.foreach_addcdiv_scalar([nan], [nan], [nan], float("nan")),
            [torch.addcdiv(nan, nan, nan, value=float("nan"))],
            torch.float32)
    else:
        raise ValueError(
            f"adversarial suite is currently defined only for gelu and "
            f"foreach_addcdiv_scalar, got {op}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--ops", default=",".join(ALL_OPS))
    parser.add_argument(
        "--suite", choices=("basic", "critical", "adversarial"),
        default="basic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ops = tuple(value.strip() for value in args.ops.split(",") if value.strip())
    sys.path.insert(0, str(args.candidate_root.parent.resolve()))

    config.set_platform(config.Backend.Model, config.Platform.Ascend950PR_9599)
    package = importlib.import_module(args.candidate_root.name)
    # Submission wrappers correctly select NPU in production. Camodel owns the
    # process here, so disable only that host switch; kernels are unchanged.
    for op in ops:
        getattr(package, op).__globals__["ensure_npu_platform"] = lambda: None

    reports = []
    for op in ops:
        before = rt.current_tick()
        try:
            runner = {
                "basic": run_case,
                "critical": run_critical_case,
                "adversarial": run_adversarial_case,
            }[args.suite]
            runner(op, package)
            status = "passed"
            error = None
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        after = rt.current_tick()
        report = {
            "operator": op,
            "status": status,
            "camodel_ticks": after - before if before is not None and after is not None else None,
            "error": error,
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    result = {
        "schema_version": 1,
        "evidence": "verified-camodel-smoke",
        "platform": "Ascend950PR_9599",
        "suite": args.suite,
        "operators": reports,
        "status": "passed" if all(v["status"] == "passed" for v in reports) else "failed",
        "limitations": [
            (
                "one small float32 route per operator"
                if args.suite == "basic"
                else (
                    "selected dtype and control-flow routes only"
                    if args.suite == "critical"
                    else "simulator-sized adversarial routes, not full shapes"
                )
            ),
            "not the official 180-case matrix",
            "camodel ticks are not silicon performance",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
