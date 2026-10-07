"""Isolated compile/compiled-kernel qualification; never submits remotely."""
import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PIN = "9069108e323746187c48d78fec4929c4afa2efc4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], required=True)
    parser.add_argument('--mode', choices=['none', 'tanh'], required=True)
    parser.add_argument('--tile', type=int, default=2048)
    parser.add_argument('--size', type=int, default=8193)
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--unroll', type=int, default=1)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--workload-size', type=int, default=0,
                        help='Separate multi-tile workload, not the sampled-range sweep')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['PYASC_CACHE_DIR'] = str(args.output.parent / 'cache')
    os.environ['PYASC_DUMP_PATH'] = str(args.output.parent / 'dump')
    result = dict(pin=PIN, args=vars(args), completed=False, compiled=False,
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save():
        args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')

    try:
        import torch
        import yaml
        import asc
        from asc.experimental import asctile
        from asc.lib import runtime
        from asc.common.compat import get_annotations
        from asc.runtime.jit import CompilePrereqs, MockTensor
        from kernel import gelu_kernel
        from inputs.precision import compare_tensors
        torch.set_num_threads(2)
        result.update(runtime_file=asc.__file__, asctile_file=asctile.__file__,
                      source_sha256=hashlib.sha256((ROOT/'kernel.py').read_bytes()).hexdigest())
        asctile.set_platform(asctile.Backend.Model, asctile.Platform.Ascend950PR_9599,
                             check=not args.compile_only)
        dtype = getattr(torch, args.dtype)
        x = MockTensor(getattr(asctile, args.dtype))
        values = (x, x, args.size, args.tile, args.mode == 'tanh',
                  args.mode == 'none' and args.dtype == 'float32', args.unroll)
        bound = inspect.signature(gelu_kernel.fn).bind(*values).arguments
        runtime_args, constants = gelu_kernel.split_args(bound, get_annotations(gelu_kernel.fn))
        config = dict(gelu_kernel.default_options, verify_sync=True, always_compile=True)
        codegen = gelu_kernel.extract_kwargs(gelu_kernel.codegen.options_cls, config)
        options = gelu_kernel.extract_kwargs(gelu_kernel.compiler.options_cls, config)
        assert not config, config
        result['requested_options'] = dataclasses.asdict(options)
        prerequisites = CompilePrereqs({k: gelu_kernel.get_arg_type(v) for k, v in runtime_args.items()},
                                       constants, codegen, options)
        compiled = gelu_kernel._compile_kernel(prerequisites)
        ub = compiled.meta.memory_consumed.get('UB')
        result.update(compiled=True, memory_consumed=compiled.meta.memory_consumed,
                      binary_sha256=hashlib.sha256(compiled.binary).hexdigest(), ub_fit=ub is not None and ub <= 253952)
        save()
        if args.compile_only:
            result['completed'] = True
            return
        if not result['ub_fit']:
            raise RuntimeError('Missing or excessive UB allocation')
        checks = []
        cases = yaml.safe_load((ROOT/'inputs/cases.yaml').read_text())['cases']
        ranges = sorted({tuple(c['value_range']) for c in cases
                         if c['dtype'][0] == args.dtype and c['attrs']['approximate'] == args.mode})
        generator = torch.Generator().manual_seed(20261007)
        vectors = [('random', torch.randn(1025, generator=generator))]
        for lo, hi in ranges:
            if math.isnan(lo):
                vector = torch.full((513,), float('nan'))
            elif not math.isfinite(lo) or not math.isfinite(hi):
                vector = torch.tensor([float('-inf'), float('inf'), 0.0]).repeat(171)
            else:
                vector = torch.linspace(lo, hi, 513)
            vectors.append((f'range_{lo}_{hi}', vector))
        vectors += [('negative_tail', torch.linspace(-8, -2, 2049))]
        if args.workload_size:
            vectors = [('multi_tile_workload', torch.linspace(-3, 3, args.workload_size))]
        segments = []
        start = 0
        for label, vector in vectors:
            segments.append((label, start, start+vector.numel()))
            start += vector.numel()
        joined = torch.cat([vector for _, vector in vectors])
        result['sample_segments'] = segments
        for label, vector in [('combined', joined)]:
            x = vector.to(dtype)
            size = x.numel()
            backing = torch.full((size + 256,), 123, dtype=dtype)
            y = backing[:size]
            expected64 = torch.nn.functional.gelu(x.double(), approximate=args.mode)
            expected_native = torch.nn.functional.gelu(x, approximate=args.mode)
            previous = None
            for repeat in range(2):
                y.fill_(float('nan'))
                result.update(phase='model', active_input=label, repeat=repeat)
                save()
                started = time.monotonic()
                before_tick = runtime.current_tick()
                gelu_kernel[args.cores](x, y, size, args.tile, args.mode == 'tanh',
                                      args.mode == 'none' and args.dtype == 'float32', args.unroll,
                                      verify_sync=True)
                ticks = runtime.current_tick() - before_tick
                check = compare_tensors(y, expected64, dtype=args.dtype, cpu_output=expected_native)
                equal = previous is None or bool(((previous == y) | (torch.isnan(previous) & torch.isnan(y))).all())
                guards = bool((backing[size:] == 123).all())
                segment_checks = []
                for segment, begin, end in segments:
                    checked = compare_tensors(y[begin:end], expected64[begin:end], dtype=args.dtype,
                                              cpu_output=expected_native[begin:end])
                    segment_checks.append(dict(input=segment, passed=checked.passed,
                                               details=checked.to_dict()))
                checks.append(dict(input=label, repeat=repeat, passed=check.passed,
                                   repeat_equal=equal, guards_intact=guards,
                                   model_ticks=ticks, warmed=repeat > 0,
                                   wall_seconds=time.monotonic()-started, details=check.to_dict(),
                                   segment_checks=segment_checks))
                if not check.passed or not all(s['passed'] for s in segment_checks):
                    torch.save(dict(x=x, y=y), args.output.parent/f'failure-{repeat}.pt')
                previous = y.clone()
                result['checks'] = checks
                save()
        result.update(completed=True, accuracy_passed=all(c['passed'] and all(s['passed'] for s in c['segment_checks']) and c['repeat_equal'] and
                                                        c['guards_intact'] for c in checks))
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        save()
        print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'requested_options')}, default=str))


if __name__ == '__main__':
    main()
