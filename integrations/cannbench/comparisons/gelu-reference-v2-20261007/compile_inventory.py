"""Compile finite dtype/mode/tile inventory and map all official shapes."""
import dataclasses
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parent


def main():
    import yaml
    from asc.experimental import asctile
    from asc.common.compat import get_annotations
    from asc.runtime.jit import CompilePrereqs, MockTensor
    from kernel import gelu_kernel
    asctile.set_platform(asctile.Backend.Model, asctile.Platform.Ascend950PR_9599, check=False)
    inventory = []
    for dtype in ('float16', 'bfloat16', 'float32'):
        for mode in ('none', 'tanh'):
            for tile in (2048, 8192, 15872):
                row = dict(dtype=dtype, mode=mode, tile_shape=[tile], unroll=1, passed=False)
                out = ROOT/'inventory'/f'{dtype}-{mode}-{tile}'
                out.mkdir(parents=True, exist_ok=True)
                os.environ['PYASC_DUMP_PATH'] = str(out)
                try:
                    x = MockTensor(getattr(asctile, dtype))
                    args = (x, x, 8193, tile, mode == 'tanh', mode == 'none' and dtype == 'float32', 1)
                    bound = inspect.signature(gelu_kernel.fn).bind(*args).arguments
                    runtime_args, constants = gelu_kernel.split_args(bound, get_annotations(gelu_kernel.fn))
                    config = dict(gelu_kernel.default_options, verify_sync=True, always_compile=True)
                    codegen = gelu_kernel.extract_kwargs(gelu_kernel.codegen.options_cls, config)
                    options = gelu_kernel.extract_kwargs(gelu_kernel.compiler.options_cls, config)
                    assert not config
                    compiled = gelu_kernel._compile_kernel(CompilePrereqs(
                        {k: gelu_kernel.get_arg_type(v) for k, v in runtime_args.items()}, constants, codegen, options))
                    ub = compiled.meta.memory_consumed.get('UB')
                    row.update(ub_bytes=ub, passed=ub is not None and ub <= 253952,
                               compiler_options=dataclasses.asdict(options),
                               binary_sha256=hashlib.sha256(compiled.binary).hexdigest())
                except Exception:
                    row['error'] = traceback.format_exc()
                inventory.append(row)
                (ROOT/'inventory.json').write_text(json.dumps(inventory, indent=2, default=str))
                print({k: row.get(k) for k in ('dtype', 'mode', 'tile_shape', 'passed', 'ub_bytes', 'error')}, flush=True)
    cases = yaml.safe_load((ROOT/'inputs/cases.yaml').read_text())['cases']
    mapping = []
    for case in cases:
        route = [r for r in inventory if r['passed'] and r['dtype'] == case['dtype'][0]
                 and r['mode'] == case['attrs']['approximate']]
        winner = max(route, key=lambda r: r['tile_shape'][0])
        elements = math.prod(case['input_shape'][0])
        tiles = math.ceil(elements/winner['tile_shape'][0])
        mapping.append(dict(case_id=case['case_id'], shape=case['input_shape'][0], elements=elements,
                            **winner, launched_at_cap64=min(64, tiles), launched_at_cap72=min(72, tiles)))
    (ROOT/'case-geometries.json').write_text(json.dumps(mapping, indent=2, default=str))


if __name__ == '__main__':
    main()
