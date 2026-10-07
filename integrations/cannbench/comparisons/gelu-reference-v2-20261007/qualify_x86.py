"""All20 host-dispatch/lowering with mocked device; NOT numerical execution."""
import contextlib
import ctypes
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent


class Tensor:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype
        self.device = types.SimpleNamespace(type='npu', index=0)

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return True

    def dim(self):
        return len(self.shape)

    def reshape(self, *_):
        return Tensor([self.numel()], self.dtype)


fake_torch = types.ModuleType('torch')
fake_torch.Tensor = Tensor
for name in ('float16', 'bfloat16', 'float32'):
    setattr(fake_torch, name, 'torch.'+name)
fake_torch.empty_like = lambda x: Tensor(x.shape, x.dtype)
sys.modules['torch'] = fake_torch


def main():
    import yaml
    from asc.experimental import asctile
    from asc._C import ir
    from asc.common.compat import get_annotations
    from asc.codegen.specialization import Specialization
    from cann_bench._device_resources import Resources
    from cann_bench._runtime_context import LaunchContext
    from cann_bench._gelu_metadata.gate import load_metadata
    module = importlib.import_module('cann_bench.gelu')
    original = module.gelu_kernel
    asctile.set_platform(asctile.Backend.Model, asctile.Platform.Ascend950PR_9599, check=False)
    assert Path(importlib.import_module('asc').__file__).is_relative_to('/out/installed')
    assert hashlib.sha256(Path(inspect.getsourcefile(original.fn)).read_bytes()).hexdigest() == hashlib.sha256((ROOT/'kernel.py').read_bytes()).hexdigest()
    # Real SDK bridge load and loaded-library identity; no active-device query.
    assert load_metadata() is not None
    provenance = json.loads(Path('/out/installed/cann_bench/provenance.json').read_text())
    assert provenance['pyasc_pin'] == '9069108e323746187c48d78fec4929c4afa2efc4'
    assert provenance['kernel_sha256'] == hashlib.sha256((ROOT/'kernel.py').read_bytes()).hexdigest()
    results, specializations = [], {}
    launches = []

    class Capture:
        def __getitem__(self, launch):
            def invoke(*args, **kwargs):
                bound = inspect.signature(original.fn).bind(*args).arguments
                runtime_args, constants = original.split_args(bound, get_annotations(original.fn))
                options = dict(original.default_options, verify_sync=True)
                options.update(kwargs)
                codegen = original.extract_kwargs(original.codegen.options_cls, options)
                compiler_options = original.extract_kwargs(original.compiler.options_cls, options)
                assert not options
                arg_types = {k: original.get_arg_type(v) for k, v in runtime_args.items()}
                key = (str(args[0].dtype), args[4], args[5], args[3], args[6])
                if str(key) not in specializations:
                    mod = original._run_codegen(Specialization(arg_types, constants), codegen)
                    compiler = original.compiler(compiler_options)
                    compiler.preprocess_module(mod)
                    compiler.run_passes(mod)
                    compiler.postprocess_module(mod)
                    source = compiler.run_translation(mod)
                    ub = mod.op.get_dict_of_int_attr(ir.attr.memory_consumed)['UB']
                    assert ub == 190464
                    path = Path('/out') / f'specialization-{len(specializations)}.cpp'
                    path.write_text(source)
                    specializations[str(key)] = dict(ub=ub, options=dataclasses.asdict(compiler_options),
                                                    source_sha256=hashlib.sha256(source.encode()).hexdigest())
                assert launch[1] is stream
                launches.append(dict(cores=launch[0], tile=args[3], unroll=args[6], specialization=str(key)))
            return invoke

    module.gelu_kernel = Capture()
    module.ensure_npu_platform = lambda: None
    module.record_launch = lambda *a, **kw: None
    module.record_capacity = lambda *a, **kw: None
    module.validate_capacity = lambda *a, **kw: dict(allocation_capacity_bytes=253952, qualified=True)
    stream = ctypes.c_void_p(123)

    @contextlib.contextmanager
    def context(resources):
        yield LaunchContext(resources, stream, resources.launch_cap)

    module.launch_context = context
    cases = yaml.safe_load((ROOT/'inputs/cases.yaml').read_text())['cases']
    for cap in (1, 8, 64, 72):
        resources = Resources(0, cap, cap, cap, None, cap, False, ())
        module.query_resources = lambda device: resources
        for case in cases:
            dtype = getattr(fake_torch, case['dtype'][0])
            x = Tensor(case['input_shape'][0], dtype)
            before = len(launches)
            result = module.gelu(x, approximate=case['attrs']['approximate'])
            assert result.shape == x.shape and result.dtype == dtype and len(launches) == before+1
            launch = launches[-1]
            assert launch['cores'] == min(cap, math.ceil(x.numel()/launch['tile']))
            results.append(dict(cap=cap, case_id=case['case_id'], passed=True, **launch))
    Path('/out/x86-results.json').write_text(json.dumps(dict(passed=True, dispatches=results,
        bridge_loadability=True, bridge_device_binding_tested=False, provenance=provenance,
        specializations=specializations, scope='Mocked host launch and stock lowering, not device accuracy'), indent=2, default=str))
    print('Passed', len(results), 'host dispatches;', len(specializations), 'stock lowerings')


if __name__ == '__main__':
    main()
