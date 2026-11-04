"""Numerical preflight in the pinned CI simulator, separate from NPU acceptance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import torch
from asc.experimental import asctile
import asc.runtime.config as config

source, output = map(Path, sys.argv[1:])
sys.modules['asctile'] = asctile  # Test-only import location for the newer CI runtime.
package = types.ModuleType('generated_candidate'); package.__path__ = [str(source)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(package.__name__ + '.swi_glu', source / 'swi_glu.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
config.set_platform(config.Backend.Model, config.Platform.Ascend950PR_9599)
module.ensure_npu_platform = lambda: None
values = [float('-inf'), float('inf'), float('nan'), -100., -1., -0., 0., 1., 100.]
pairs = torch.tensor([[a, b] for a in values for b in values])
results = []
torch.manual_seed(19)
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    # 1025 output columns exercise a full 1024-column tile plus a tail; a
    # short-row-only smoke cannot detect a tail incorrectly restarting at zero.
    for shape, dim in (((129, 2), 1), ((3, 2050), -1), ((6, 32), 0), ((3, 4, 32), 1)):
        x = (torch.rand(shape) * 100 - 50).to(dtype)
        if shape == (129, 2):
            x[:len(pairs)] = pairs.to(dtype)
        a, b = x.float().chunk(2, dim=dim)
        expected = (torch.nn.functional.silu(a) * b).to(dtype)
        actual = module.swi_glu(x, dim)
        try:
            torch.testing.assert_close(actual, expected, equal_nan=True)
        except AssertionError as exc:
            raise AssertionError(f'{dtype}, shape={shape}, dim={dim}: {exc}') from exc
        results.append({'shape': list(shape), 'dtype': str(dtype), 'dim': dim, 'passed': True})
        print('PASS', results[-1], flush=True)
output.write_text(json.dumps({'passed': True, 'cases': results,
    'candidate_sha256': hashlib.sha256((source/'swi_glu.py').read_bytes()).hexdigest(),
    'scope': 'representative simulator checks with import adapter; exact packaged runtime qualified separately'}, indent=2)+'\n')
