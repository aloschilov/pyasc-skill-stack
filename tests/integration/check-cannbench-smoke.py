#!/usr/bin/env python3
"""Simulator regression for short-row copies, tails and IEEE special values.

Runs in the pinned CI image. Its newer PyASC exposes AscTile under
asc.experimental; only the test import is adapted. This is not qualification
of the separately supplied CANNBench runtime or its NPU runner.
"""
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch
from asc.experimental import asctile
import asc.runtime.config as config

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'integrations/cannbench/smoke-kernel'
sys.modules['asctile'] = asctile
package = types.ModuleType('smoke_candidate')
package.__path__ = [str(SOURCE)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location('smoke_candidate.swi_glu', SOURCE / 'swi_glu.py')
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)
config.set_platform(config.Backend.Model, config.Platform.Ascend950PR_9599)
kernel.ensure_npu_platform = lambda: None
kernel._MAX_CORES = 2

torch.manual_seed(19)
values = [float('-inf'), float('inf'), float('nan'), -100., -1., -0., 0., 1., 100.]
pairs = torch.tensor([[a, b] for a in values for b in values])
passed = 0
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    for half_cols in (1, 7):
        # 129 rows crosses both a block boundary and the final partial tile.
        x = (torch.rand(129, 2 * half_cols) * 150 - 50).to(dtype)
        x[:len(pairs), 0] = pairs[:, 0].to(dtype)
        x[:len(pairs), half_cols] = pairs[:, 1].to(dtype)
        actual = kernel.swi_glu(x, -1)
        left, right = x.float().chunk(2, dim=-1)
        expected = (torch.nn.functional.silu(left) * right).to(dtype)
        torch.testing.assert_close(actual, expected, equal_nan=True)
        passed += 1
        print(f'PASS {dtype} shape={list(x.shape)} tails and special values', flush=True)
print(f'Passed {passed}/6 simulator regressions; NPU qualification remains separate.')

# CaModel tensors skip the standalone template's NPU setup and stream lookup.
# Run that host branch with the real AscTile API and a recorded launch.
template_path = ROOT / 'skills/asctile-api-patterns/templates/streamed-swiglu-flat.py'
spec = importlib.util.spec_from_file_location('swiglu_template', template_path)
template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template)
x = torch.ones(2, 14)
device = types.SimpleNamespace(type='npu', index=1)
with (
    patch.object(torch.Tensor, 'device', device),
    patch.object(torch, 'npu', create=True) as npu,
    patch.object(asctile, 'set_platform', autospec=True) as set_platform,
    patch.object(template, 'swiglu_flat') as launch,
):
    npu.current_stream.return_value.npu_stream = 12345
    result = template.swi_glu(x)
    set_platform.assert_called_once_with(asctile.Backend.NPU)
    npu.current_stream.assert_called_once_with(device)
    blocks, stream = launch.__getitem__.call_args.args[0]
    assert blocks == 1 and stream.value == 12345
    launch.__getitem__.return_value.assert_called_once()
    assert tuple(result.shape) == (2, 7)
print('PASS standalone NPU wrapper platform selection and caller stream (mock launch).')
