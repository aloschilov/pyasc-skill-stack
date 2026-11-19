#!/usr/bin/env python3
"""Simulator regression for short-row copies, tails and IEEE special values.

Runs in the pinned CI image. Its newer PyASC exposes AscTile under
asc.experimental; only the test import is adapted. This is not qualification
of the separately supplied CANNBench runtime or its NPU runner.
"""
import importlib.util
from contextlib import contextmanager
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
# A non-final split flattens H=303, exercising the template's padded tail.
for seed in (19, 23):
    torch.manual_seed(seed)
    x = torch.randn(2, 6, 101, dtype=torch.bfloat16)
    with patch.object(torch, 'npu', create=True) as npu:
        actual = template.swi_glu(x, 1)
        assert not npu.mock_calls, 'CPU CaModel must not enter the NPU route'
    left, right = x.float().chunk(2, dim=1)
    torch.testing.assert_close(actual, (torch.nn.functional.silu(left) * right).to(x.dtype))
print('PASS standalone CPU CaModel BF16 middle-dimension tail on seeds 19/23.')

x = torch.ones(2, 14)
device = types.SimpleNamespace(type='npu', index=1)
active_device = 0
new_empty = torch.Tensor.new_empty

@contextmanager
def device_guard(selected):
    global active_device
    previous = active_device
    active_device = selected.index
    try:
        yield
    finally:
        active_device = previous

def allocate(tensor, shape):
    assert active_device == 1, 'Allocation precedes device selection'
    return new_empty(tensor, shape)

def select_platform(backend):
    assert active_device == 1, 'Platform queries use the wrong device'

def current_stream(selected):
    assert active_device == selected.index == 1
    return types.SimpleNamespace(npu_stream=12345)

for fail_launch in (False, True):
    failure = RuntimeError('mock launch failure')
    def invoke(*args):
        assert active_device == 1, 'Launch uses the wrong device'
        if fail_launch:
            raise failure
    with (
        patch.object(torch.Tensor, 'device', device),
        patch.object(torch.Tensor, 'new_empty', allocate),
        patch.object(torch, 'npu', create=True) as npu,
        patch.object(asctile, 'set_platform', autospec=True, side_effect=select_platform) as set_platform,
        patch.object(template, 'swiglu_flat') as launch,
    ):
        npu.device.side_effect = device_guard
        npu.current_stream.side_effect = current_stream
        launch.__getitem__.return_value.side_effect = invoke
        try:
            result = template.swi_glu(x)
        except RuntimeError as exc:
            assert fail_launch and exc is failure
        else:
            assert not fail_launch
            assert tuple(result.shape) == (2, 7)
        assert active_device == 0, 'Caller device was not restored'
        npu.device.assert_called_once_with(device)
        set_platform.assert_called_once_with(asctile.Backend.NPU)
        npu.current_stream.assert_called_once_with(device)
        blocks, stream = launch.__getitem__.call_args.args[0]
        assert blocks == 1 and stream.value == 12345
        launch.__getitem__.return_value.assert_called_once()
print('PASS standalone NPU device, platform and stream; restored after success/error (mock launch).')
