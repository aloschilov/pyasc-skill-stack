"""Logging a failed comparison before raising must preserve attribution."""
import torch
from _base import add_kernel, launch, make_inputs

if __name__ == '__main__':
    x, y = make_inputs()
    out = launch(add_kernel, x, y)
    if not torch.allclose(out, x + y, atol=1e-5, rtol=1e-5):
        print('Output differs from x + y')
        raise ValueError('verification failed')
