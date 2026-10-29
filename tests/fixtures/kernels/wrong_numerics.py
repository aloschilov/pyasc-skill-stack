#!/usr/bin/env python3
"""Computes x - y but verifies against x + y. Harness contract: FAIL.

Guards against a harness that treats "the script ran" as "the results are
right".
"""

import torch

from _base import launch, make_inputs, sub_kernel

if __name__ == "__main__":
    x, y = make_inputs()
    out = launch(sub_kernel, x, y)
    if not torch.allclose(out, x + y, atol=1e-5, rtol=1e-5):
        raise ValueError("verification failed: output differs from x + y")
    print("[PASS] verified")
