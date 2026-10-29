#!/usr/bin/env python3
"""Defines a kernel and never launches it. Harness contract: FAIL.

Guards against a harness that accepts a clean exit as evidence of execution.
"""

from _base import make_inputs  # noqa: F401  (imports the jit kernels too)

if __name__ == "__main__":
    x, y = make_inputs()
    print("set up inputs but launched nothing")
