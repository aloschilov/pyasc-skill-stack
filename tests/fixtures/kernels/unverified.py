#!/usr/bin/env python3
"""Launches correctly but never checks the result.

Harness contract: PASS at verification_level "executed", and never
"numerically_verified" -- there is no check here that could have caught a wrong
answer.
"""

from _base import add_kernel, launch, make_inputs

if __name__ == "__main__":
    x, y = make_inputs()
    out = launch(add_kernel, x, y)
    print(f"computed {out.numel()} elements, checked nothing")
