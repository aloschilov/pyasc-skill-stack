#!/usr/bin/env python3
"""Launches successfully, then raises. Harness contract: FAIL.

Guards against a harness that reports the launch it observed and ignores how
the run ended.
"""

from _base import add_kernel, launch, make_inputs

if __name__ == "__main__":
    x, y = make_inputs()
    launch(add_kernel, x, y)
    raise RuntimeError("deliberate post-launch failure")
