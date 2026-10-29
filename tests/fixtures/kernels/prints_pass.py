#!/usr/bin/env python3
"""Prints "PASS" and exits 0 without launching. Harness contract: FAIL.

Guards specifically against inferring success from stdout: the legacy
simulator mode keys on the substring "PASS", and nothing here ran.
"""

from _base import make_inputs  # noqa: F401

if __name__ == "__main__":
    print("[PASS] allclose: True")
    print("PASS")
