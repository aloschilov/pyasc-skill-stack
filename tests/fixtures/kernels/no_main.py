#!/usr/bin/env python3
"""Defines a kernel with no __main__ block at all. Harness contract: FAIL.

Guards against entry-point discovery that invents work to do, and against
importing a module being mistaken for running it.
"""

from _base import add_kernel, launch, make_inputs  # noqa: F401


def unused_entry_point():
    x, y = make_inputs()
    return launch(add_kernel, x, y)
