#!/usr/bin/env python3
"""Run the shared compile gate while recording the exact current v2 commit."""

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parents[3]
sys.path.insert(0, str(REPO / "integrations/cannbench/workers"))

import local_compile_gate as gate  # noqa: E402


gate.PYASC_COMMIT = "0a631f70968c3cb7c33ce45330a85768dd5a6f06"


if __name__ == "__main__":
    raise SystemExit(gate.main())
