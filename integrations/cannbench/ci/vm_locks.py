#!/usr/bin/env python3
"""Run CANN work under the pre-existing shared VM locks without replacing them."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import subprocess
import sys
from contextlib import ExitStack
from contextlib import contextmanager

VM_ROOT = Path("/home/aloschilov/workspace/pyasc-skill-stack")
EXECUTION_LOCKS = [VM_ROOT / "integrations/cannbench/comparisons/gelu-adaptive-20260910" / name
                   for name in ("screen.lock", "local.lock")]
UPLOAD_LOCK = VM_ROOT / "integrations/cannbench/comparisons/gelu-geometry-strategy-20260908/artifacts/remote.lock"


@contextmanager
def shared_locks(kind):
    if platform.system() != "Linux" or not VM_ROOT.is_dir():
        raise ValueError("CANN execution is restricted to the designated Linux VM")
    paths = EXECUTION_LOCKS if kind == "execution" else [UPLOAD_LOCK]
    with ExitStack() as stack:
        for path in paths:
            handle = stack.enter_context(path.open("rb"))
            print("Waiting for shared " + kind + " lock: " + path.name, flush=True, file=sys.stderr)
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("execution", "upload"), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if platform.system() != "Linux" or not VM_ROOT.is_dir():
        parser.error("CANN execution is restricted to the designated Linux VM")
    if not command:
        parser.error("A command is required")
    with shared_locks(args.kind):
        return subprocess.run(command, env={**os.environ, "PYASC_CANN_LOCK_HELD": args.kind}).returncode


if __name__ == "__main__":
    raise SystemExit(main())
