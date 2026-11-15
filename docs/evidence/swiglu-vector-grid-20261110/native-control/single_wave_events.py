"""Compare hardware events and buffer IDs at the same W79 single-wave boundaries."""

import argparse
from hashlib import sha256
from pathlib import Path
import re

from output_l2_priority import W79
from row_dma_padding import run_control

# JIT register names in the VF can differ between otherwise identical builds.
# Pin the full surrounding scalar code, transfer arguments and buffer addresses.
SURROUNDING = {
    11: "35198bf2cc570903023d33ca5e16ab143c1464f95f6e01675cfccfb01d4152b9",
    13: "259dda4b017577823a985ffa33f09da44a9b17bdb58b4a79faaf5eac18a4643e",
}


def flag(action, edge):
    return f"AscendC::{action}Flag<AscendC::HardEvent::{edge}>(EVENT_ID0);"


def render(source, case, mode):
    surrounding, n = re.subn(r"  __VEC_SCOPE__\n.*?\n  };\n", "", source, flags=re.S)
    if n != 1 or sha256(surrounding.encode()).hexdigest() != SURROUNDING[case]:
        raise ValueError("expected the W79 single-wave transfer schedule")
    if mode == "base":
        return source
    if mode != "events":
        raise ValueError(mode)
    replacements = {}
    for buf in (0, 1):
        # Optional tail fills complete before MTE2 writes to their input UB.
        # Each conditional consumes its own flag before the ID can be reused.
        replacements[f"    get_buf(PIPE_V, {buf}, 0);"] = []
        replacements[f"    rls_buf(PIPE_V, {buf}, 0);"] = [
            flag("Set", "V_MTE2"), flag("Wait", "V_MTE2")]
        replacements[f"  get_buf(PIPE_MTE2, {buf}, 0);"] = []
        replacements[f"  rls_buf(PIPE_MTE2, {buf}, 0);"] = []
    # Both reads are on MTE2. The one VF consumes both tiles and writes its
    # result into the gate tile. There is only one output DMA and no reuse.
    replacements.update({
        "  rls_buf(PIPE_MTE2, 1, 0);": [flag("Set", "MTE2_V")],
        "  get_buf(PIPE_V, 0, 0);": [flag("Wait", "MTE2_V")],
        "  get_buf(PIPE_V, 1, 0);": [],
        "  rls_buf(PIPE_V, 1, 0);": [],
        "  rls_buf(PIPE_V, 0, 0);": [flag("Set", "V_MTE3")],
        "  get_buf(PIPE_MTE3, 0, 0);": [flag("Wait", "V_MTE3")],
        "  rls_buf(PIPE_MTE3, 0, 0);": [],
    })
    result = []
    for line in source.splitlines(keepends=True):
        key = line.rstrip("\n")
        if key in replacements:
            indent = key[:len(key) - len(key.lstrip())]
            result.extend(indent + expr + "\n" for expr in replacements.pop(key))
        else:
            result.append(line)
    if replacements:
        raise ValueError("incomplete match for the qualified schedule")
    return "".join(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("base", "events"))
    parser.add_argument("case", type=int, choices=SURROUNDING)
    for key in ("package", "dump", "original_dump", "worker"):
        parser.add_argument(key, type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sha256((args.package / "swi_glu.py").read_bytes()).hexdigest() != W79:
        raise ValueError("expected the independent W79 module")
    run_control(args, lambda src: render(src, args.case, args.mode), "events.json")
