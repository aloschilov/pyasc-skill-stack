"""Assemble the measured skill template and OpenCode's host dispatch.

Usage: python assemble.py parent.py template.py events.jsonl output.py
The parent is the UB-height candidate in the adjacent ub directory.
"""
import ast
import json
from pathlib import Path
import re
import sys
from textwrap import dedent, indent


def assemble(parent, template, events):
    responses = [e["part"]["text"] for e in map(json.loads, events.splitlines())
                 if e["type"] == "text"]
    block, = re.findall(r"```(?:python|py)\s*\n(.*?)```", responses[-1], re.S)
    dispatch = dedent(block).strip()
    branch, = ast.parse(dispatch).body
    if not isinstance(branch, ast.If):
        raise ValueError("Expected the generated conditional dispatch")
    kernel = template[template.index("@asctile.jit"):].rstrip()
    source = parent[parent.index("import torch"):]
    source = source.replace("def swi_glu(", kernel + "\n\n\ndef swi_glu(")
    anchor = "    ct = _col_tile(cols)"
    assert source.count(anchor) == 1
    source = source.replace(anchor, indent(dispatch, "    ") + "\n\n" + anchor)
    # The UB-height parent also handles fallthrough from the packed selector.
    # A missing capacity must not participate in integer arithmetic.
    query = "    ceiling = (_query_platform_ub_bytes() - _UB_FIXED_BYTES) // bytes_per_row"
    assert source.count(query) == 1
    source = source.replace(query, "    ub_bytes = _query_platform_ub_bytes()\n"
                            "    if ub_bytes is None:\n        return _RT\n"
                            "    ceiling = (ub_bytes - _UB_FIXED_BYTES) // bytes_per_row")
    compile(source, "<assembled>", "exec")
    return '"""SwiGLU with runtime vector grids and UB-sized BF16 pair tiles."""\n' + source


if __name__ == "__main__":
    parent, template, events, output = map(Path, sys.argv[1:])
    output.write_text(assemble(parent.read_text(), template.read_text(), events.read_text()))
