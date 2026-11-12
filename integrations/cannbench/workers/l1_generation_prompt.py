"""Build a clean L1 generation prompt from a task contract and the checked-out skills.

Usage: python l1_generation_prompt.py TASK_DIR SKILLS_DIR
Task directories must be named after the public Python entry point.
"""
import sys
from pathlib import Path


def prompt(task, skills):
    op = task.name
    api = skills / "asctile-api-patterns"
    sections = []

    def include(path, start="", end=""):
        body = path.read_text()
        if start:
            body = body[body.index(start):]
        if end:
            body = body[:body.index(end)]
        sections.append(f"= {path.name}\n{body.strip()}\n")

    include(task / "proto.yaml")
    include(task / "golden.py")
    include(api / "SKILL.md", "## Surface to use", "`asctile.concat` is variadic")
    if op == "swi_glu":
        include(api / "SKILL.md", "### Choose a memory layout", "### Batch short rows")
        include(api / "references/reindex-slabs.md", "# Row geometry", "## Reducing the index reads")
        include(api / "references/strided-row-tiles.md", "# Height and unroll", "Exercise the selector")
        include(api / "references/streamed-halves.md", "# Flat SwiGLU", "Compare consecutive chunks")
        include(api / "templates/streamed-swiglu-flat.py")
    elif op == "exp":
        include(api / "SKILL.md", "## Elementwise pattern", "### Choose a memory layout")
        include(skills / "asctile-cannbench-kernel/references/operator-patterns.md",
                "## Common elementwise path", "### High-level GeLU")
        include(api / "SKILL.md", "## Tails and padding", "## Host/JIT boundary")
    else:
        raise ValueError(f"no prompt defined for {op!r}")
    sections.append(
        f"Write the complete runnable {op}.py from scratch for the public archive-bundled "
        "AscTile on Ascend950PR_9599. Implement the public signature, every supported dtype, "
        "shape, attribute and tail. Use the skills to choose the kernel, tile, unroll and "
        "grid automatically from layout, dtype and detected resources. Your sibling "
        "._pyasc_runtime provides ensure_npu_platform(), query_ub_bytes(), query_aivs(), "
        "vector_blocks(work_items). Set the platform before querying. Pass caller tensors "
        "directly to the AscTile JIT and allocate with input.new_empty. Use public tile math. "
        "Do not call a prebuilt operator, compute the output in Torch, hardcode cases, import "
        "another candidate, or leave placeholders. Return the entire module as one Python "
        "code block, ready for evaluation unchanged.\n"
    )
    return "\n".join(sections)


if __name__ == "__main__":
    sys.stdout.write(prompt(Path(sys.argv[1]), Path(sys.argv[2])))
