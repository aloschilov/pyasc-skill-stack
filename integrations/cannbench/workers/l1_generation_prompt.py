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
    # Keep the executable Torch oracle in the independent evaluation path.
    # The Exp skill and public proto specify every scalar and promotion rule.
    if op == "swi_glu":
        include(task / "golden.py")
    include(api / "SKILL.md", "## Surface to use", "`asctile.concat` is variadic")
    if op == "swi_glu":
        include(api / "SKILL.md", "### Choose a memory layout", "### Batch short rows")
        include(api / "references/reindex-slabs.md", "# Row geometry", "## Reducing the index reads")
        include(api / "references/reindex-slabs.md", "## Two-wave BF16", "## Dense FP32 slabs")
        include(api / "references/strided-row-tiles.md", "# Height and unroll", "Exercise the selector")
        include(api / "references/strided-row-tiles.md", "### Padded FP16 rows", "At R=3003")
        include(api / "references/streamed-halves.md", "# Flat SwiGLU", "Compare consecutive chunks")
        include(api / "templates/streamed-swiglu-flat.py")
        include(api / "templates/two-wave-swiglu-slab.py")
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
        "directly to the AscTile JIT and allocate with input.new_empty. The evaluator supplies "
        "CPU-backed Torch tensors even when the AscTile kernel runs on a physical NPU. "
        "Do not check for an NPU tensor, reject CPU tensors, or call .npu()/.to('npu'); "
        "the public launcher handles transfers. Use public tile math. "
        "Do not call a prebuilt operator, compute the output in Torch, hardcode cases, import "
        "another candidate, or leave placeholders. Return the entire module as one Python "
        "code block, ready for evaluation unchanged.\n"
    )
    if op == "swi_glu":
        sections.append(
            "Integrate both supplied kernel templates into the returned module. "
            "In the public swi_glu function, after deriving R, H, actual dtype, AIVs "
            "and available UB, call two_wave_tile before the ordinary slab selector. "
            "A positive height launches two_wave_swiglu on that AIV count with one cached "
            "signed-int16 gate-index tensor and returns the output. Zero falls through "
            "to the generic slab, rows or flat kernels. Trace this exact branch before "
            "returning the module: at 72 AIVs and 253952 UB bytes, R=133221 and H=7 "
            "give height 928 and two tiles per block; H=1 reaches the ordinary slab. "
            "Use the general selector for arbitrary inputs. Include the branch itself "
            "in the executable entry point. For wide aligned FP32 half-rows with a nearly "
            "full remainder, apply the UB-bounded exact-divisor search from the flat skill "
            "before its ordinary width/unroll fallback. Use the selected values in the launch."
        )
    return "\n".join(sections)


if __name__ == "__main__":
    sys.stdout.write(prompt(Path(sys.argv[1]), Path(sys.argv[2])))
