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
        include(api / "templates/moderate-swiglu-stream.py")
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
        "runtime is packaged beside the generated module. Import it with the literal package-relative "
        "statement 'from ._pyasc_runtime import ensure_npu_platform, query_ub_bytes, query_aivs, vector_blocks'. "
        "Evaluation imports candidate.exp or candidate.swi_glu from the package parent. It supplies ensure_npu_platform(), query_ub_bytes(), query_aivs(), "
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
            "Integrate the supplied kernel templates into the returned module. "
            "Inline the widened gate division and value multiplication directly in each JIT "
            "body, including full tiles and remainders. A call from a JIT to an ordinary "
            "Python swiglu element helper fails dependency resolution on this wheel. "

            "Normalize dim to d. Use shape[d]//2 for the output shape at that axis, "
            "and H=(shape[d]//2)*prod(shape[d+1:]) for every tile selector, global "
            "descriptor and JIT width. R=prod(shape[:d]). Verify 2*R*H equals the input "
            "numel. Trace (8192,16384),dim=0: output (4096,16384), R=1, H=67108864. "
            "Trace (11,13,16,67),dim=2: output (11,13,8,67), R=143, H=536. "
            "In the public swi_glu function, after deriving R, H, actual dtype, AIVs "
            "and available UB, call two_wave_tile before the ordinary slab selector. "
            "Place the ordinary BF16 slab immediately after the two-wave branch and before "
            "every generic row selector. When it fits, launch and return. Trace the actual "
            "public BF16 (1000003,2) call: reindex with T4096/U2 and 72 blocks. "
            "Call whole_bf16_grid for aligned BF16 whole rows before the generic 2D route. A positive "
            "grid launches swiglu_flat with that many blocks, tile H and unroll 4. "
            "Before the ordinary row path, call whole_fp32_grid for moderate aligned FP32 "
            "half-rows. A positive grid launches whole_fp32_rows at that grid and passes "
            "the grid to its block-stride loop. Trace 512 rows of half-width 2048: on "
            "72 AIVs the JIT takes (512,2048,64) at grid 64 with unroll 4. "
            "Before the ordinary 8 KiB row-height rule, call three_stage_row_height on "
            "R, H, dtype, AIVs and usable UB. A positive height launches the strided row kernel "
            "with that height, width H, unroll 3 and the natural grid, then returns the output. "
            "Trace R=2048, H=2048, FP32, 72 AIVs and 253952 UB bytes through the public entry "
            "point: it must pass height 2 and unroll 3 to the actual JIT launch. "
            "Before the ordinary row fallback, call padded_fp16_two_stage_height. "
            "A positive result launches the two-dimensional row JIT at that height, the "
            "padded width, natural grid and unroll 2. The measured 20 bytes per lane "
            "already cover both stages. With R=3003 and H=506 in FP16 on 72 AIVs "
            "and 253952 UB bytes, trace the actual JIT arguments: height 11, width "
            "512, unroll 2, 72 blocks. Zero continues to the ordinary selector. "
            "Call ordinary_dma_height with the logical H. Use padded W only for the UB fit "
            "and local allocation. Trace FP32 R=273/H=509: four rows yield only 8144 useful "
            "bytes, so T=8, W=512 and N=35 on 72 AIVs. "
            "For the ordinary and single-wave row paths, call fit_row_height with "
            "the physical width and the one-stage live bytes per lane. When even "
            "one row exceeds usable UB, return zero and use column chunks. A positive "
            "height alone cannot certify a fit. Check R=2039/H=32760 FP32 and "
            "R=1/H=1046017 BF16 through the actual launcher, as well as short rows. "
            "For a single long FP16 stream, call long_fp16_grid before the ordinary chunk "
            "fallback. With a positive grid, launch the same flat kernel with 8192 elements "
            "and unroll 4 and return the output. The test input R=1, H=67108864 has 8192 "
            "full chunks and selects 64 of 72 AIVs; derive this from the general selector. "
            "For the narrow two-wave selector, a positive height launches two_wave_swiglu on that AIV count with one cached "
            "signed-int16 gate-index tensor and returns the output. Zero falls through "
            "to the generic slab, rows or flat kernels. Trace this exact branch before "
            "returning the module: at 72 AIVs and 253952 UB bytes, R=133221 and H=7 "
            "give height 928 and two tiles per block; H=1 reaches the ordinary slab. "
            "Use the general selector for arbitrary inputs. Include the branch itself "
            "in the executable entry point. For wide aligned FP32 half-rows with a nearly "
            "full remainder, apply the UB-bounded exact-divisor search from the flat skill "
            "before its ordinary width/unroll fallback. For the flat route call launch_moderate_flat "
            "before launching the generic chunk kernel. Return the allocated output when this helper "
            "launches. Use the selected values in the actual public entry point."
        )
    elif op == "exp":
        sections.append(
            "In the public Exp entry point, select the qualified short 16-bit regime before "
            "generic streaming. Keep all host tiles, grids and unrolls as native Python ints. "
            "Use ConstExpr only as a kernel parameter annotation, passing native ints or bools at launch. "
            "For 24–32 KiB of input per available AIV and a UB-qualified "
            "aligned three-tile schedule, call short_exp_geometry in the public entry point and "
            "pass unroll 3 in both paths. Use its same affine boolean to select the DMA "
            "quantum and ownership. Reduce the grid only for BF16 log-only. Use the contiguous "
            "schedule if any of apply_scale, apply_shift or apply_log is active, including "
            "the base-only case. For FP16 identity use grid-stride with the same unroll. "
            "Trace the JIT selected for short scale/shift streams: use opt_level=3 on "
            "a separate contiguous kernel with the same arithmetic and runtime loop. "
            "Keep the default contiguous JIT on BF16 log-only. "
            "Trace both public launches at 72 AIVs: BF16 1023x1023, base 2, unit scale and "
            "zero shift must select contiguous T5504 U3 on 64 blocks. Use the 256-byte DMA quantum for short affine streams and the qualified near-capacity grid rule for BF16 log-only. For FP16 shift-only 2049x513 trace N72/T4992/U3. FP16 1024x1024, base -1, unit "
            "scale and zero shift must select stride T4864 U3. Use the general input-byte, "
            "alignment and UB rules from the skill for arbitrary inputs. Compute contiguous "
            "tiles per block inside the JIT from runtime size, tile and block_num(), retaining "
            "compile-time unroll 3. For a one-wave tile budget at least T*(8+itemsize) bytes "
            "for the measured live values. Reject a tile exceeding available UB and enter "
            "the qualified streaming fallback. Keep one-wave, U3 short-stream and long-stream "
            "unroll independent in their actual JIT loops."
        )
    return "\n".join(sections)


if __name__ == "__main__":
    sys.stdout.write(prompt(Path(sys.argv[1]), Path(sys.argv[2])))
