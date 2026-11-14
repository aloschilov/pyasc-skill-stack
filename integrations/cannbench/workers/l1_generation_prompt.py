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
        include(api / "references/streamed-halves.md", "For this four-wave FP32 stream", "For moderate single-row BF16")
        include(api / "references/reduction-tiling.md", "On the public CANN 9.2 950PR build", "## Work distribution")
        include(api / "templates/streamed-swiglu-flat.py")
        include(api / "templates/two-wave-swiglu-slab.py")
        include(api / "templates/paired-fp16-rows.py")
        include(api / "templates/moderate-swiglu-stream.py")
    elif op == "exp":
        include(api / "SKILL.md", "## Elementwise pattern", "### Choose a memory layout")
        include(skills / "asctile-cannbench-kernel/references/operator-patterns.md",
                "## Common elementwise path", "### High-level GeLU")
        include(api / "SKILL.md", "## Tails and padding", "## Host/JIT boundary")
    elif op in {"sigmoid", "mish", "gelu", "masked_scale", "foreach_addcdiv_scalar", "foreach_norm"}:
        patterns = skills / "asctile-cannbench-kernel/references/operator-patterns.md"
        include(task / "golden.py")
        include(patterns, "## Common elementwise path", "### Exp host dispatch")
        include(patterns, "### Shared tile constraints", "## StridedSlice")
        include(api / "SKILL.md", "## Tails and padding", "## Host/JIT boundary")
        specializations = {
            "masked_scale": ("## MaskedScale", "## SwiGLU"),
            "foreach_addcdiv_scalar": ("## ForeachAddcdivScalar", "## ForeachNorm"),
            "foreach_norm": ("## ForeachNorm", "## RMSNorm"),
        }
        if op in specializations:
            include(patterns, *specializations[op])
        if op in {"sigmoid", "mish", "gelu"}:
            sections.append(
                "Trace the actual public launch for both a short input and a multi-wave "
                "input. When two DMA stages fit and there are at least two tiles per AIV, "
                "pass U=2 to the actual loop unroll_factor. Use U=1 on the short path. "
                "Allocate query_ub_bytes() per AIV and account for both stages once. "
                "Bound the final partial copy and keep fixed extents for complete tiles."
            )
        if op == "foreach_norm":
            include(api / "SKILL.md", "For a sum reduction", "## Tails and padding")
            include(skills.parent / "reference/asctile/target/test_reduce_sum.py",
                    "@asctile.jit(reuse_alloc=2)", "@asctile.jit(reuse_alloc=2)\ndef reduce_sum_cols")
        sections.append(
            "Use the public oracle to identify the operation, input promotion, scalar attributes "
            "and output structure. Implement all tensor arithmetic in the device kernels. "
            "For tensor lists preserve the member order, shapes and individual dtypes. "
            "Choose a tile using the live input, output and temporary bytes per lane, and "
            "use the detected AIV count. Include all phases of a reduction in the public call."
        )
    else:
        raise ValueError(f"no prompt defined for {op!r}")
    sections.append(
        f"Write the complete runnable {op}.py from scratch for the public archive-bundled "
        "AscTile on Ascend950PR_9599. Implement the public signature, every supported dtype, "
        "shape, attribute and tail. Use the skills to choose the kernel, tile, unroll and "
        "grid automatically from layout, dtype and detected resources. Your sibling "
        "runtime is packaged beside the generated module. Import it with the literal package-relative "
        "statement 'from ._pyasc_runtime import ensure_npu_platform, query_ub_bytes, query_aivs, vector_blocks'. "
        f"Evaluation imports candidate.{op} from the package parent. The runtime supplies ensure_npu_platform(), query_ub_bytes(), query_aivs(), "
        "vector_blocks(work_items). Set the platform before querying. Pass caller tensors "
        "directly to the AscTile JIT and allocate with input.new_empty. The evaluator supplies "
        "NPU-resident Torch tensors in the restored-input evaluator or CPU tensors in the staged worker. "
        "Accept both placements without moving the input or output on the host; "
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

            "Check -rank <= dim < rank and normalize a negative dim once. Use shape[d]//2 for the output shape at that axis, "
            "and H=(shape[d]//2)*prod(shape[d+1:]) for every tile selector, global "
            "descriptor and JIT width. R=prod(shape[:d]). Verify 2*R*H equals the input "
            "numel. Trace (8192,16384),dim=0: output (4096,16384), R=1, H=67108864. "
            "Trace (11,13,16,67),dim=2: output (11,13,8,67), R=143, H=536. "
            "In the public swi_glu function, after deriving R, H, actual dtype, AIVs "
            "and available UB, call two_wave_tile before the ordinary slab selector. "
            "Place the ordinary BF16 slab immediately after the two-wave branch and before "
            "every generic row selector. Call ordinary_bf16_slab_tile and divide 16384 "
            "by the physical input bytes per row once. For a positive tile, call ordinary_slab_launch to get blocks and unroll from the number of row slabs ceil(R/T). Pass blocks as the grid and unroll as a ConstExpr argument to the JIT. Reference that argument in the slab loop unroll_factor and return from the host. Trace the actual "
            "public BF16 (1000003,2) call: reindex with T4096/U2 and 62 blocks on a 72-AIV device. For dense BF16 pairs use the four-wave even grid from ordinary_slab_launch. In the ordinary gather JIT iterate asctile.range(block_idx, count, block_num, unroll_factor=unroll) with the actual asctile.block_idx() and asctile.block_num() values. Block zero visits tile ordinals 0,62,124,186; preserve the separate two-wave multi-lane kernel’s consecutive ownership. The ordinary gather takes only input, output and one signed-int16 gate index as tensors. Call ordinary_gate_index and pass its returned tensor directly to the slab JIT. For H=1 it first tries dense_pair_gate_index. Other admitted ordinary halves use the int32 gate table in the same JIT. For every gather kernel shared by several half widths, derive value indices as gate_index + H in UB. Trace BF16 [2048,4] to the ordinary reindex JIT with one int32 gate table and T16/N72. Each row [1,2,4,8] has values [4,8]. "
            "Call whole_bf16_grid for aligned BF16 whole rows before the generic 2D route. A positive "
            "grid launches swiglu_flat with that many blocks, tile H and unroll 4. "
            "Before the ordinary row path, call whole_fp32_grid for moderate aligned FP32 "
            "half-rows. A positive grid launches whole_fp32_rows at that grid and passes "
            "the grid to its block-stride loop. Trace 512 rows of half-width 2048: on "
            "72 AIVs the JIT takes (512,2048,64) at grid 64 with unroll 4. "
            "Before the ordinary 8 KiB row-height rule, call three_stage_row_height on "
            "R, H, dtype, AIVs and usable UB. A positive height launches the strided row kernel "
            "with that height, width H, unroll 3 and the natural grid, then returns the output. "
            "When R is divisible by that height and W equals H, launch a dedicated "
            "opt_level=3 full-tile 2D JIT with default L2 descriptors and fixed-size copies. "
            "Align H in elements with lanes_per_dma = 32 // itemsize. Check actual launches for FP32 H2048/T2: both R2048 and R2046 take the fixed-copy JIT, and R2047 takes the bounded JIT. "
            "Trace R=2048, H=2048, FP32, 72 AIVs and 253952 UB bytes through the public entry "
            "point: it must pass height 2 and unroll 3 to the actual JIT launch. "
            "Use the padded FP16 launch from the row reference before the ordinary row fallback. Bind its recorded two tensors and all five scalar arguments to the bounded row JIT signature. "
            "A positive result launches the two-dimensional row JIT at that height, the "
            "padded width, natural grid and unroll 2. The measured 20 bytes per lane "
            "already cover both stages. With R=3003 and H=506 in FP16 on 72 AIVs "
            "and 253952 UB bytes, trace the actual JIT arguments: height 11, width "
            "512, unroll 2, 72 blocks. Zero continues to the ordinary selector. "
            "Call ordinary_dma_height with the logical H. Use padded W only for the UB fit "
            "and local allocation. Select U for the ordinary bounded body with bounded_row_unroll "
            "after its one-stage fit. Check R2049/H512 FP16 at N72/T8/U2 and "
            "R52224/H50 FP32 at N72/T64/W56/U2 through the actual kernel call. "
            "Trace FP32 R=273/H=509: four rows yield only 8144 useful "
            "bytes, so use the ordinary T=8, W=512 and N=35 on 72 AIVs. "
            "Use the strided 2D tile-index loop from the skill for ordinary multi-wave row loops. "
            "Check actual write intervals: R52224/T64/N72 gives 816 disjoint writes, "
            "the merged FP16 path at R1009/T7/N72 gives 144. Each output row has one owner. "
            "For the single-wave FP32 row JIT only, disable L2 on its input descriptor "
            "and keep default output caching. Issue both batched 2D "
            "reads and the store at the same logical tile start. For W == H rotate the tile ordinal by half the launched grid as in the row skill; for padded FP32 keep the natural grid and block_idx*T. Use the true row count for every DMA. "
            "Select this FP32 body for aligned and padded widths after the UB check. "
            "For W > H issue the value copy first, then the gate copy, then all arithmetic. Use gate for exp. "
            "For W == H issue gate then value before the arithmetic. Keep the two loads and one store bounded. "
            "For this single-wave FP32 launch on a resident Torch NPU tensor, use the "
            "current Torch stream of that input device in the public [blocks, stream] "
            "subscript as shown in the row reference. Keep the ordinary [blocks] call "
            "for CPU inputs. The two paths must use the same JIT and geometry. "
            "Assert that R273/H509 on 72 AIVs launches the single-wave JIT at N35/T8/W512. Audit both inputs and the store on the final one-row block. Keep batched half reads on rank-two [R,2*H]. "
            "For aligned FP16 apply the two-wave-plus-tail "
            "test before ordinary rows. Trace R1009/H1008 on 72 AIVs to T7/U2/N72, "
            "Call paired_fp16_height and launch paired_rows on a positive result before computing ordinary_dma_height. The ordinary helper also returns a positive height on this shape, so placing the paired launch after the ordinary return makes it unreachable. Verify the actual selected JIT name as well as its geometry. At R577/H1032 the selector returns zero: 2064 bytes per FP16 half-row is not 32-byte aligned. On the aligned R1009/H1008 shape the last block reads both halves and stores eight rows starting at 1001. "
            "At every call to a uniform-height full-copy 2D body prove R % T == 0 and W == H. "
            "Trace the bounded FP32 fallback on R2047/T2: its final DMA spans one row. "
            "For the ordinary and single-wave row paths, call fit_row_height with "
            "the physical width and the one-stage live bytes per lane. When even "
            "one row exceeds usable UB, return zero and use column chunks. A positive "
            "height alone cannot certify a fit. Check R=2039/H=32760 FP32 and "
            "R=1/H=1046017 BF16 through the actual launcher, as well as short rows. "
            "For a single long FP16 stream, call long_fp16_grid before the ordinary chunk "
            "fallback. With a positive grid, launch the same flat kernel with 8192 elements "
            "and unroll 4. Pass cache_input=False when the physical FP16 input exceeds 64 MiB. Return the output. The test input R=1, H=67108864 has 8192 "
            "full chunks and selects 64 of 72 AIVs; derive this from the general selector. "
            "For the narrow two-wave selector, a positive height launches two_wave_swiglu on that AIV count with one cached "
            "signed-int16 gate-index tensor. Obtain it by calling two_wave_gate_index(half, tile, device, cache) from the two-wave template. On NPU, launch this indexed gather with the pointer-width handle of the current Torch stream of the input device, using the same grid and JIT arguments as the CPU grid-only launch. Scope cached indices to the producing stream when needed. The ordinary index helper produces int32 for H>1 and belongs to the ordinary branch. Trace the actual two-wave argument for H7: 6496 int16 values occupying 12992 bytes. Return the output. Zero falls through "
            "to the generic slab, rows or flat kernels. Trace this exact branch before "
            "returning the module: at 72 AIVs and 253952 UB bytes, R=133221 and H=7 "
            "give height 928 and two tiles per block; H=1 reaches the ordinary slab. "
            "Use the general selector for arbitrary inputs. Include the branch itself "
            "in the executable entry point. For wide aligned FP32 half-rows with a nearly "
            "full remainder, apply the UB-bounded exact-divisor search from the flat skill "
            "before its ordinary width/unroll fallback. Call wide_fp32_chunk and count all R*(H//T) tiles for the three-wave test. Trace R2039/H32760 to T6552/U3 on 72 AIVs. For a positive wide_fp32_chunk and input above 64 MiB, pass False for cache_input in the actual swiglu_flat launch; the long FP16 rule above also bypasses input L2. Use True on other flat paths. Preserve the output descriptor default. For the flat route call launch_moderate_flat "
            "before launching the generic chunk kernel. In its column_stripe BF16 branch, cast the widened product to gate.dtype before copy_out, including partial chunks. Return the allocated output when this helper "
            "launches. For a single moderate BF16 stream, call four_wave_bf16_chunk "
            "before the generic flat fallback. On success launch swiglu_flat with its "
            "positive width, U4 and the detected AIV count. Trace BF16 H1046017 to T3632/U4. "
            "In the generic flat fallback call generic_flat_geometry and launch swiglu_flat "
            "with those exact three results. At cap=7936 the floor power of two is 4096. "
            "Use the measured complete two-stage "
            "allocation once, with enough full chunks for both stages. Trace single-row "
            "BF16 H2080641 through launch_medium_bf16_stream(input, out, R, H, aivs, usable_ub) before generic_flat_geometry. Return out if it launches. With 72 AIVs and 253952 B the actual four_chunk_stream launch uses T7232/U2, four consecutive bounded chunks per block. Preserve its floor-tile and generic fallbacks for constrained UB. For single-row FP16 call eight_wave_fp16_chunk before generic_flat_geometry. A positive width launches swiglu_flat at that width, U2 and the queried AIV count. Trace H2095104 to T3648/U2/N72. "
            "For the FP32 single-row edge launch and the fixed-copy three-stage FP32 row "
            "launch, pass the input device's current Torch stream when the input is resident "
            "on NPU. Use (blocks, ctypes.c_void_p(torch.npu.current_stream(x.device).npu_stream)) "
            "with the actual selected block count and input variable. A CPU-staged input takes "
            "the integer grid. Obtain the stream on each call and propagate lookup failures. "
            "Use the selected values in the actual public entry point. Preserve host helper "
            "names from definitions through every caller: wide_fp32_chunk is distinct from "
            "whole_fp32_grid. Verify every global function called by the public dispatcher "
            "exists in the returned module or its imports. Walk the public FP32 short "
            "and wide branches as well as the BF16 branches."
        )
    elif op == "exp":
        sections.append(
            "Use x.element_size() for the host item size. Trace each branch including any default: "
            "pass the caller's x unchanged into the JIT. For an unsupported dtype raise TypeError. "
            "Derive log_base with the positive-base predicate from the Exp skill. For a NaN base and finite x, the golden uses ordinary exp. "
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
            "For resident FP16 and BF16 on either short contiguous affine path, and for resident "
            "FP16 identity on its short grid-stride path, launch the chosen "
            "JIT with (blocks, ctypes.c_void_p(torch.npu.current_stream(x.device).npu_stream)). "
            "Read the stream inside each call and propagate lookup errors before any NPU launch. "
            "CPU-staged tensors use the integer grid. "
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
