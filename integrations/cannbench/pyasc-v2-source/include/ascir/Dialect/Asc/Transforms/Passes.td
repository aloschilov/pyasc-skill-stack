/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_PASSES_TD
#define ASC_PASSES_TD

include "mlir/Pass/PassBase.td"

def AllocateTensor : Pass<"ascendc-allocate-tensor", "func::FuncOp"> {
  let summary = "Assign memory addresses to LocalTensorAutoOp placeholders for static UB allocation";
  let constructor = "mlir::ascendc::createAllocateTensorPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Converts `LocalTensorAutoOp` placeholders to `LocalTensorV3Op` with concrete memory addresses and tile sizes.
    This pass performs static memory allocation for on-chip buffers by assigning consecutive aligned addresses
    to tensors based on their normalized memory position (A1/L1, A2/L0A, B2/L0B, CO1/L0C, VECCALC/UB).

    Position normalization: B1→A1, VECIN/VECOUT→VECCALC. Addresses are aligned to `ubBlockSize` (256 bytes).
    After allocation, `LocalTensorAutoOp` is replaced with `LocalTensorV3Op` containing position, address, and tile size.
  }];
}

def ComputeMemoryConsumption : Pass<"ascendc-compute-memory-consumption", "ModuleOp"> {
  let summary = "Calculate total memory consumption per buffer location and attach to module";
  let constructor = "mlir::ascendc::createComputeMemoryConsumptionPass()";
  let description = [{
    Analyzes all `LocalTensorV3Op` operations and computes total memory usage for each hardware buffer location:
    L1, L0A, L0B, L0C, and UB. The results are stored as a dictionary attribute `memoryConsumed` on the module,
    with keys like "L1", "L0A", etc. and values in bytes.

    This pass is useful for verifying that kernel memory usage stays within hardware limits.
  }];
}

def ComputeReuseGroup : Pass<"ascendc-compute-reuse-group", "func::FuncOp"> {
  let summary = "Compute reuse group indices for unrolled loop iterations to guide tensor allocation reuse";
  let constructor = "mlir::ascendc::createComputeReuseGroupPass()";
  let description = [{
    Computes `ascendc.reuse_group` indices for operations in unrolled loops to guide downstream tensor
    allocation reuse. The pass analyzes `scf.execute_region` operations with `asctile.unroll_factor` and
    `asctile.unroll_iter` attributes, and remaps iteration indices into reuse groups.

    The reuse group is computed as `(unroll_iter + startIndex) % unrollFactor`, where `startIndex` accounts
    for nesting level and epilogue iterations. An epilogue iteration is detected when `unroll_iter` equals
    `unroll_factor` (i.e., the remainder-handling tail of an unrolled loop). Epilogue iterations shift the
    start index for subsequent loops at the same nesting level, ensuring correct reuse group alignment.

    After processing, `asctile.unroll_iter` attributes are replaced with `ascendc.reuse_group` on all
    operations within unrolled loops.
  }];
}

def DeclarePyStruct : Pass<"ascendc-declare-py-struct", "ModuleOp"> {
  let summary = "Insert emitasc.declare_py_struct operations for all Python struct types used in the module";
  let constructor = "mlir::ascendc::createDeclarePyStructPass()";
  let dependentDialects = ["emitasc::EmitAscDialect"];
  let description = [{
    Collects all `emitasc.py_struct` types referenced in function arguments, block arguments, and operation results,
    then inserts `emitasc.declare_py_struct` operations at the module beginning to declare them for C++ code emission.

    The pass deduplicates struct types while preserving the order of first occurrence. Nested struct types
    (structs containing other structs) are also declared.
  }];
}

def DefineCubeOnly : Pass<"ascendc-define-cube-only", "ModuleOp"> {
  let summary = "Define ASCENDC_CUBE_ONLY macro for cube-only (matrix multiplication) kernels";
  let constructor = "mlir::ascendc::createDefineCubeOnlyPass()";
  let dependentDialects = ["emitc::EmitCDialect"];
  let description = [{
    Inserts `#define ASCENDC_CUBE_ONLY` verbatim at module start and sets `matmul_cube_only` attribute.
    This macro enables cube-only optimizations in the Ascend C runtime for kernels that only perform
    matrix multiplication without vector operations.
  }];
}

def DetectEnableDebug : Pass<"ascendc-detect-enable-debug", "ModuleOp"> {
  let summary = "Detect debug utility usage (printf, dump_tensor) and set enable_debug attribute";
  let constructor = "mlir::ascendc::createDetectEnableDebugPass()";
  let description = [{
    Scans the module for `PrintfOp` or `DumpTensorOp` operations. If any are found, sets the `enable_debug`
    unit attribute on the module to enable debug runtime support during kernel execution.
  }];
}

def DetectKernelType : Pass<"ascendc-detect-kernel-type", "ModuleOp"> {
  let summary = "Classify kernel as vector, cube, or mixed based on operation types present";
  let constructor = "mlir::ascendc::createDetectKernelTypePass()";
  let description = [{
    Analyzes operations in the module to determine kernel type and sets `kernel_type` string attribute:
    - "vector": Only vector operations present (no `MmadOp` or `RegistMatmulObjOp`)
    - "cube": Only matrix multiplication operations present (no `VectorOp`)
    - "mixed": Both vector and cube operations present

    This classification affects synchronization strategy and runtime configuration.
  }];
}

def DispatchAlloc : Pass<"ascendc-dispatch-alloc", "func::FuncOp"> {
  let summary = "Dispatch tensor allocation to static or TPipe-backed strategy based on target architecture";
  let constructor = "mlir::ascendc::createDispatchAllocPass()";
  let description = [{
    Selects between static allocation and TPipe-backed allocation for on-chip tensor buffers. The decision
    is based on the `asc.static_alloc` module attribute, target architecture, and presence of incompatible
    operations. When targeting C310 architectures without `BroadcastOp`, static allocation is preferred;
    otherwise TPipe-backed allocation is used.
  }];
}

def EraseSync : Pass<"ascendc-erase-sync", "func::FuncOp"> {
  let summary = "Remove TQueBind synchronization operations and replace deque_tensor with allocated tensors";
  let constructor = "mlir::ascendc::createEraseSyncPass()";
  let description = [{
    Removes intra-core synchronization infrastructure for static allocation mode:
    - Erases `TQueBindEnqueTensorOp`, `SetFlagOp`, `WaitFlagOp`, `PipeBarrierOp`
    - Replaces `TQueBindDequeTensorOp` with the tensor from corresponding `TQueBindAllocTensorOp`

    This pass is used when tensors are statically allocated and queue-based synchronization is not needed.
  }];
}

def FillAscOperands : Pass<"ascendc-fill-asc-operands", "func::FuncOp"> {
  let summary = "Fill default cal_count, mask, repeat_times, and repeat_params operands for L0/L2 Ascend C operations";
  let constructor = "mlir::ascendc::createFillAscOperandsPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
  let description = [{
    Populates default operands for vector operations based on tensor shape:
    - **L0 operations**: Fills `mask`, `repeat_times`, and `repeat_params` (stride values)
    - **L2 operations**: Fills `cal_count` with number of elements

    Default stride values: block_stride=1, repeat_stride=8. Mask is computed based on element type size (1, 2, 4, 8).
    Repeat times calculated as ceil(numElements / numPerRepeat).
  }];
}

def FixupMmadAccParams : Pass<"ascendc-fixup-mmad-acc-params", "func::FuncOp"> {
  let summary = "Fix accumulator initialization for sequential mmad operations using the same destination tensor";
  let constructor = "mlir::ascendc::createFixupMmadAccParamsPass()";
  let dependentDialects = ["emitasc::EmitAscDialect", "memref::MemRefDialect"];
  let description = [{
    Handles matrix multiplication accumulation where multiple `mmad` operations write to the same destination tensor.
    Creates a runtime boolean variable to track whether this is the first mmad to the destination:
    - First mmad: `cmatrixInitVal=true` (initialize accumulator)
    - Subsequent mmads: `cmatrixInitVal=false` (accumulate into existing values)

    Only processes mmad operations that have `cmatrixInitVal` parameter in their mmad_params.

    Example transformation:
    ```mlir
    // Before: mmad with fixed cmatrixInitVal in loop
    %acc = ascendc.local_tensor_auto co1() : <16x16xf32>
    scf.for %i = 0 to %n {
      %params = emitasc.init_struct !ascendc.mmad_params(
        "cmatrixInitVal" = %c1  // always true (reinit each iteration)
      )
      ascendc.mmad %acc, %a, %b, %params
    }

    // After: runtime variable controls initialization
    %acc = ascendc.local_tensor_auto co1() : <16x16xf32>
    %var = emitasc.variable true, memref<1xi1>  // init flag
    scf.for %i = 0 to %n {
      %init_val = memref.load %var[%c0]
      %params = emitasc.init_struct !ascendc.mmad_params(
        "cmatrixInitVal" = %init_val  // true first iter, false after
      )
      ascendc.mmad %acc, %a, %b, %params
      %false = arith.constant false
      memref.store %false, %var[%c0]  // set false for next iter
    }
    ```
  }];
}

def FuseBufIdSync : Pass<"ascendc-fuse-bufid-sync", "func::FuncOp"> {
  let summary = "Remove redundant get_buf/rls_buf synchronization between consecutive operations with matching bufId";
  let constructor = "mlir::ascendc::createFuseBufIdSyncPass()";
  let description = [{
    Optimizes BufId-based synchronization by removing redundant `get_buf`/`rls_buf` operations:
    - For consecutive operations with the same `bufId` attribute and same pipeline type:
      - Remove `get_buf` before middle operations (only first op needs it)
      - Remove `rls_buf` after middle operations (only last op needs it)
    - Pipeline types tracked: PIPE_V (vector), PIPE_MTE2 (GM→UB load), PIPE_MTE3 (UB→GM store)

    After optimization, removes the `bufId` attribute from all operations. Used on C310 architecture.
  }];
}

def GenerateBoilerplate : Pass<"ascendc-generate-boilerplate", "ModuleOp"> {
  let summary = "Insert required C++ include headers based on operations used in the kernel";
  let constructor = "mlir::ascendc::createGenerateBoilerplatePass()";
  let dependentDialects = ["emitc::EmitCDialect"];
  let description = [{
    Adds necessary Ascend C header includes at module start:
    - Always includes `kernel_operator.h` (core runtime)
    - If `ListTensorDescOp` or `ListTensorDescV2Op` present: includes `kernel_operator_list_tensor_intf.h`
    - If `RegistMatmulObjOp` present: includes `lib/matmul_intf.h`
    - If `TensorDescOp` present: includes `kernel_operator_list_tensor_intf.h`
  }];
}

def HoistQueBind : Pass<"ascendc-hoist-que-bind", "func::FuncOp"> {
  let summary = "Move queue and buffer initialization operations to the function entry block";
  let constructor = "mlir::ascendc::createHoistQueBindPass()";
  let description = [{
    Hoists TQueBind-related initialization ops from loops to function root:
    - `QueueOp`, `QueBindOp`, `TBufOp` declarations
    - `TPipeInitQueueOp`, `TPipeInitBufferOp` initialization
    - `TBufGetTensorOp` tensor retrieval

    This optimization ensures queue/buffer setup happens once at kernel entry rather than repeatedly in loops.
  }];
}

def HoistTensorAllocation : Pass<"ascendc-hoist-tensor-allocation", "func::FuncOp"> {
  let summary = "Move LocalTensorAutoOp allocations from loops to the function entry block";
  let constructor = "mlir::ascendc::createHoistTensorAllocationPass()";
  let options = [
    Option<"excludeInOut", "exclude-in-out", "bool", "false",
           "Keep input/output tensors inside loops, only hoist temporary tensors">,
  ];
  let description = [{
    Hoists `LocalTensorAutoOp` tensor allocations to the function root to reduce repeated allocations in loops.
    When `exclude-in-out=true`: only hoists tensors without `input` or `output` flags (pure temporary tensors).
    When `exclude-in-out=false`: hoists all tensors regardless of input/output status.

    Input/output tensors are those connected to GM via `DataCopyOp` with `GlobalToLocal`/`LocalToGlobal` direction.
  }];
}

def InputOutputTensor : Pass<"ascendc-input-output-tensor", "func::FuncOp"> {
  let summary = "Mark tensors as input/output based on GM data copy direction and handle input-only to output copies";
  let constructor = "mlir::ascendc::createInputOutputTensorPass()";
  let description = [{
    Analyzes `DataCopyOp` to set `input`/`output` flags on `LocalTensorAutoOp`:
    - `gm_ubuf` direction → tensor is input (loaded from GM)
    - `ubuf_gm` direction → tensor is output (stored to GM)

    Also handles special case: if an input-only tensor is used for `ubuf_gm` copy, creates a separate output tensor
    and inserts `DataCopyL2Op` to copy input tensor to output tensor before the GM store.

    Example transformation:
    ```mlir
    // Before: unmarked tensors
    %0 = ascendc.local_tensor_auto vecin() : <64xf32>
    ascendc.data_copy_l2 %0, %gm_in, %c64  // gm→ub (input)
    %1 = ascendc.local_tensor_auto vecout() : <64xf32>
    ascendc.data_copy_l2 %gm_out, %1, %c64  // ub→gm (output)

    // After: tensors marked with input/output flags
    %0 = ascendc.local_tensor_auto vecin() input : <64xf32>
    %1 = ascendc.local_tensor_auto vecout() output : <64xf32>
    ```
  }];
}

def InsertBufIdSync : Pass<"ascendc-insert-bufid-sync", "func::FuncOp"> {
  let summary = "Insert get_buf/rls_buf synchronization around operations for BufId-based tracking on C310 arch";
  let constructor = "mlir::ascendc::createInsertBufIdSyncPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Inserts BufId synchronization for Ascend hardware with C310 architecture using buffer ID tracking:
    1. Assigns unique `bufId` to each tensor allocation (`LocalTensorV3Op`, `TBufGetTensorOp`)
    2. For each operation using a tensor, inserts `get_buf` before and `rls_buf` after
    3. Tracks pipeline type per operation: PIPE_MTE1 (L1 load), PIPE_MTE2 (GM load), PIPE_MTE3 (GM store),
       PIPE_FIX (fixpipe), PIPE_S (scalar), PIPE_M (matmul), PIPE_V (vector)
    4. For `scf.for` loops (except `vec_scope_loop` or parallel), inserts MTE3-MTE2/MTE3-MTE1 sync at loop end

    Sets `bufId` array attribute on operations for downstream `FuseBufIdSync` optimization.
  }];
}

def InsertBiasBufIdSync : Pass<"ascendc-insert-bias-bufid-sync", "func::FuncOp"> {
  let summary = "Insert get_buf/rls_buf synchronization around MmadOp for bias tensors copied to BT";
  let constructor = "mlir::ascendc::createInsertBiasBufIdSyncPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Inserts buffer synchronization (`get_buf`/`rls_buf`) around `MmadOp` operations that use bias tensors
    in C2 position (bias tensor memory).

    This pass ensures proper synchronization between bias tensor access and matrix multiplication operations
    on C310 architecture. The pass:
    1. Finds `LocalTensorV3Op` in C2 position with a `bufId` attribute
    2. Extracts the `bufId` value from this tensor
    3. For each `MmadOp` with `cmatrixSource` parameter:
       - Inserts `get_buf pipe_m, <bufId>` before the `MmadOp`
       - Inserts `rls_buf pipe_m, <bufId>` after the `MmadOp`

    **Important**: This pass assumes there is only one bias tensor with `bufId` per function. The `UnifyBiasTensor`
    pass consolidates all C2 tensors into a single tensor, ensuring only one `bufId` exists.
  }];
}

def InsertCrossCoreSync : Pass<"ascendc-insert-cross-core-sync", "func::FuncOp"> {
  let summary = "Insert CrossCoreSetFlag and CrossCoreWaitFlag synchronization between if_aic and if_aiv operations";
  let constructor = "mlir::ascendc::createInsertCrossCoreSyncPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    This pass inserts `ascendc.cross_core_set_flag` and `ascendc.cross_core_wait_flag` operations
    between consecutive `ascendc.if_aic` and `ascendc.if_aiv` operations when they are
    of different types.

    Synchronization rules:
    - No sync is inserted for the first group operation in sequence.
    - If two consecutive groups are of the same type (both if_aic or both if_aiv), no sync is inserted.
    - When an `if_aic` is followed by an `if_aiv`:
      - `CrossCoreSetFlag<4, pipe>(0)` and `CrossCoreSetFlag<4, pipe>(16)` are inserted at the end
        of the if_aic block, where `pipe` is determined by the last operation in the block.
      - `CrossCoreWaitFlag<4, PIPE_S>(0)` is inserted at the beginning of the if_aiv block.
    - When an `if_aiv` is followed by an `if_aic`:
      - `CrossCoreSetFlag<4, pipe>(0)` is inserted at the end of the if_aiv block.
      - `CrossCoreWaitFlag<4, PIPE_S>(0)` is inserted at the beginning of the if_aic block.

    The pipe for `CrossCoreSetFlag` is determined by the last non-yield operation in the block:
    - `mmad`, `mmad_with_bias` → `PIPE_M`
    - `fixpipe` → `PIPE_FIX`
    - `copy_to_l0` operations → `PIPE_MTE1`
    - `fill` → `PIPE_MTE2`
    - `data_copy` (GlobalToLocal) → `PIPE_MTE2`
    - `data_copy` (LocalToGlobal) → `PIPE_MTE3`
    - `local_tensor.get_value`, `local_tensor.set_value` → `PIPE_S`
    - All other operations → `PIPE_V`
  }];
}

def InsertQueSync : Pass<"ascendc-insert-que-sync", "func::FuncOp"> {
  let summary = "Insert TQueBind enqueue/dequeue synchronization and scalar get/set_value barriers";
  let constructor = "mlir::ascendc::createInsertQueSyncPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Inserts intra-core synchronization for TQueBind-managed tensors and scalar operations:
    1. **Enqueue tensors**: After each `OpWithDst`, inserts `TQueBindEnqueTensorOp` or `PipeBarrierOp(pipe_v)`
    2. **Dequeue tensors**: For enqueued tensors, inserts `TQueBindDequeTensorOp` before first user
    3. **Get/Set value sync**: Wraps `get_value`/`set_value` with V_S/S_V event sync (fetch_event_id, set_flag, wait_flag)
    4. **Loop handling**: Special sync placement for `set_value` inside single-op loops
    5. **Canonicalization**: Adds `PipeBarrierOp(pipe_all)` at function end and runs canonicalization

    TQueBind synchronization enables double-buffering and overlapping compute with data transfer.

    Example: enqueue/dequeue synchronization:
    ```mlir
    // Before: tensor used directly after allocation
    %tensor = ascendc.que_bind.alloc_tensor %queue
    ascendc.add_l2 %dst, %tensor, %tensor, %count

    // After: tensor dequeued before use, enqueued after compute
    %tensor = ascendc.que_bind.alloc_tensor %queue
    %dequeued = ascendc.que_bind.deque_tensor %queue
    ascendc.add_l2 %dst, %dequeued, %dequeued, %count
    ascendc.que_bind.enque_tensor %queue, %dst
    ```

    Example: scalar get/set_value with V_S/S_V sync:
    ```mlir
    // Before: direct scalar access
    %val = ascendc.local_tensor.get_value %tensor, %offset

    // After: wrapped with event sync
    %event = ascendc.pipe.fetch_event_id %pipe, v_s
    ascendc.set_flag v_s, %event
    ascendc.wait_flag v_s, %event
    %val = ascendc.local_tensor.get_value %tensor, %offset
    %event2 = ascendc.pipe.fetch_event_id %pipe, s_v
    ascendc.set_flag s_v, %event2
    ascendc.wait_flag s_v, %event2
    ```
  }];
}

def InsertSubBlockGuard : Pass<"ascendc-insert-subblock-guard", "func::FuncOp"> {
  let summary = "Insert early return guard for non-zero sub-block indices in mixed kernel functions";
  let constructor = "mlir::ascendc::createInsertSubBlockGuardPass()";
  let dependentDialects = ["emitc::EmitCDialect"];
  let description = [{
    Inserts a guard at the beginning of kernel functions that returns early if the current sub-block
    index is not zero. This is useful for multi-core kernels where only sub-block 0 should execute
    the main kernel logic.

    The pass only operates on functions within modules that have `asc.kernel_type = "mixed"` attribute.
    For other kernel types (vector, cube) or modules without the attribute, the pass is a no-op.

    The pass inserts the following C++ code at the function entry:
    ```cpp
    if (AscendC::GetSubBlockIdx() != 0) return;
    ```

    This ensures that only sub-block 0 executes the kernel body, while other sub-blocks return immediately.
    The pass is idempotent and skips functions that already have the guard inserted.
  }];
}

def MaterializeTensor : Pass<"ascendc-materialize-tensor", "func::FuncOp"> {
  let summary = "Convert LocalTensorAutoOp placeholders to concrete TBuf or TQueBind allocations";
  let constructor = "mlir::ascendc::createMaterializeTensorPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
  let options = [
    Option<"alwaysBuf", "always-buf", "bool", "false",
           "Use TBuf for all tensors; otherwise use TQueBind for input/output tensors">,
  ];
  let description = [{
    Materializes tensor allocation placeholders into concrete Ascend C buffer objects:
    - **TQueBind path** (for input/output tensors when `alwaysBuf=false`):
      Creates `QueueOp` + `TPipeInitQueueOp` + `TQueBindAllocTensorOp` + `TQueBindFreeTensorOp` at function end
    - **TBuf path** (for temporary tensors or when `alwaysBuf=true`):
      Creates `TBufOp` + `TPipeInitBufferOp` + `TBufGetTensorOp`

    Input tensors use `VECIN` queue position, output tensors use `VECOUT`. Temporary tensors use `VECCALC` TBuf.
    Buffer size computed from tensor shape (static or dynamic via shape operands).

    Example transformation (TQueBind path for input tensor):
    ```mlir
    // Before: placeholder
    %0 = ascendc.local_tensor_auto vecin() input : <64xf32>

    // After: queue-based allocation
    %pipe = ascendc.pipe
    %queue = ascendc.queue : <vecin, 1>
    ascendc.pipe.init_queue %pipe, %queue, %c1, %c256
    %0 = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<64xf32>
    // ... at function end:
    ascendc.que_bind.free_tensor %queue, %0
    ```

    Example transformation (TBuf path for temporary tensor):
    ```mlir
    // Before: placeholder
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>

    // After: buffer-based allocation
    %pipe = ascendc.pipe
    %tbuf = ascendc.tbuf : <veccalc>
    ascendc.pipe.init_buffer %pipe, %tbuf, %c256
    %0 = ascendc.tbuf.get_tensor %tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<64xf32>
    ```
  }];
}

def LegalizeKernelArgs : Pass<"ascendc-legalize-kernel-args", "ModuleOp"> {
  let summary = "Attach kernel argument attributes and insert ffts_addr handling for multi-core kernels";
  let constructor = "mlir::ascendc::createLegalizeKernelArgsPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect", "scf::SCFDialect"];
  let options = [
    Option<"setFftsAddr", "set-ffts-addr", "bool", "false",
           "Append ffts_addr kernel argument and call set_ffts_base_addr for cross-core sync">,
  ];
  let description = [{
    Processes kernel entry functions (marked with `ascendc.global` attribute):
    1. Marks all existing arguments as `Explicit` kernel arguments
    2. If `setFftsAddr=true`: adds `ffts_addr` memref argument and calls `SetFftsBaseAddrOp`
    3. If matmul present and not cube-only: inserts `AscendIsAICOp` check with `FftsCrossCoreSyncOp` for AIC mode

    Kernel argument attributes (`emitasc.kernel_arg`) control how arguments are handled by the runtime.
  }];
}

def LowerToL0 : Pass<"ascendc-lower-to-l0", "func::FuncOp"> {
  let summary = "Lower L2 vector operations to L0 intrinsics with default repeat parameters";
  let constructor = "mlir::ascendc::createLowerToL0Pass()";
  let dependentDialects = ["ascendc::AscendCDialect", "emitasc::EmitAscDialect"];
  let description = [{
    Converts L2-level vector operations to L0-level hardware intrinsics:
    - Unary L2→L0: `abs_l2`→`abs_l0`, `exp_l2`→`exp_l0`, etc. with mask, repeat_times=0, empty repeat_params
    - Binary L2→L0: `add_l2`→`add_l0`, `mul_l2`→`mul_l0`, etc. with similar parameters
    - Special ops: `duplicate_l2`→`duplicate_l0`, `cast_l2`→`cast_l0`, `compare_l2`→`compare_l0`
    - Scalar ops: `adds_l2`→`adds_l0`, `muls_l2`→`muls_l0`

    L0 operations provide direct hardware control but require explicit parameters; this pass provides defaults.
    L2 operations are simpler API with automatic parameter calculation (done by `FillAscOperands`).
  }];
}

def Noop : Pass<"ascendc-noop", "func::FuncOp"> {
  let summary = "Placeholder pass that performs no transformations";
  let constructor = "mlir::ascendc::createNoopPass()";
  let description = [{
    A no-operation pass useful for testing, pipeline debugging, or as a placeholder in pass schedules.
    The pass walks the function but performs no modifications to the IR.
  }];
}

def PrivatizeFunc : Pass<"ascendc-privatize-func", "ModuleOp"> {
  let summary = "Mark non-kernel functions as private and kernel functions as public for emission";
  let constructor = "mlir::ascendc::createPrivatizeFuncPass()";
  let description = [{
    Adjusts function visibility based on kernel status:
    - Functions without `ascendc.global` attribute → set as `private` (internal helper functions)
    - Functions with `ascendc.global` and body → set as `public` (kernel entry points)
    - Functions with `ascendc.global` but no body (declarations) → remain as-is

    This ensures only kernel entry points are exported while helper functions remain internal.
  }];
}

def PromoteCVBlock : Pass<"ascendc-promote-cv-block", "ModuleOp"> {
  let summary = "Detect kernel type and promote conditional AIC/AIV blocks for pure cube or vector kernels";
  let constructor = "mlir::ascendc::createPromoteCVBlockPass()";
  let description = [{
    Inspects each function for `if_aic` (AI Cube) and `if_aiv` (AI Vector) conditional blocks and sets the
    module-level `asc.kernel_type` attribute to `cube`, `vector`, or `mixed` accordingly. For pure cube or
    vector kernels, unwraps the conditional blocks by moving their body operations into the parent block and
    erasing the conditional wrapper. Mixed kernels retain their conditional structure since operations target
    different compute units.
  }];
}

def RefineCubePosition : Pass<"ascendc-refine-cube-position", "func::FuncOp"> {
  let summary = "Refine A1 cube tensor positions to B1 or C1 based on L0 consumer destinations";
  let constructor = "mlir::ascendc::createRefineCubePositionPass()";
  let description = [{
    Refines the memory position of `LocalTensorAutoOp` tensors initially placed in A1 (L1) by analyzing how
    they are consumed by L0 load and copy operations (`LoadDataL0V2Op`, `DataCopyL0Op`).

    The pass inspects the destination tensor position of each consumer and remaps the source tensor position:
    - Destination in A2 (L0A) -> source stays in A1
    - Destination in B2 (L0B) -> source refined to B1
    - Destination in C2 (bias) -> source refined to C1

    Only tensors with position A1 are processed. This refinement ensures cube tensors are placed in the
    correct hardware buffer (L1, B1, or C1) matching the actual data flow to L0A, L0B, or bias memory.
  }];
}

def ReuseTensorAllocation : Pass<"ascendc-reuse-tensor-allocation", "func::FuncOp"> {
  let summary = "Reuse freed on-chip allocations based on unroll_iter lifetime analysis";
  let constructor = "mlir::ascendc::createReuseTensorAllocationPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Reuses freed on-chip buffer allocations by inserting `LocalTensorReinterpretCastOp`.
    Covers VECCALC (UB), A1 (L1), A2 (L0A), B2 (L0B), and CO1 (L0C) positions.
    Reuse is restricted to tensors of the same position, since each position maps to
    a distinct hardware buffer.

    Reuse decisions are based on `asctile.unroll_iter` attribute: if top tensor's max
    unroll_iter is less than bottom tensor's min unroll_iter, their lifetimes don't
    overlap and reuse is valid.

    Input/output tensors cannot be reused across different unroll_iters because each
    iteration loads/stores different data. Reusing would cause data races. Input/output
    tensors can only be reused with tensors that have the same unroll_iter or no
    unroll_iter at all.

    For in/out tensors: all users share the same unroll_iter (guaranteed by upstream passes),
    so max == min. Two in/out tensors at different iterations can reuse each other.
    Two in/out tensors at the same iteration overlap and cannot reuse.

    For tensors without unroll_iter: fall back to position-based ordering within their scope.

    For cube positions (A1, A2, B2), allocation size is computed with cube-block alignment
    to match the downstream AllocateTensor pass.

    When reusing: larger tensor wraps smaller via reinterpret_cast, or smaller moved before larger and cast.
  }];
}

def ReuseUBAllocation : Pass<"ascendc-reuse-ub-allocation", "func::FuncOp"> {
  let summary = "Reuse freed UB allocations via reinterpret_cast to reduce memory consumption";
  let constructor = "mlir::ascendc::createReuseUBAllocationPass()";
  let options = [
    Option<"reuseInOut", "reuse-in-out", "bool", "false",
           "Allow reuse between input/output tensors even in same loop">,
  ];
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Analyzes tensor lifetimes and reuses freed UB allocations by inserting `LocalTensorReinterpretCastOp`:
    1. For each block, collects `LocalTensorAutoOp` tensors and their last users
    2. Sorts tensors by last user position (earlier-freed tensors can be reused first)
    3. Checks reusability: same block, non-overlapping lifetimes, static shape, VECCALC position
    4. When reusing: larger tensor wraps smaller via reinterpret_cast, or smaller moved before larger and cast

    Special handling for control flow: tensors used across different regions (if/else, while before/after) can be reused.
    Input/output tensors in same loop are not reused unless `reuse-in-out=true`.

    Example: two tensors with non-overlapping lifetimes reused:
    ```mlir
    // Before: separate allocations for sequential tensors
    %0 = ascendc.local_tensor_auto veccalc() : <333xi32>
    ascendc.data_copy_l2 %0, %gm, %c1
    %1 = ascendc.local_tensor_auto veccalc() : <333xi32>
    ascendc.data_copy_l2 %1, %gm, %c1

    // After: single allocation with reinterpret_cast
    %0 = ascendc.local_tensor_auto veccalc() : <333xi32>
    %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<333xi32> to !ascendc.local_tensor<333xi32>
    ascendc.data_copy_l2 %0, %gm, %c1
    ascendc.data_copy_l2 %1, %gm, %c1
    ```

    Example: tensors in different loops can be reused:
    ```mlir
    // Before: output tensor in loop 1, input tensor in loop 2
    scf.for { %output = ascendc.local_tensor_auto veccalc() output }
    scf.for { %input = ascendc.local_tensor_auto veccalc() input }

    // After: output reused as input across different loops
    %output = ascendc.local_tensor_auto veccalc() output
    %input = ascendc.reinterpret_cast %output
    scf.for { use %output }
    scf.for { use %input }
    ```
  }];
}

def UnifyBiasTensor : Pass<"ascendc-unify-bias-tensor", "func::FuncOp"> {
  let summary = "Unify C2 (bias) tensor with identical type and size, setting addr to 0";
  let constructor = "mlir::ascendc::createUnifyBiasTensorPass()";
  let description = [{
    Consolidates all `LocalTensorV3Op` operations in C2 position (bias tensor memory) into a single tensor.

    When multiple C2 tensors exist (typically created during loop unrolling or tiling), this pass:
    1. Keeps the first encountered C2 tensor
    2. Replaces all uses of subsequent C2 tensors with the first tensor
    3. Erases the duplicate tensor operations
    4. Sets the address of the unified tensor to 0

    This optimization ensures a single bias tensor allocation per function, which is required for
    correct `bufId`-based synchronization in `InsertBiasBufIdSync` pass.
  }];
}

def UnifyPipe : Pass<"ascendc-unify-pipe", "func::FuncOp"> {
  let summary = "Replace multiple PipeOp instances with a single unified pipe at function entry";
  let constructor = "mlir::ascendc::createUnifyPipePass()";
  let description = [{
    Consolidates multiple `PipeOp` declarations into a single pipe object at function entry.
    All uses of individual pipes are replaced with the unified pipe, and original `PipeOp` instances are erased.

    This ensures consistent pipe management across the kernel and reduces redundant TPipe object creation.
  }];
}

def VerifySync : Pass<"ascendc-verify-sync", "func::FuncOp"> {
  let summary = "Validate TQueBind synchronization correctness and emit warnings for mismatches";
  let constructor = "mlir::ascendc::createVerifySyncPass()";
  let description = [{
    Verifies proper pairing of TQueBind operations and emits warnings for incorrect usage:
    - `alloc_tensor` without corresponding `free_tensor`
    - `free_tensor` for already-freed tensor or without `alloc_tensor`
    - `enque_tensor` without corresponding `deque_tensor`
    - `deque_tensor` without `enque_tensor` in queue
    - Unexpected tensor uses between `enque_tensor` and `deque_tensor`

    This is a verification-only pass that helps detect synchronization bugs during development.
  }];
}

#endif // ASC_PASSES_TD
