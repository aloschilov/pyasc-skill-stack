/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOWERTOASC_PASSES_TD
#define LOWERTOASC_PASSES_TD

include "mlir/Pass/PassBase.td"

def DisplaceConcat : Pass<"asclower-displace-concat", "func::FuncOp"> {
  let summary = "Transform concat operations by rewriting related tensor allocations to eliminate actual data copies";
  let constructor = "mlir::asclower::createDisplaceConcatPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
  let description = [{
    Rewrites `tensor.concat` operations to avoid unnecessary data copies by adjusting tensor allocations:
    1. **Optimization phase**: If all concat operands are from `LocalTensorAutoOp` with same position, creates
       a single combined tensor at function entry and rewrites each operand's allocation as a sub-index view
       into this combined tensor. The concat itself becomes a no-op (just an unrealized conversion cast).
    2. **Fallback phase**: If operands cannot be optimized, creates a destination tensor and inserts
       `DataCopyL2Op` operations to copy each operand into its corresponding sub-index region.

    This pass eliminates redundant UB allocations and copies when concatenating temporaries, reducing memory usage.
  }];
}

def ExpandMask : Pass<"asclower-expand-mask", "func::FuncOp"> {
  let summary = "Expand mask regions by moving nested operations outside and updating mask parameters";
  let constructor = "mlir::asclower::createExpandMaskPass()";
  let dependentDialects = ["emitasc::EmitAscDialect"];
  let description = [{
    Transforms `asctile.count_mask` and `asctile.bitwise_mask` regions by:
    1. Moving all operations from inside the mask region to outside (before the mask op)
    2. Lowering L2 operations inside the mask to L0 variants via `populateLowerToL0Patterns`
    3. Updating each L0 operation's mask parameter to use the mask region's count or bitwise mask
    4. If the mask has `other` value, inserting `DuplicateL2Op` to fill the destination with that value

    After expansion, the mask region op is erased and all operations execute with explicit mask parameters.
  }];
}

def ExpandMath : Pass<"asclower-expand-math", "func::FuncOp"> {
  let summary = "Expand math operations that lack direct hardware support into equivalent sequences";
  let constructor = "mlir::asclower::createExpandMathPass()";
  let dependentDialects = ["arith::ArithDialect"];
  let description = [{
    Applies MLIR's standard math expansion patterns to decompose operations without direct Ascend C support:
    - `rsqrt(x)` → `1.0 / sqrt(x)`
    - `exp2(x)` → `exp(x * ln(2))`

    These expanded forms use math operations that have direct Ascend C library support.
  }];
}

def LowerArith : Pass<"asclower-arith", "func::FuncOp"> {
  let summary = "Lower arith dialect constant, bitcast, and negf operations to ascendc equivalents";
  let constructor = "mlir::asclower::createLowerArithPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Converts selected arithmetic operations on tensor types to Ascend C equivalents:
    - **Splat constants**: Creates `LocalTensorAutoOp` + `DuplicateL2Op` to broadcast scalar to all elements
    - **Dense non-splat constants**: Creates tensor + series of `LocalTensorSetValueOp` for each element
    - **Bitcast**: Converts to `LocalTensorReinterpretCastOp` for type reinterpretation
    - **Negf**: Converts to `MulsL2Op` with -1.0 scalar to negate floating-point tiles
    - **SelectOp**: Creates `SelectL0Op` with VSEL_TENSOR_TENSOR mode

    `asctile::LocalTensorType` is converted to `ascendc::LocalTensorType` via unrealized conversion casts.
  }];
}

def LowerArithBinary : Pass<"asclower-arith-binary", "func::FuncOp"> {
  let summary = "Lower binary arith dialect operations on tiles to ascendc L2 vector operations";
  let constructor = "mlir::asclower::createLowerArithBinaryPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
  let description = [{
    Converts binary arithmetic operations on tensor types to Ascend C L2-level vector operations.
    Supports floating-point and integer arithmetic (add, sub, mul, div, max, min variants).
    For bitwise operations (andi, ori), performs reinterpret casting to i8 elements if needed since
    the hardware processes bitwise ops on 8-bit granularity.

    Each operation creates a destination tensor allocation and the corresponding L2 compute op with calCount.
  }];
}

def LowerAscTile : Pass<"asclower-asctile", "func::FuncOp"> {
  let summary = "Lower asctile operations to ascendc operations representing framework and high-level API";
  let constructor = "mlir::asclower::createLowerAscTilePass()";
  let dependentDialects = [
    "arith::ArithDialect", "ascvf::AscVFDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect",
  ];
  let description = [{
    Converts AscTile high-level operations to Ascend C and AscVF equivalents. Handles tensor setup, neural network
    operations (softmax, rmsnorm, layernorm) with computed tiling parameters, reshape and broadcast, reductions with
    pattern-based shape decomposition, inline vector fusion code, and other.
  }];
}

def LowerAscTileDataTransfer : Pass<"asclower-asctile-data-transfer", "func::FuncOp"> {
  let summary = "Lower asctile data transfer operations (load, store, copy) to ascendc memory operations";
  let constructor = "mlir::asclower::createLowerAscTileDataTransferPass()";
  let dependentDialects = [
    "arith::ArithDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect", "emitc::EmitCDialect",
    "scf::SCFDialect"
  ];
  let description = [{
    Converts AscTile memory transfer operations to Ascend C equivalents:
    - **LoadOp (UB destination)**: Creates `DataCopyPadExtL0Op` from GM to UB with optional padding
    - **LoadOp (L1 destination)**: Creates `DataCopyL2Op` with Nd2Nz params for cube matrix format
      - Supports `transposeAL1` attribute for transposed matrix A loads (uses Dn2Nz params when srcDValue > 65535)
      - Supports `transposeBL1` attribute for transposed matrix B loads
    - **StoreOp (from UB)**: Creates `DataCopyPadExtL2Op` to GM
    - **StoreFixpipeOp (from L0C)**: Creates `FixpipeOp` with optional relu/quantize flags
    - **CopyFixpipeOp (L0C to L1/UB)**: Creates `FixpipeOp` with NZ layout for L0C-to-L1 or L0C-to-UB transfer
    - **CopyOp (L1 to L0A/L0B)**: Creates `LoadDataG2LOp` or `LoadDataWithTransposeOp` with fractal layout
      - Supports `transposeAL0` attribute for transposed matrix A copies
      - Supports `transposeBL0` attribute for transposed matrix B copies
    - **GetValueOp/SetValueOp**: Creates global tensor scalar get/set operations

    Handles both static and dynamic tensor shapes, computing linear offsets from multi-dimensional indices.
    Quantization modes (F322F16, DEQF16, QF322B8_PRE, REQ8) are inferred from source/destination type pairs.
  }];
}

def LowerAscTileToBasic : Pass<"asclower-asctile-to-basic", "func::FuncOp"> {
  let summary = "Lower asctile operations to ascendc operations representing basic API";
  let constructor = "mlir::asclower::createLowerAscTileToBasicPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect"];
  let description = [{
    Converts AscTile basic operations to their Ascend C equivalents. Handles element-wise arithmetic with scalar
    operands (add, sub, mul, div, min, max, shifts), unary operations (relu, cast, leaky_relu, bitwise_not), matrix
    multiplication (matmul, matmul_acc), reductions (sum, max, min), transpose on UB tensors, and scalar comparisons.
    For C310 target architectures, scalar subtraction and division are lowered through a duplicate-then-vector pattern
    instead of direct scalar operations.
  }];
}

def LowerAtomic : Pass<"asclower-atomic", "func::FuncOp"> {
  let summary = "Lower atomic read-modify-write operations to ascendc atomic store sequences";
  let constructor = "mlir::asclower::createLowerAtomicPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect"];
  let description = [{
    Converts `asctile.atomic_rmw` operations to atomic store sequences:
    1. Sets atomic mode via `SetAtomicAddOp`, `SetAtomicMaxOp`, or `SetAtomicMinOp`
    2. Converts `atomic_rmw` to `asctile.store` (which is later lowered by `LowerAscTileDataTransfer`)
    3. Resets atomic mode via `SetAtomicNoneOp`

    Supported atomic kinds: Add (accumulator), Max (maximum), Min (minimum).
    The atomic mode affects how subsequent stores modify existing memory values.
  }];
}

def LowerMath : Pass<"asclower-math", "func::FuncOp"> {
  let summary = "Lower math dialect unary operations on tiles to ascendc library and L2 operations";
  let constructor = "mlir::asclower::createLowerMathPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Converts math dialect unary operations on tensor types to Ascend C equivalents.
    Library operations (log, erf, trigonometric functions, ceil, floor, round) use Ascend C's
    high-precision library calls with optional shared temporary buffer.
    Direct L2 operations (abs, exp, sqrt, rsqrt) map to hardware-accelerated vector instructions.

    Each operation creates a destination tensor allocation and the corresponding compute op with calCount.
  }];
}

def LowerSCF : Pass<"asclower-scf", "func::FuncOp"> {
  let summary = "Convert tensor type operands and results in scf control flow operations to local tensor types";
  let constructor = "mlir::asclower::createLowerSCFPass()";
  let dependentDialects = ["ascendc::AscendCDialect", "scf::SCFDialect"];
  let description = [{
    Uses MLIR's `populateSCFStructuralTypeConversionsAndLegality` to convert scf operations with tensor types:
    - **scf.if**: Result types converted from `asctile::LocalTensorType` to `ascendc::LocalTensorType`
    - **scf.for**: Iter argument types converted, loop body operations wrapped with conversion casts
    - **scf.while**: Before/after region argument types converted

    This enables scf control flow to work with converted tensor types after AscTile lowering.
    Uses `TensorTypeConverter` to map `asctile::LocalTensorType` → `ascendc::LocalTensorType`.
  }];
}

def LowerTensor : Pass<"asclower-tensor", "func::FuncOp"> {
  let summary = "Lower tensor dialect operations to ascendc operations";
  let constructor = "mlir::asclower::createLowerTensorPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
}

def RealizeConversionCast : Pass<"asclower-realize-conversion-cast", "func::FuncOp"> {
  let summary = "Convert unrealized conversion casts between tensor types into ascendc reinterpret cast operations";
  let constructor = "mlir::asclower::createRealizeConversionCastPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Simplifies and materializes `UnrealizedConversionCastOp` chains between tensor types:
    1. **FoldCastChain**: Eliminates redundant cast chains where source and destination types match
       (e.g., cast(cast(x → y) → x) → x, or three-level chains)
    2. **RewriteReinterpretCast**: Converts nested casts between compatible tensor types to
       `LocalTensorReinterpretCastOp` when memory size is identical

    Compatible tensors have same total memory size (numElements × bitWidth), enabling type reinterpretation
    without data movement. This pass cleans up conversion artifacts from the dialect conversion process.
  }];
}

def RedressI1Tensor : Pass<"asclower-redress-i1-tensor", "func::FuncOp"> {
  let summary = "Convert i1 tensor types to packed i8 representation and adjust constants accordingly";
  let constructor = "mlir::asclower::createRedressI1TensorPass()";
  let dependentDialects = ["arith::ArithDialect", "asctile::AscTileDialect"];
  let description = [{
    Rewrites i1 (boolean) tensor types to use packed i8 representation before lowering:
    - **LocalTensorType conversion**: `tensor<Nxi1>` → `tensor<ceil(N/8)xi8>` (8 boolean values packed per byte)
    - **Splat constants**: `dense<true>` → `dense<-1>` (all bits set), `dense<false>` → `dense<0>`
    - **Dense constants**: Bit-packed representation where bit index corresponds to original element index

    This preprocessing enables hardware support for boolean operations, which operate on 8-bit granularity.
    After redressing, LowerAscTileI1 can convert comparison/select operations to packed L0 variants.
  }];
}

#endif // LOWERTOASC_PASSES_TD
