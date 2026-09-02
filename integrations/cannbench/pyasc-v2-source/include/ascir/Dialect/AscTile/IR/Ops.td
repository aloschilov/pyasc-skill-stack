/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_OPS_TD
#define ASCTILE_OPS_TD

include "Attributes.td"
include "Dialect.td"
include "Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class AscTile_Op<string mnemonic, list<Trait> traits = []> : Op<AscTile_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::asctile";
}

def AccumulatorOp : AscTile_Op<"accumulator"> {
  let summary = "Create a zeros tensor in L0C or initialize with bias";
  let arguments = (ins Optional<LocalTensorOfRankAndType<[1], [BF16, F16, F32]>>:$bias);
  let results = (outs NumericLocalTensor:$result);
  let assemblyFormat = "($bias^)? attr-dict `:` qualified(type($result)) (`,` type($bias)^)?";
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def TensorOp : AscTile_Op<"tensor", [Pure]> {
  let summary = "Create tensor";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base, Variadic<I32>:$sizes);
  let results = (outs GlobalTensor:$result);
  let assemblyFormat = "$base `(` $sizes `)` attr-dict `:` qualified(type($base)) `,` qualified(type($result))";
  let hasVerifier = 1;
}

def DimOp : AscTile_Op<"dim", [Pure]> {
  let summary = "Obtain dimension of given tensor";
  let arguments = (ins GlobalTensor:$base, I32Attr:$index);
  let results = (outs I32:$result);
  let assemblyFormat = "$base `,` $index attr-dict `:` qualified(type($base))";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def CopyOp : AscTile_Op<"copy", [AllElementTypesMatch<["base", "result"]>]> {
  let summary = "Copy from a tensor into a new sub-tensor";
  let arguments = (ins NumericLocalTensor:$base, Variadic<I32>:$offsets);
  let results = (outs NumericLocalTensor:$result);
  let assemblyFormat = "$base `[` $offsets `]` attr-dict `:` qualified(type($base)) `,` qualified(type($result))";
  let hasCanonicalizeMethod = 1;
}

def LoadOp : AscTile_Op<"load", [AttrSizedOperandSegments, AllElementTypesMatch<["base", "result"]>,
  OptionalShapedMatchElemType<"result", "padValue">, MemoryEffectsOpInterface]> {
  let summary = "Load from the global tensor into a new local tensor";
  let arguments = (ins Arg<GlobalTensor, "tensor to load from", [MemRead]>:$base, Variadic<I32>:$offsets,
                   Optional<AscTile_Numeric>:$padValue, Variadic<I32>:$realShape);
  let results = (outs NumericLocalTensor:$result);
  let assemblyFormat = [{
    $base `[` $offsets `]` (`,` $padValue^)? (`,` `(` $realShape^ `)`)? attr-dict `:` qualified(type($base)) `,`
    qualified(type($result))
  }];
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def StoreOp : AscTile_Op<"store", [AttrSizedOperandSegments, AllElementTypesMatch<["base", "value"]>,
  MemoryEffectsOpInterface]> {
  let summary = "Store a local tensor into a global tensor";
  let arguments = (ins NumericLocalTensor:$value, Arg<GlobalTensor, "tensor to store to", [MemWrite]>:$base,
                   Variadic<I32>:$offsets, Variadic<I32>:$realShape);
  let assemblyFormat = [{
    $value `,` $base `[` $offsets `]` (`,` `(` $realShape^ `)`)? attr-dict `:` qualified(type($value)) `,`
    qualified(type($base))
  }];
  let hasVerifier = 1;
}

def GetValueOp : AscTile_Op<"get_value", [ShapedMatchElemType<"base", "result">, MemoryEffectsOpInterface]> {
  let summary = "Load value from a global tensor";
  let arguments = (ins Arg<GlobalTensor, "tensor to read from", [MemRead]>:$base, Variadic<I32>:$offsets);
  let results = (outs AscTile_Numeric:$result);
  let assemblyFormat = "$base `[` $offsets `]` attr-dict `:` qualified(type($base))";
}

def SetValueOp : AscTile_Op<"set_value", [AllElementTypesMatch<["base", "value"]>, MemoryEffectsOpInterface]> {
  let summary = "Store value into a global tensor";
  let arguments = (ins AnyTypeOf<[AscTile_Numeric, LocalTensorOfLengthAndType<[1], [AscTile_Numeric]>]>:$value,
                   Arg<GlobalTensor, "tensor to be updated", [MemWrite]>:$base, Variadic<I32>:$offsets);
  let assemblyFormat = "$value `,` $base `[` $offsets `]` attr-dict `:` type($value) `,` qualified(type($base))";
}

def YieldOp : AscTile_Op<"yield", [Terminator]> {
  let summary = "Terminator for operations with region";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let builders = [OpBuilder<(ins), "/* empty yield */">];
}

def CountMaskOp : AscTile_Op<"count_mask", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Apply mask (count mode) to nested operations";
  let arguments = (ins I64:$count, Optional<AscTile_Numeric>:$other);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "$count (`,` $other^ `:` type($other))? $region attr-dict";
}

def BitwiseMaskOp : AscTile_Op<"bitwise_mask", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Apply mask (bit mode) to nested operations";
  let arguments = (ins I64:$highBits, I64:$lowBits, Optional<AscTile_Numeric>:$other);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "$highBits `,` $lowBits (`,` $other^ `:` type($other))? $region attr-dict";
}

def CubeGroupOp : AscTile_Op<"cube_group", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Group operations targeting the AI Cube compute unit (L0A/L0B/L0C/L1/FIX/BT)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "(`(` $operands^ `:` type($operands) `)`)? $region (`:` type($results)^)? attr-dict";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def VectorGroupOp : AscTile_Op<"vector_group", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Group operations targeting the AI Vector compute unit (UB)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "(`(` $operands^ `:` type($operands) `)`)? $region (`:` type($results)^)? attr-dict";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

class AscTile_VecScalarOp<string mnemonic, string kind, list<Trait> traits = []> : AscTile_Op<mnemonic, traits # [
  Pure, AllTypesMatch<["base", "result"]>, ShapedMatchElemType<"base", "value">]> {
  let summary = "Compute element-wise " # kind # " of tensor and scalar";
  let arguments = (ins NumericLocalTensor:$base, AscTile_Numeric:$value);
  let results = (outs NumericLocalTensor:$result);
  let assemblyFormat = "$base `,` $value attr-dict `:` qualified(type($base))";
}

def AddSOp : AscTile_VecScalarOp<"adds", "addition">;
def SubSOp : AscTile_VecScalarOp<"subs", "subtraction">;
def MulSOp : AscTile_VecScalarOp<"muls", "multiplication">;
def ShLSOp : AscTile_VecScalarOp<"shls", "bitwise left shift">;
def ShRSOp : AscTile_VecScalarOp<"shrs", "bitwise right shift">;
def DivSOp : AscTile_VecScalarOp<"divs", "division">;
def MinSOp : AscTile_VecScalarOp<"mins", "minimum value">;
def MaxSOp : AscTile_VecScalarOp<"maxs", "maximum value">;
def LeakyReluOp : AscTile_VecScalarOp<"leaky_relu", "leaky ReLU">;

def CmpSOp : AscTile_Op<"cmps", [Pure, ShapedMatchElemType<"base", "value">, TypesMatchWith<
    "result type has i1 element type and same shape as operands", "base", "result", "::getI1SameShape($_self)">]> {
  let summary = "Compute element-wise comparison of tensor and scalar";
  let arguments = (ins NumericLocalTensor:$base, AscTile_Numeric:$value, AscTile_CompareModeAttr:$cmpMode);
  let results = (outs LocalTensorOf<[I1]>:$result);
  let assemblyFormat = "$cmpMode $base `,` $value attr-dict `:` qualified(type($base))";
}

def ReluOp : AscTile_Op<"relu", [Pure, AllTypesMatch<["operand", "result"]>]> {
  let summary = "Compute element-wise ReLU value of tensor";
  let arguments = (ins NumericLocalTensor:$operand);
  let results = (outs NumericLocalTensor:$result);
  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
}

def CastOp : AscTile_Op<"cast", [Pure, SameOperandsAndResultShape, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Cast tensor elements to other data type";
  let arguments = (ins NumericLocalTensor:$in, AscTile_RoundModeAttr:$roundMode);
  let results = (outs NumericLocalTensor:$out);
  let assemblyFormat = "$roundMode $in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
  let hasFolder = 1;
}

def ReshapeOp : AscTile_Op<"reshape", [Pure, SameOperandsAndResultElementType,
  DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Create tensor with same data but different shape";
  let arguments = (ins NumericLocalTensor:$in);
  let results = (outs NumericLocalTensor:$out);
  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
  let hasFolder = 1;
}

def ReduceAs1dOp : AscTile_Op<"reduce_as_1d", [Pure, AllElementTypesMatch<["operand", "result"]>]> {
  let summary = "Reduce tensor to 1-element tensor or a scalar value";
  let arguments = (ins NumericLocalTensor:$operand, AscTile_ReduceKindAttr:$kind);
  let results = (outs AnyTypeOf<[AscTile_Numeric, LocalTensorOfLengthAndType<[1], [AscTile_Numeric]>]>:$result);
  let assemblyFormat = "$kind $operand attr-dict `:` qualified(type($operand)) `,` type($result)";
}

def AtomicRMWOp : AscTile_Op<"atomic_rmw", [AllElementTypesMatch<["base", "value"]>, MemoryEffectsOpInterface]> {
  let summary = "Atomic accumulation the tensor data into a tensor";
  let arguments = (ins NumericLocalTensor:$value, Arg<GlobalTensor, "tensor to store to", [MemWrite]>:$base,
                   Variadic<I32>:$offsets, AscTile_AtomicKindAttr:$kind);
  let assemblyFormat = [{
    `<` $kind `>` $value `,` $base `[` $offsets `]` attr-dict `:` qualified(type($value)) `,` qualified(type($base))
  }];
}

def SoftmaxOp : AscTile_Op<"softmax", [Pure, AllTypesMatch<["operand", "result"]>]> {
  let summary = "Compute row-wise softmax of 1D or 2D tensor";
  Type allowedType = LocalTensorOfRankAndType<[1, 2], [AscTile_Numeric]>;
  let arguments = (ins allowedType:$operand);
  let results = (outs allowedType:$result);
  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
}

def RmsNormOp : AscTile_Op<"rms_norm", [Pure, AllElementTypesMatch<["input", "gamma"]>,
  AllTypesMatch<["input", "result"]>, ShapedMatchElemType<"input", "epsilon">]> {
  let summary = "Compute RmsNorm function over a tensor";
  let arguments = (ins FloatLocalTensor:$input, FloatLocalTensor:$gamma, AscTile_Float:$epsilon);
  let results = (outs FloatLocalTensor:$result);
  let assemblyFormat = [{
    $input `,` $gamma `,` $epsilon attr-dict `:` qualified(type($input)) `,` qualified(type($gamma))
  }];
}

def LayerNormOp : AscTile_Op<"layer_norm", [Pure, AllElementTypesMatch<["gamma", "beta"]>,
  AllTypesMatch<["input", "output"]>]> {
  let summary = "Compute LayerNorm function over a tensor";
  let arguments = (ins FloatLocalTensor:$input, FloatLocalTensor:$gamma, FloatLocalTensor:$beta,
                   F32:$epsilon, UnitAttr:$outputRstd);
  let results = (outs FloatLocalTensor:$output, FloatLocalTensor:$mean, FloatLocalTensor:$outputVarRstd);
  let assemblyFormat = [{
    $input `,` $gamma `,` $beta `,` $epsilon attr-dict `:` type($input) `,` type($gamma) `,` type($beta) `,` type($mean)
    `,` type($outputVarRstd)
  }];
}

def MatmulOp : AscTile_Op<"matmul", [Pure, AllElementTypesMatch<["matrixA", "matrixB"]>]> {
  let summary = "Compute matrix multiplication on 2D tiles with optional bias";
  Type inputType = LocalTensorOfRankAndType<[2], [AscTile_Float]>;
  let arguments = (ins inputType:$matrixA, inputType:$matrixB,
                   Optional<LocalTensorOfRankAndType<[1], [BF16, F16, F32]>>:$bias, UnitAttr:$hf32);
  let results = (outs LocalTensorOfRankAndType<[2], [F32]>:$result);
  let assemblyFormat = [{
    $matrixA `,` $matrixB (`,` $bias^)? attr-dict `:` type($matrixA) `,` type($matrixB) (`,` type($bias)^)? `->`
    type($result)
  }];
}

def MatmulAccOp : AscTile_Op<"matmul_acc", [AllElementTypesMatch<["matrixA", "matrixB"]>, MemoryEffectsOpInterface]> {
  let summary = "Compute matrix multiplication with accumulator on 2D tiles";
  Type inputType = LocalTensorOfRankAndType<[2], [AscTile_Float]>;
  let arguments = (ins Arg<LocalTensorOfRankAndType<[2], [F32]>, "tensor to store to", [MemWrite]>:$acc,
                   inputType:$matrixA, inputType:$matrixB, UnitAttr:$hf32);
  let assemblyFormat = "$acc `,` $matrixA `,` $matrixB attr-dict `:` type($acc) `,` type($matrixA) `,` type($matrixB)";
}

def BroadcastOp : AscTile_Op<"broadcast", [Pure, AllElementTypesMatch<["operand", "result"]>]> {
  let summary = "Broadcast tensor to the provided shape";
  let arguments = (ins NumericLocalTensor:$operand);
  let results = (outs NumericLocalTensor:$result);
  let assemblyFormat = "$operand attr-dict `:` qualified(type($operand)) `to` qualified(type($result))";
  let hasFolder = 1;
}

def ReduceOp : AscTile_Op<"reduce", [Pure, AllElementTypesMatch<["operand", "result"]>]> {
  let summary = "Reduce tensor to the provided shape";
  let arguments = (ins NumericLocalTensor:$operand, I32ArrayAttr:$dims, AscTile_ReduceKindAttr:$kind);
  let results = (outs NumericLocalTensor:$result);
  let assemblyFormat = "$kind $operand attr-dict `:` qualified(type($operand)) `,` qualified(type($result))";
}

def StoreFixpipeOp : AscTile_Op<"store_fixpipe", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Store a local tensor into a global tensor via FixPipe";
  let arguments = (ins NumericLocalTensor:$value, Arg<GlobalTensor, "tensor to store to", [MemWrite]>:$base,
                   Variadic<I32>:$offsets, Variadic<I32>:$realShape, UnitAttr:$relu, UnitAttr:$quantize);
  let assemblyFormat = [{
    $value `,` $base `[` $offsets `]` (`,` `(` $realShape^ `)`)? attr-dict `:` qualified(type($value)) `,`
    qualified(type($base))
  }];
  let hasVerifier = 1;
}

def CopyFixpipeOp : AscTile_Op<"copy_fixpipe"> {
  let summary = "Store a tensor into a sub-tensor via FixPipe";
  let arguments = (ins NumericLocalTensor:$base, Variadic<I32>:$offsets, UnitAttr:$relu, UnitAttr:$quantize);
  let results = (outs NumericLocalTensor:$result);
  let assemblyFormat = "$base `[` $offsets `]` attr-dict `:` qualified(type($base)) `,` qualified(type($result))";
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def TransposeOp : AscTile_Op<"transpose", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Transpose a tensor";
  let arguments = (ins NumericLocalTensor:$operand, I32ArrayAttr:$dims);
  let results = (outs NumericLocalTensor:$result);
  let assemblyFormat = "$operand `,` $dims attr-dict `:` qualified(type($operand)) `to` qualified(type($result))";
  let hasVerifier = 1;
}

def InlineVFOp : AscTile_Op<"inline_vf", [Pure]> {
  let summary = "Declare vector function provided as C++ code";
  let arguments = (ins Variadic<NumericLocalTensor>:$inputs, StrAttr:$code);
  let results = (outs NumericLocalTensor:$result);
  let assemblyFormat = "`(` $inputs `)` $code attr-dict `:` functional-type($inputs, results)";
}

def BitwiseNotOp : AscTile_Op<"bitwise_not", [Pure, AllTypesMatch<["operand", "result"]>]> {
  let summary = "Compute the bitwise NOT of a tensor";
  let arguments = (ins IntLocalTensor:$operand);
  let results = (outs IntLocalTensor:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def GatherOp : AscTile_Op<"gather", [AllElementTypesMatch<["src", "result"]>, AttrSizedOperandSegments,
  OptionalShapedMatchElemType<"result", "padValue">, MemoryEffectsOpInterface]> {
  let summary = "Load elements from a global tensor by index";
  let arguments = (ins Arg<GlobalTensor, "tensor to read from", [MemRead]>:$src, Variadic<I32>:$offsets,
                   I32Attr:$dim, LocalTensorOfRankAndType<[1], [I8, I16, I32, I64]>:$index, UnitAttr:$checkBounds,
                   Optional<I32>:$numIndices, Optional<AscTile_Numeric>:$padValue);
  let results = (outs NumericLocalTensor:$result);
  let assemblyFormat = [{
    $src `[` $offsets `]` `,` $index (`,` $numIndices^ )? `,` $dim (`,` $padValue^ )? attr-dict `:` type($src) `,`
    type($index) `,` type($result)
  }];
}

def ScatterOp : AscTile_Op<"scatter", [AllElementTypesMatch<["dst", "src"]>, AttrSizedOperandSegments,
  MemoryEffectsOpInterface]> {
  let summary = "Store elements by index to a global tensor";
  let arguments = (ins NumericLocalTensor:$src, I32Attr:$dim, LocalTensorOfRankAndType<[1], [I8, I16, I32, I64]>:$index,
                   Arg<GlobalTensor, "tensor to store to", [MemWrite]>:$dst, Variadic<I32>:$offsets,
                   UnitAttr:$checkBounds, Optional<I32>:$numIndices);
  let assemblyFormat = [{
    $dst `[` $offsets `]` `,` $index `,` $src ( `,` $numIndices^ )? `,` $dim attr-dict `:` type($dst) `,` type($index)
    `,` type($src)
  }];
}

#endif // ASCTILE_OPS_TD
