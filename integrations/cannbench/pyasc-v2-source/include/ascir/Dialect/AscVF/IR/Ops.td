/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCVF_OPS_TD
#define ASCVF_OPS_TD

include "Dialect.td"
include "Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class AscVF_Op<string mnemonic, list<Trait> traits = []> : Op<AscVF_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::ascvf";
}

def LoadOp : AscVF_Op<"load"> {
  let description = "Load RegTensor from LocalTensor";
  let arguments = (ins AnyType:$dstReg, AnyType:$srcTensor, AnyType:$offset);
  let assemblyFormat = [{
    $dstReg `,` $srcTensor `[` $offset `]` attr-dict `:` type($dstReg) `,` type($srcTensor) `,` type($offset)
  }];
  let hasVerifier = 1;
}

def StoreOp : AscVF_Op<"store"> {
  let description = "Store RegTensor to LocalTensor";
  let arguments = (ins AnyType:$dstTensor, AnyType:$offset, AnyType:$srcReg, AnyType:$mask);
  let assemblyFormat = [{
    $dstTensor `[` $offset `]` `,` $srcReg `,` $mask attr-dict `:` type($dstTensor) `,` type($offset) `,` type($srcReg)
    `,` type($mask)
  }];
  let hasVerifier = 1;
}

def VFGroupOp : AscVF_Op<"vf_group",
                         [SingleBlockImplicitTerminator<"::mlir::ascvf::YieldOp">, AttrSizedOperandSegments]> {
  let summary = "block with fused operations";
  let arguments = (ins Variadic<AnyType>:$dstList, Variadic<AnyType>:$srcList, AnyType:$calCount);
  let regions = (region SizedRegion<1> : $region);
  let assemblyFormat = "operands `:` qualified(type(operands)) $region attr-dict";
  let extraClassDeclaration = [{
    Type getGroupType();
  }];
}

def VFForOp : AscVF_Op<"vf_for", [AutomaticAllocationScope, SingleBlockImplicitTerminator<"::mlir::ascvf::YieldOp">,
                                  RecursivelySpeculatable, RecursiveMemoryEffects]> {
  let arguments = (ins AnySignlessIntegerOrIndex:$upperBound);
  let regions = (region SizedRegion<1>:$region);

  let extraClassDeclaration = [{
    BlockArgument getInductionVar() { return getBody()->getArgument(0); }
    int64_t getLowerBoundAsInt() { return 0; }
    int64_t getStepAsInt() { return 1; }
    void setUpperBound(Value bound) { getOperation()->setOperand(0, bound); }
    Type getUnderlyingType() { return IntegerType::get(getContext(), 16, IntegerType::Unsigned); }
  }];

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$upperBound)>];

  let hasCanonicalizeMethod = 1;
  let assemblyFormat = "operands `:` qualified(type(operands)) $region attr-dict";
  let hasVerifier = 1;
}

def VecScopeOp : AscVF_Op<"vec_scope", [SingleBlockImplicitTerminator<"::mlir::ascvf::YieldOp">]> {
  let regions = (region SizedRegion<1> : $region);
  let assemblyFormat = "$region attr-dict";
}

def YieldOp : AscVF_Op<"yield", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

#endif // ASCVF_OPS_TD
