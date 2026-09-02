/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_ATTRIBUTES_TD
#define ASCTILE_ATTRIBUTES_TD

include "Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class AscTile_Attr<string name, string attrMnemonic, list<Trait> traits = []> : AttrDef<AscTile_Dialect, name, traits> {
  let cppNamespace = "::mlir::asctile";
  let mnemonic = attrMnemonic;
}

class AscTile_EnumAttr<EnumAttrInfo info, string name = ""> : EnumAttr<AscTile_Dialect, info, name> {
  let assemblyFormat = "`<` $value `>`";
}

def AscTile_TensorLocationAttr : I32EnumAttr<"TensorLocation", "tensor location", [
  I32EnumAttrCase<"Auto", 0, "auto">,
  I32EnumAttrCase<"L0A", 23>,
  I32EnumAttrCase<"L0B", 24>,
  I32EnumAttrCase<"L0C", 25>,
  I32EnumAttrCase<"UB", 26>,
  I32EnumAttrCase<"L1", 27>,
  I32EnumAttrCase<"FIX", 28>,
  I32EnumAttrCase<"BT", 29>,
]> {
  let cppNamespace = "::mlir::asctile";
  let description = "Tensor memory location";
}

def AscTile_AtomicKindAttr : I32EnumAttr<"AtomicKind", "kind of atomic", [
  I32EnumAttrCase<"Add", 0>,
  I32EnumAttrCase<"Max", 1>,
  I32EnumAttrCase<"Min", 2>,
]> {
  let cppNamespace = "::mlir::asctile";
  let description = "Kind of atomic operation";
}

def AscTile_CompareModeAttr : I32EnumAttr<"CompareMode", "comparsion type", [
  I32EnumAttrCase<"LT", 0>,
  I32EnumAttrCase<"GT", 1>,
  I32EnumAttrCase<"EQ", 2>,
  I32EnumAttrCase<"LE", 3>,
  I32EnumAttrCase<"GE", 4>,
  I32EnumAttrCase<"NE", 5>,
]> {
  let cppNamespace = "::mlir::asctile";
  let description = "Comparsion mode";
}

def AscTile_ReduceKind : I32EnumAttr<"ReduceKind", "kind of reduce", [
  I32EnumAttrCase<"Sum", 0, "sum">,
  I32EnumAttrCase<"Max", 1, "max">,
  I32EnumAttrCase<"Min", 2, "min">,
  I32EnumAttrCase<"Prod", 3, "prod">,
  I32EnumAttrCase<"Mean", 4, "mean">,
  I32EnumAttrCase<"All", 5, "all">,
  I32EnumAttrCase<"Any", 6, "any">,
  I32EnumAttrCase<"XorSum", 7, "xor_sum">,
]> {
  let cppNamespace = "::mlir::asctile";
  let description = "Kind of reduce operation";
  let genSpecializedAttr = 0;
}

def AscTile_ReduceKindAttr : AscTile_EnumAttr<AscTile_ReduceKind, "kind">;

def AscTile_RoundMode : I32EnumAttr<"RoundMode", "cast rounding mode", [
  I32EnumAttrCase<"NoRound", 0, "noround">,
  I32EnumAttrCase<"Rint", 1, "rint">,
  I32EnumAttrCase<"Floor", 2, "floor">,
  I32EnumAttrCase<"Ceil", 3, "ceil">,
  I32EnumAttrCase<"Round", 4, "round">,
  I32EnumAttrCase<"Trunc", 5, "trunc">,
  I32EnumAttrCase<"Odd", 6, "odd">,
  I32EnumAttrCase<"Default", 100, "default">,
]> {
  let cppNamespace = "::mlir::asctile";
  let description = "Cast rounding mode";
  let genSpecializedAttr = 0;
}

def AscTile_RoundModeAttr : AscTile_EnumAttr<AscTile_RoundMode, "round_mode">;

def AscTile_GlobalTensorAttr : AscTile_Attr<"GlobalTensor", "global"> {
  let summary = "An encoding for a global tensor";
}

def AscTile_LocalTensorAttr : AscTile_Attr<"LocalTensor", "local"> {
  let summary = "An encoding for a local tensor";
  let parameters = (ins "TensorLocation":$loc);
  let assemblyFormat = "`<` $loc `>`";
}

#endif // ASCTILE_ATTRIBUTES_TD
