/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_TYPES_TD
#define ASCTILE_TYPES_TD

include "Dialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

class AscTile_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<AscTile_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def GlobalTensor : Type<CPred<"::llvm::isa<::mlir::asctile::GlobalTensorType>($_self)">, "global tensor",
  "::mlir::asctile::GlobalTensorType">;

def LocalTensor : Type<CPred<"::llvm::isa<::mlir::asctile::LocalTensorType>($_self)">, "local tensor",
  "::mlir::asctile::LocalTensorType">;

def IsLocalTensorTypePred : And<[
  CPred<"::llvm::isa<::mlir::asctile::LocalTensorType>($_self)">,
  CPred<"::llvm::cast<::mlir::asctile::LocalTensorType>($_self).getRank() > 0">,
]>;

class IsLocalTensorOfRankPred<list<int> allowedRanks> : And<[
  IsLocalTensorTypePred,
  Or<!foreach(allowedlength, allowedRanks,
     CPred<"::llvm::cast<::mlir::asctile::LocalTensorType>($_self).getRank() == " # allowedlength>)>,
]>;

class IsLocalTensorOfLengthPred<list<int> allowedLengths> : And<[
  IsLocalTensorTypePred,
  Or<!foreach(allowedlength, allowedLengths,
     CPred<"::llvm::cast<::mlir::asctile::LocalTensorType>($_self).getNumElements() == " # allowedlength>)>,
]>;

class LocalTensorOf<list<Type> allowedTypes> : ShapedContainerType<allowedTypes, IsLocalTensorTypePred, "local tensor",
  "::mlir::asctile::LocalTensorType">;

class LocalTensorOfRank<list<int> allowedRanks> : Type<
  IsLocalTensorOfRankPred<allowedRanks>, " of ranks " # !interleave(allowedRanks, "/"),
  "::mlir::asctile::LocalTensorType">;

class LocalTensorOfRankAndType<list<int> allowedRanks, list<Type> allowedTypes> : AllOfType<
  [LocalTensorOf<allowedTypes>, LocalTensorOfRank<allowedRanks>],
  LocalTensorOf<allowedTypes>.summary # LocalTensorOfRank<allowedRanks>.summary, "::mlir::asctile::LocalTensorType">;

class LocalTensorOfLength<list<int> allowedLengths> : Type<IsLocalTensorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"), "::mlir::asctile::LocalTensorType">;

class LocalTensorOfLengthAndType<list<int> allowedLengths, list<Type> allowedTypes> : AllOfType<
  [LocalTensorOf<allowedTypes>, LocalTensorOfLength<allowedLengths>],
  LocalTensorOf<allowedTypes>.summary # LocalTensorOfLength<allowedLengths>.summary,
  "::mlir::asctile::LocalTensorType">;

def AscTile_Float : AnyTypeOf<[F16, BF16, F32, F64]>;
def AscTile_Int : AnyTypeOf<[I8, I16, I32, I64]>;
def AscTile_Numeric : AnyTypeOf<[AscTile_Float, AscTile_Int]>;

def FloatLocalTensor : LocalTensorOf<[AscTile_Float]>;
def IntLocalTensor : LocalTensorOf<[AscTile_Int]>;
def NumericLocalTensor : LocalTensorOf<[AscTile_Numeric]>;

class ShapedMatchElemType<string shapedType, string type> : TypesMatchWith<
  "type of '" # type # "' matches element type of '" # shapedType # "'", shapedType, type,
  "::llvm::cast<ShapedType>($_self).getElementType()">;

class OptionalShapedMatchElemType<string shapedType, string type> : OptionalTypesMatchWith<
  "type of '" # type # "' matches element type of '" # shapedType # "'", shapedType, type,
  "::llvm::cast<ShapedType>($_self).getElementType()">;

#endif // ASCTILE_TYPES_TD
