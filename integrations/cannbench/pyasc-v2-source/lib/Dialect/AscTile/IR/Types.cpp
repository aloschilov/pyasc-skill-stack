/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/IR/AscTile.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::asctile;

#define GET_TYPEDEF_CLASSES
#include "ascir/Dialect/AscTile/IR/AscTileTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// GlobalTensorType
//===----------------------------------------------------------------------===//

RankedTensorType GlobalTensorType::get(ArrayRef<int64_t> shape, Type elementType)
{
    return RankedTensorType::get(shape, elementType, GlobalTensorAttr::get(elementType.getContext()));
}

TypeID GlobalTensorType::resolveTypeID() { return TypeID::get<RankedTensorType>(); }

bool GlobalTensorType::classof(Type type)
{
    auto base = llvm::dyn_cast<RankedTensorType>(type);
    return base && llvm::isa<GlobalTensorAttr>(base.getEncoding());
}

//===----------------------------------------------------------------------===//
// LocalTensorType
//===----------------------------------------------------------------------===//

TensorLocation LocalTensorType::getLoc() const { return llvm::cast<LocalTensorAttr>(getEncoding()).getLoc(); }

RankedTensorType LocalTensorType::get(ArrayRef<int64_t> shape, Type elementType, TensorLocation loc)
{
    return RankedTensorType::get(shape, elementType, LocalTensorAttr::get(elementType.getContext(), loc));
}

TypeID LocalTensorType::resolveTypeID() { return TypeID::get<RankedTensorType>(); }

bool LocalTensorType::classof(Type type)
{
    auto base = llvm::dyn_cast<RankedTensorType>(type);
    return base && llvm::isa<LocalTensorAttr>(base.getEncoding());
}

//===----------------------------------------------------------------------===//
// AscTileDialect
//===----------------------------------------------------------------------===//

void AscTileDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "ascir/Dialect/AscTile/IR/AscTileTypes.cpp.inc"
        >();
}
