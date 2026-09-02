/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASCTILE_IR_ASCTILE_H
#define ASCIR_DIALECT_ASCTILE_IR_ASCTILE_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "ascir/Dialect/AscTile/IR/AscTileDialect.h.inc"

#include "ascir/Dialect/AscTile/IR/AscTileEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ascir/Dialect/AscTile/IR/AscTileAttributes.h.inc"

namespace mlir {
namespace asctile {

class GlobalTensorType : public RankedTensorType {
public:
    using RankedTensorType::RankedTensorType;

    static RankedTensorType get(ArrayRef<int64_t> shape, Type elementType);
    static TypeID resolveTypeID();
    static bool classof(Type type);
};

class LocalTensorType : public RankedTensorType {
public:
    using RankedTensorType::RankedTensorType;

    TensorLocation getLoc() const;

    static RankedTensorType get(ArrayRef<int64_t> shape, Type elementType, TensorLocation loc = TensorLocation::UB);
    static TypeID resolveTypeID();
    static bool classof(Type type);
};

} // namespace asctile
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "ascir/Dialect/AscTile/IR/AscTileTypes.h.inc"

#define GET_OP_CLASSES
#include "ascir/Dialect/AscTile/IR/AscTileOps.h.inc"

namespace mlir {
namespace asctile {

void registerExternalModels(DialectRegistry& registry);

} // namespace asctile
} // namespace mlir

#endif // ASCIR_DIALECT_ASCTILE_IR_ASCTILE_H
