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
#include "ascir/Dialect/Utils/Inlining.h"

#include "mlir/IR/DialectImplementation.h"

#include "ascir/Dialect/AscTile/IR/AscTileDialect.cpp.inc"

using namespace mlir;
using namespace mlir::asctile;

//===----------------------------------------------------------------------===//
// AscTileDialect
//===----------------------------------------------------------------------===//

void AscTileDialect::initialize()
{
    registerAttributes();
    registerTypes();
    registerOps();
}

//===----------------------------------------------------------------------===//
// External models
//===----------------------------------------------------------------------===//

void asctile::registerExternalModels(DialectRegistry& registry)
{
    registry.addExtension(
        +[](MLIRContext*, AscTileDialect* dialect) { dialect->addInterface<ascir::PermissiveInlinerInterface>(); });
}
