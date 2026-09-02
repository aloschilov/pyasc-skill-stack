/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <unordered_set>

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_REFINECUBEPOSITION
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

std::optional<ascendc::TPosition> getPossiblePosition(Operation* op)
{
    auto pos = llvm::TypeSwitch<Operation*, std::optional<ascendc::TPosition>>(op)
                   .Case<ascendc::LoadDataL0V2Op, ascendc::DataCopyL0Op>(
                       [](auto op) { return getAllocationRoot(op.getDst()).getPosition(); })
                   .Default([](auto) { return std::nullopt; });
    if (!pos)
        return std::nullopt;
    switch (pos.value()) {
        case ascendc::TPosition::A2:
            return ascendc::TPosition::A1;
        case ascendc::TPosition::B2:
            return ascendc::TPosition::B1;
        case ascendc::TPosition::C2:
            return ascendc::TPosition::C1;
        default:
            return std::nullopt;
    }
}

void refineCubePosition(ascendc::LocalTensorAutoOp tensor)
{
    if (tensor.getPosition() != ascendc::TPosition::A1)
        return;
    auto users = collectAllUsers(tensor);
    std::unordered_set<ascendc::TPosition> possiblePositions;
    for (auto* use : users) {
        if (auto pos = getPossiblePosition(use))
            possiblePositions.insert(pos.value());
    }
    if (auto it = possiblePositions.begin(); it != possiblePositions.end())
        tensor.setPosition(*it);
}

struct RefineCubePositionPass : public ascendc::impl::RefineCubePositionBase<RefineCubePositionPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk(refineCubePosition);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createRefineCubePositionPass()
{
    return std::make_unique<RefineCubePositionPass>();
}
