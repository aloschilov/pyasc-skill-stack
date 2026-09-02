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

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_COMPUTEMEMORYCONSUMPTION
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

StringLiteral positionToStr(TPosition position)
{
    switch (position) {
        case ascendc::TPosition::A1:
            return "L1";
        case ascendc::TPosition::A2:
            return "L0A";
        case ascendc::TPosition::B2:
            return "L0B";
        case ascendc::TPosition::CO1:
            return "L0C";
        case ascendc::TPosition::VECCALC:
            return "UB";
        case ascendc::TPosition::C2:
            return "BT";
        default:
            llvm_unreachable("unexpected TPosition value");
    }
}

auto calculateMemoryConsumption(ModuleOp moduleOp)
{
    std::map<StringLiteral, int64_t> mem;
    moduleOp.walk([&mem](ascendc::LocalTensorV3Op op) {
        mem[positionToStr(op.getPos())] += getElementTypeSize(op.getType()) * static_cast<int64_t>(op.getTileSize());
    });
    return mem;
}

struct ComputeMemoryConsumptionPass : public ascendc::impl::ComputeMemoryConsumptionBase<ComputeMemoryConsumptionPass> {
    void runOnOperation() override
    {
        ModuleOp moduleOp = getOperation();
        auto mem = calculateMemoryConsumption(moduleOp);
        Builder builder(moduleOp);
        SmallVector<NamedAttribute, 6> attrs;
        for (auto [key, value] : mem)
            attrs.emplace_back(builder.getStringAttr(key), builder.getI64IntegerAttr(value));
        moduleOp->setAttr(attr::memoryConsumed, builder.getDictionaryAttr(attrs));
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createComputeMemoryConsumptionPass()
{
    return std::make_unique<ComputeMemoryConsumptionPass>();
}
