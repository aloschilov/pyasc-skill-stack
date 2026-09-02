/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_COMPUTEREUSEGROUP
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

struct LoopInfo {
    int64_t unrollFactor = 0;
    int64_t startIndex = -1;
    int64_t groupId = -1;
    bool hasEpilogue = false;
};

std::optional<int64_t> getUnrollFactor(Operation* op)
{
    if (auto attr = op->getAttrOfType<IntegerAttr>(asctile::attr::unrollFactor))
        return attr.getValue().getSExtValue();
    return std::nullopt;
}

std::optional<int64_t> getUnrollIter(Operation* op)
{
    while (op && !isa<scf::ExecuteRegionOp>(op)) {
        if (auto attr = op->getAttrOfType<IntegerAttr>(asctile::attr::unrollIter))
            return attr.getValue().getSExtValue();
        op = op->getParentOp();
    }
    return std::nullopt;
}

std::optional<int64_t> getLoopId(Operation* op)
{
    if (!isa<scf::ExecuteRegionOp>(op)) {
        op = op->getParentOfType<scf::ExecuteRegionOp>();
    }
    if (!op)
        return std::nullopt;
    if (auto attr = op->getAttrOfType<IntegerAttr>(asctile::attr::unrolledLoop)) {
        return attr.getValue().getSExtValue();
    }
    return std::nullopt;
}

struct ComputeReuseGroupPass : public ascendc::impl::ComputeReuseGroupBase<ComputeReuseGroupPass> {
    void runOnOperation() override
    {
        auto op = getOperation();
        std::map<int64_t, LoopInfo> loops;
        std::map<scf::ExecuteRegionOp, int64_t> nestedLevel;
        op.walk<WalkOrder::PreOrder>([&](scf::ExecuteRegionOp exec) {
            if (auto parent = exec->getParentOfType<scf::ExecuteRegionOp>()) {
                nestedLevel[exec] = nestedLevel[parent] + 1;
            } else {
                nestedLevel[exec] = 0;
            }
            int64_t groupId = nestedLevel[exec];
            std::optional<int64_t> loopId = getLoopId(exec);
            if (loopId) {
                std::optional<int64_t> unrollFactor = getUnrollFactor(exec);
                assert(unrollFactor && "expected unroll_factor for ExecuteRegionOp");
                loops[loopId.value()].groupId = groupId;
                loops[loopId.value()].unrollFactor = unrollFactor.value();
            }
        });

        op.walk([&](Operation* op) {
            auto unrollIter = getUnrollIter(op);
            auto loopId = getLoopId(op);
            if (unrollIter && loopId && loops.count(loopId.value()) &&
                unrollIter.value() == loops[loopId.value()].unrollFactor) {
                auto& info = loops[loopId.value()];
                info.hasEpilogue = true;
            }
        });

        std::map<int64_t, int64_t> startIndexes;
        for (auto& [loopId, info] : loops) {
            int64_t& startIndex = startIndexes[info.groupId];
            info.startIndex = startIndex % info.unrollFactor;
            startIndex += info.hasEpilogue ? 1 : 0;
        }

        OpBuilder builder(op.getContext());
        op->walk([&](Operation* op) {
            auto unrollIter = getUnrollIter(op);
            auto loopId = getLoopId(op);
            if (loopId && unrollIter) {
                auto& info = loops[loopId.value()];
                int64_t newUnrollIter = (unrollIter.value() + info.startIndex) % info.unrollFactor;
                op->setAttr(ascendc::attr::reuseGroup, builder.getI64IntegerAttr(newUnrollIter));
                op->removeAttr(asctile::attr::unrollIter);
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createComputeReuseGroupPass() { return std::make_unique<ComputeReuseGroupPass>(); }
