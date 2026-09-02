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
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTCROSSCORESYNC
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

constexpr uint8_t crossCoreMode = 4;

SmallVector<ascendc::Pipe> getGroupOutPipes(Operation* groupOp)
{
    // TODO: Reduce number of pipes requiring syncronization based on copy operations in group
    if (isa<ascendc::IfAICOp>(groupOp))
        return {ascendc::Pipe::PIPE_FIX, ascendc::Pipe::PIPE_MTE1};
    return {ascendc::Pipe::PIPE_MTE3};
}

struct InsertCrossCoreSyncPass : public ascendc::impl::InsertCrossCoreSyncBase<InsertCrossCoreSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        SmallVector<Operation*> groupOps;
        funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
            if (isa<ascendc::IfAICOp, ascendc::IfAIVOp>(op)) {
                groupOps.push_back(op);
                return WalkResult::skip();
            }
            return WalkResult::advance();
        });
        if (groupOps.size() < 2)
            return;
        OpBuilder builder(funcOp.getContext());
        ascir::ConstantOpBuilder consts(builder);
        for (size_t i = 1; i < groupOps.size(); ++i) {
            Operation* prev = groupOps[i - 1];
            Operation* next = groupOps[i];
            if (prev->getName() == next->getName())
                continue;
            Block& prevBlock = prev->getRegion(0).front();
            Operation* prevYield = prevBlock.getTerminator();
            builder.setInsertionPoint(prevYield);
            auto syncPipes = getGroupOutPipes(prev);
            for (auto setPipe : syncPipes) {
                builder.create<ascendc::CrossCoreSetFlagOp>(prev->getLoc(), consts.i32(0), crossCoreMode, setPipe);
                if (isa<ascendc::IfAICOp>(prev)) {
                    // TODO: Add only if MIX_1_2 is used
                    builder.create<ascendc::CrossCoreSetFlagOp>(prev->getLoc(), consts.i32(16), crossCoreMode, setPipe);
                }
            }
            Block& nextBlock = next->getRegion(0).front();
            builder.setInsertionPointToStart(&nextBlock);
            // TODO: Add second wait on Cube from second AIV if MIX_1_2 is used
            for (int i = 0; i < syncPipes.size(); i++)
                builder.create<ascendc::CrossCoreWaitFlagOp>(
                    next->getLoc(), consts.i32(0), crossCoreMode, ascendc::Pipe::PIPE_S);
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInsertCrossCoreSyncPass()
{
    return std::make_unique<InsertCrossCoreSyncPass>();
}
