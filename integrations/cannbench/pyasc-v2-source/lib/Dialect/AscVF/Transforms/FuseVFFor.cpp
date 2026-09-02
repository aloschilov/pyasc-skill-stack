/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/AscVF/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_FUSEVFFOR
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

void merge(ascvf::VFForOp firstLoop, ascvf::VFForOp secondLoop)
{
    OpBuilder builder(firstLoop.getBody()->getTerminator());
    SmallVector<Operation*> opList;
    IRMapping mapper;
    mapper.map(secondLoop.getInductionVar(), firstLoop.getInductionVar());
    for (auto& op : secondLoop.getBody()->without_terminator()) {
        builder.clone(op, mapper);
    }
    secondLoop.erase();
}

bool canMerge(ascvf::VFForOp firstLoop, ascvf::VFForOp secondLoop)
{
    Value val1 = firstLoop.getUpperBound();
    Value val2 = secondLoop.getUpperBound();
    return val1 == val2 || getConstantIntValue(val1) == getConstantIntValue(val2);
}

void fuseLoops(ascvf::VecScopeOp vecScopeOp)
{
    auto& ops = vecScopeOp.getBody()->getOperations();
    if (ops.size() < 2)
        return;
    auto it = ops.begin();
    while (std::next(it) != ops.end()) {
        bool merged = false;
        if (auto curLoop = dyn_cast<ascvf::VFForOp>(*it)) {
            auto nextIt = std::next(it);
            if (auto nextLoop = dyn_cast<ascvf::VFForOp>(*nextIt)) {
                if (canMerge(curLoop, nextLoop)) {
                    merge(curLoop, nextLoop);
                    merged = true;
                }
            }
        }
        if (!merged)
            ++it;
    }
}

struct FuseVFForPass : public ascvf::impl::FuseVFForBase<FuseVFForPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk([](ascvf::VecScopeOp vecScope) { fuseLoops(vecScope); });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createFuseVFForPass() { return std::make_unique<FuseVFForPass>(); }
