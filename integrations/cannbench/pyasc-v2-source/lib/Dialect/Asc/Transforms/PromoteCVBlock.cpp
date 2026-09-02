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
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_PROMOTECVBLOCK
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

void runOnFunction(func::FuncOp func)
{
    SmallVector<Operation*> ifOps;
    StringRef kernelType;
    func.walk([&](IfAICOp op) {
        ifOps.push_back(op);
        kernelType = attr::kernelCube;
    });
    auto walk = func.walk([&](IfAIVOp op) {
        if (kernelType == attr::kernelCube) {
            kernelType = attr::kernelMixed;
            return WalkResult::interrupt();
        }
        ifOps.push_back(op);
        kernelType = attr::kernelVector;
        return WalkResult::advance();
    });
    if (ifOps.empty())
        return;
    getModule(func)->setAttr(attr::kernelType, StringAttr::get(func.getContext(), kernelType));
    if (walk.wasInterrupted())
        return;
    for (auto* ifOp : ifOps) {
        auto& block = *ifOp->getRegion(0).begin();
        auto yields = block.getTerminator()->getOperands();
        for (auto& op : llvm::make_early_inc_range(block.without_terminator()))
            op.moveBefore(ifOp);
        ifOp->replaceAllUsesWith(yields);
        ifOp->erase();
    }
}

struct PromoteCVBlockPass : public ascendc::impl::PromoteCVBlockBase<PromoteCVBlockPass> {
    void runOnOperation() override { getOperation().walk(runOnFunction); }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createPromoteCVBlockPass() { return std::make_unique<PromoteCVBlockPass>(); }
