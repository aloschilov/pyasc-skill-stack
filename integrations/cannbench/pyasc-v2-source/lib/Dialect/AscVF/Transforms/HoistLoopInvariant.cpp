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
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/AscVF/Transforms/Passes.h"
#include "ascir/Dialect/AscVF/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_HOISTLOOPINVARIANT
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

void addUsers(Value value, llvm::DenseSet<Operation*>& deps)
{
    for (auto* use : value.getUsers()) {
        deps.insert(use);
    }
}

void addUsers(Operation* op, llvm::DenseSet<Operation*>& deps)
{
    for (auto dst : ascvf::getDst(op)) {
        addUsers(dst, deps);
    }
}

void hoistLoopInvariant(ascvf::VFForOp forOp)
{
    llvm::DenseSet<Operation*> deps;
    addUsers(forOp.getInductionVar(), deps);
    for (auto& op : llvm::make_early_inc_range(forOp.getBody()->without_terminator())) {
        if (deps.contains(&op) || isa<MemoryEffectOpInterface>(&op)) {
            addUsers(&op, deps);
        } else {
            op.moveBefore(forOp);
        }
    }
}

struct HoistLoopInvariantPass : public ascvf::impl::HoistLoopInvariantBase<HoistLoopInvariantPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk<WalkOrder::PostOrder>(hoistLoopInvariant);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createHoistLoopInvariantPass() { return std::make_unique<HoistLoopInvariantPass>(); }
