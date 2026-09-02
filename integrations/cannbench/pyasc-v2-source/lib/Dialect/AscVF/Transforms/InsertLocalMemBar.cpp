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

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_INSERTLOCALMEMBAR
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

struct InsertLocalMemBarPass : public ascvf::impl::InsertLocalMemBarBase<InsertLocalMemBarPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk([&](ascvf::VFGroupOp vfGroupOp) {
            SmallVector<ascvf::VFForOp> loops;
            vfGroupOp.walk([&](ascvf::VFForOp forOp) { loops.push_back(forOp); });
            if (loops.empty())
                return;
            OpBuilder builder(funcOp.getContext());
            for (auto forOp : ArrayRef(loops).drop_back()) {
                builder.setInsertionPointAfter(forOp);
                builder.create<ascendc::LocalMemBarOp>(
                    builder.getUnknownLoc(), ascendc::MemType::VEC_STORE, ascendc::MemType::VEC_LOAD);
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createInsertLocalMemBarPass() { return std::make_unique<InsertLocalMemBarPass>(); }
