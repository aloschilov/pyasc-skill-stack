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
#define GEN_PASS_DEF_ELIMINATECOMMONMASK
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

void eliminateCommonMask(ascvf::VFGroupOp groupOp)
{
    groupOp.walk([](Block* block) {
        Operation* firstUpdateMask = nullptr;
        for (auto& op : llvm::make_early_inc_range(*block)) {
            auto updateMask = dyn_cast<ascendc::UpdateMaskOp>(op);
            if (!updateMask)
                continue;
            if (firstUpdateMask) {
                updateMask.replaceAllUsesWith(firstUpdateMask->getResult(0));
                updateMask.erase();
            } else {
                firstUpdateMask = &op;
            }
        }
    });

    groupOp.walk([](Block* block) {
        llvm::DenseMap<ascendc::MaskPattern, ascendc::CreateMaskOp> createMaskMap;
        for (auto& op : llvm::make_early_inc_range(*block)) {
            auto createMask = dyn_cast<ascendc::CreateMaskOp>(op);
            if (!createMask)
                continue;
            auto it = createMaskMap.find(createMask.getMask());
            if (it == createMaskMap.end()) {
                createMaskMap[createMask.getMask()] = createMask;
            } else {
                createMask.replaceAllUsesWith(it->second.getResult());
                createMask.erase();
            }
        }
    });
}

struct EliminateCommonMaskPass : public ascvf::impl::EliminateCommonMaskBase<EliminateCommonMaskPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk([](ascvf::VFGroupOp fusedOp) { eliminateCommonMask(fusedOp); });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createEliminateCommonMaskPass()
{
    return std::make_unique<EliminateCommonMaskPass>();
}
