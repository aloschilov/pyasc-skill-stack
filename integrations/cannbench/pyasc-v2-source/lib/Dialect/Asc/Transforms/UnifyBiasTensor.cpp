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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_UNIFYBIASTENSOR
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {
struct UnifyBiasTensorPass : public ascendc::impl::UnifyBiasTensorBase<UnifyBiasTensorPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        SmallVector<LocalTensorV3Op> biasTensors;
        funcOp.walk<WalkOrder::PreOrder>([&](LocalTensorV3Op op) {
            if (op.getPos() == TPosition::C2) {
                biasTensors.push_back(op);
            }
        });
        if (biasTensors.empty())
            return;
        auto firstOp = biasTensors[0];
        for (size_t i = 1; i < biasTensors.size(); ++i) {
            biasTensors[i].getResult().replaceAllUsesWith(firstOp.getResult());
            biasTensors[i].erase();
        }
        firstOp.setAddr(0);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createUnifyBiasTensorPass() { return std::make_unique<UnifyBiasTensorPass>(); }
