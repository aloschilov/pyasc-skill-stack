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
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_DISPATCHHOIST
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

struct DispatchHoistPass : public ascvf::impl::DispatchHoistBase<DispatchHoistPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        OpPassManager pm(func::FuncOp::getOperationName(), OpPassManager::Nesting::Explicit);
        for (int i = 0; i < 2; ++i) {
            pm.addPass(ascvf::createEliminateDataTransferPass());
            pm.addPass(ascvf::createHoistLoopInvariantPass());
        }
        if (runPipeline(pm, funcOp).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createDispatchHoistPass() { return std::make_unique<DispatchHoistPass>(); }
