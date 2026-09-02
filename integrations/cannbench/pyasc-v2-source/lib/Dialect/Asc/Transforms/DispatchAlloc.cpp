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
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "ascendc-dispatch-alloc"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_DISPATCHALLOC
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

struct DispatchAllocPass : public ascendc::impl::DispatchAllocBase<DispatchAllocPass> {
    void runOnOperation() override
    {
        func::FuncOp op = getOperation();
        auto mod = ascendc::getModule(op);
        bool archC310 = ascendc::isTargetArchC310(mod);
        std::optional<bool> staticAlloc(std::nullopt);
        if (auto attr = mod->getAttrOfType<BoolAttr>(ascendc::attr::staticAlloc)) {
            staticAlloc = attr.getValue();
            LLVM_DEBUG(llvm::dbgs() << "'" << ascendc::attr::staticAlloc << "' is set to " << *staticAlloc << '\n');
        } else {
            LLVM_DEBUG(llvm::dbgs() << "'" << ascendc::attr::staticAlloc << "' is not set\n");
            if (archC310) {
                // Known Ascend C issue: Broadcast allocates through TPipe internally
                bool hasIncompatibleOps =
                    op.walk([](ascendc::BroadcastOp) { return WalkResult::interrupt(); }).wasInterrupted();
                staticAlloc = !hasIncompatibleOps;
            } else {
                staticAlloc = false;
            }
        }
        assert(staticAlloc.has_value());
        OpPassManager pm(func::FuncOp::getOperationName(), OpPassManager::Nesting::Explicit);
        if (*staticAlloc) {
            LLVM_DEBUG(llvm::dbgs() << "static tensor allocation is selected\n");
            pm.addPass(ascendc::createAllocateTensorPass());
        } else {
            LLVM_DEBUG(llvm::dbgs() << "TPipe-backed tensor allocation is selected (alwaysBuf=" << archC310 << ")\n");
            pm.addPass(ascendc::createMaterializeTensorPass(archC310));
        }
        if (runPipeline(pm, op).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createDispatchAllocPass() { return std::make_unique<DispatchAllocPass>(); }
