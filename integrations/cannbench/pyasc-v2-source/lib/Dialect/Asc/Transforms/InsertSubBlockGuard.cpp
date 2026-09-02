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

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTSUBBLOCKGUARD
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

class InsertSubBlockGuardPass : public ascendc::impl::InsertSubBlockGuardBase<InsertSubBlockGuardPass> {
public:
    void runOnOperation() override
    {
        auto funcOp = getOperation();
        auto kernelTypeAttr = ascendc::getModule(funcOp)->getAttrOfType<StringAttr>(ascendc::attr::kernelType);
        if (!kernelTypeAttr || kernelTypeAttr.getValue() != ascendc::attr::kernelMixed)
            return;
        // Unused Vector sub block must execute SyncAll to prevent deadlock.
        if (funcOp.walk([](ascendc::SyncAllHardOp) { return WalkResult::interrupt(); }).wasInterrupted())
            return;
        auto builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
        builder.create<emitc::VerbatimOp>(funcOp.getLoc(), "if (AscendC::GetSubBlockIdx() != 0) return;");
    }
};

} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createInsertSubBlockGuardPass() { return std::make_unique<InsertSubBlockGuardPass>(); }
} // namespace ascendc
} // namespace mlir
