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
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTBIASBUFIDSYNC
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {
struct InsertBiasBufIdSyncPass : public ascendc::impl::InsertBiasBufIdSyncBase<InsertBiasBufIdSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        int64_t bufId = -1;
        funcOp.walk([&](ascendc::LocalTensorV3Op tensorOp) {
            if (tensorOp.getPos() != ascendc::TPosition::C2)
                return;
            auto bufIdAttr = tensorOp->getAttrOfType<IntegerAttr>(ascendc::attr::bufId);
            if (!bufIdAttr)
                return;
            bufId = bufIdAttr.getInt();
        });
        if (bufId < 0)
            return;
        funcOp.walk([&](ascendc::MmadOp mmadOp) {
            auto mmadParams = mmadOp.getMmadParams();
            auto* paramsDefOp = mmadParams.getDefiningOp();
            if (!paramsDefOp)
                return;
            auto initStructOp = dyn_cast<emitasc::InitStructOp>(paramsDefOp);
            if (!initStructOp || !initStructOp.hasField("cmatrixSource"))
                return;
            OpBuilder builder(mmadOp);
            builder.create<ascendc::GetBufOp>(mmadOp->getLoc(), ascendc::Pipe::PIPE_M, bufId, false);
            builder.setInsertionPointAfter(mmadOp);
            builder.create<ascendc::RlsBufOp>(mmadOp->getLoc(), ascendc::Pipe::PIPE_M, bufId, false);
        });
    }
};
} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInsertBiasBufIdSyncPass()
{
    return std::make_unique<InsertBiasBufIdSyncPass>();
}
