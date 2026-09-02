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
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_FIXUPMMADACCPARAMS
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

class FixupMmadAccParamsPass : public ascendc::impl::FixupMmadAccParamsBase<FixupMmadAccParamsPass> {
    std::optional<StringRef> getFieldWithTrueValue(emitasc::InitStructOp mmadParamsOp)
    {
        for (const auto& field : {"cmatrixInitVal", "cmatrixSource"}) {
            if (matchPattern(mmadParamsOp.getField(field), m_One()))
                return field;
        }
        return std::nullopt;
    }

    void runOnOperation() override
    {
        llvm::DenseMap<Operation*, SmallVector<ascendc::MmadOp>> mmadOpsOfDst;

        func::FuncOp funcOp = getOperation();
        funcOp.walk([&mmadOpsOfDst](ascendc::MmadOp op) {
            auto mmadParamsOp = op.getMmadParams().getDefiningOp<emitasc::InitStructOp>();
            if (mmadParamsOp && mmadParamsOp.hasField("cmatrixInitVal")) {
                mmadOpsOfDst[op.getDst().getDefiningOp()].push_back(op);
            }
        });
        if (mmadOpsOfDst.empty()) {
            return;
        }

        OpBuilder builder(funcOp.getFunctionBody());
        ascir::ConstantOpBuilder consts(builder);
        Value zero = consts.index(0);
        Value falseVal = consts.i1(false);
        for (const auto& [dst, mmadOps] : mmadOpsOfDst) {
            builder.setInsertionPointAfter(dst);
            Value initVar = builder.create<emitasc::VariableOp>(dst->getLoc(), builder.getBoolAttr(true));
            for (ascendc::MmadOp mmadOp : mmadOps) {
                auto mmadParamsOp = mmadOp.getMmadParams().getDefiningOp<emitasc::InitStructOp>();
                builder.setInsertionPoint(mmadParamsOp);
                auto fieldName = getFieldWithTrueValue(mmadParamsOp);
                if (!fieldName.has_value())
                    continue;
                Value loadInitVal = builder.create<memref::LoadOp>(mmadParamsOp.getLoc(), initVar, zero);
                mmadParamsOp.setField(fieldName.value(), loadInitVal);
                builder.setInsertionPointAfter(mmadOp);
                builder.create<memref::StoreOp>(mmadParamsOp.getLoc(), falseVal, initVar, zero);
            }
        }
    }
};
} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createFixupMmadAccParamsPass() { return std::make_unique<FixupMmadAccParamsPass>(); }
} // namespace ascendc
} // namespace mlir
