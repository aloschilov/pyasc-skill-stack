/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INPUTOUTPUTTENSOR
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

using TensorOp = ascendc::LocalTensorAutoOp;

template <typename ControlFlowOp>
void createDataCopyIfNeeded(ControlFlowOp op)
{
    for (auto& use : op->getUses()) {
        auto copyOp = dyn_cast<ascendc::DataCopyOp>(use.getOwner());
        if (!copyOp || !copyOp.isLocalToGlobal())
            return;
        OpBuilder builder(op);
        ascir::ConstantOpBuilder consts(builder);
        auto type = cast<ascendc::BaseTensorType>(use.get().getType());
        auto dst = builder.create<TensorOp>(op->getLoc(), type, /*input*/ false, /*output*/ true, ValueRange{});
        builder.setInsertionPointAfter(op);
        Value calCount = consts.i64(type.getNumElements());
        auto extraOp = builder.create<ascendc::DataCopyL2Op>(op->getLoc(), dst, use.get(), calCount);
        extraOp.setDirection(ascendc::TPosition::VECCALC, ascendc::TPosition::VECCALC);
        copyOp.setSrc(dst);
    }
}

TensorOp findTensorOrigin(Value tensor)
{
    auto* defOp = tensor.getDefiningOp();
    if (!defOp)
        return {};
    if (auto op = dyn_cast<TensorOp>(defOp))
        return op;
    if (auto op = dyn_cast<ascendc::LocalTensorReinterpretCastOp>(defOp))
        return findTensorOrigin(op.getIn());
    if (auto op = dyn_cast<ascendc::LocalTensorSubIndexOp>(defOp))
        return findTensorOrigin(op.getTensor());
    return {};
}

void setInOutTensors(func::FuncOp funcOp)
{
    funcOp.walk([](ascendc::DataCopyOp op) {
        auto src = findTensorOrigin(op.getSrc());
        if (src && op.isLocalToGlobal())
            src.setOutput(true);
        auto dst = findTensorOrigin(op.getDst());
        if (dst && op.isGlobalToLocal())
            dst.setInput(true);
        if (!op.isLocalToLocal())
            return;
        if (isa<ascendc::CopyToL0Op, ascendc::FixpipeOp>(*op)) {
            if (src)
                src.setOutput(true);
            if (dst)
                dst.setInput(true);
        }
    });
    funcOp.walk(createDataCopyIfNeeded<scf::ForOp>);
    funcOp.walk(createDataCopyIfNeeded<scf::IfOp>);
    funcOp.walk(createDataCopyIfNeeded<scf::WhileOp>);
}

void fixInOutTensor(func::FuncOp& funcOp)
{
    funcOp.walk([](TensorOp inTensor) {
        if (!inTensor.getInput() || inTensor.getOutput())
            return;
        auto loc = inTensor.getLoc();
        OpBuilder builder(inTensor);
        auto tensorType = inTensor.getResult().getType();
        inTensor.setOutput(false);
        for (auto& use : inTensor->getUses()) {
            auto* owner = use.getOwner();
            auto copyOp = dyn_cast<ascendc::DataCopyOp>(owner);
            if (!copyOp || !copyOp.isLocalToGlobal())
                return builder.setInsertionPoint(owner);
            ascir::ConstantOpBuilder consts(builder);
            Value calCount = consts.i64(tensorType.getNumElements());
            auto outTensor = builder.create<TensorOp>(loc, tensorType, /*input*/ false, /*output*/ true, ValueRange{});
            auto extraOp = builder.create<ascendc::DataCopyL2Op>(loc, outTensor, inTensor, calCount);
            extraOp.setDirection(ascendc::TPosition::VECCALC, ascendc::TPosition::VECCALC);
            owner->setOperand(use.getOperandNumber(), outTensor);
        }
    });
}

struct InputOutputTensorPass : public ascendc::impl::InputOutputTensorBase<InputOutputTensorPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        setInOutTensors(funcOp);
        fixInOutTensor(funcOp);
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInputOutputTensorPass() { return std::make_unique<InputOutputTensorPass>(); }
