/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Transforms/Passes.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_VERIFYTENSORLOCATION
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

using TL = TensorLocation;
using LocVector = SmallVector<TL, 8>;

bool badLoc(Operation* op, LocalTensorType type, StringRef name, ArrayRef<TL> allow)
{
    auto loc = type.getLoc();
    if (loc != TL::Auto && !llvm::is_contained(allow, loc)) {
        op->emitError() << name << " tensor location must be "
                        << llvm::join(llvm::map_range(allow, stringifyTensorLocation), ", ") << ", got "
                        << stringifyTensorLocation(loc);
        return true;
    }
    return false;
}

struct VerifyTensorLocationPass : public asctile::impl::VerifyTensorLocationBase<VerifyTensorLocationPass> {
    void runOnOperation() override;
};

void VerifyTensorLocationPass::runOnOperation()
{
    func::FuncOp funcOp = getOperation();
    funcOp.walk([this](Operation* op) {
        if (op->mightHaveTrait<OpTrait::IsTerminator>())
            return;
        auto isAutoTensor = [](Type type) {
            auto tensor = dyn_cast<LocalTensorType>(type);
            return tensor && tensor.getLoc() == TL::Auto;
        };
        if (llvm::any_of(op->getOperandTypes(), isAutoTensor)) {
            op->emitError(
                "Unable to resolve location for input tensor(s) of the current operation. "
                "Please provide explicit tensor locations in related creation or memory operations.");
            signalPassFailure();
        }
        if (llvm::any_of(op->getResultTypes(), isAutoTensor)) {
            op->emitError(
                "Unable to resolve location for result tensor(s) of the current operation. "
                "Please provide explicit tensor locations in related memory operations.");
            signalPassFailure();
        }
    });
    // Further checks are meaningless if tensor locations are not fully resolved
    if (getPassState().irAndPassFailed.getInt())
        return;
    funcOp.walk([this](LoadOp op) {
        if (badLoc(op, op.getType(), "result", {TL::UB, TL::L1, TL::L0A, TL::L0B, TL::BT}))
            signalPassFailure();
    });
    funcOp.walk([this](CopyOp op) {
        LocVector allowedSrcLocs{TL::L1, TL::L0C, TL::UB};
        LocVector allowedDstLocs;
        auto srcType = op.getBase().getType();
        auto srcLoc = srcType.getLoc();
        auto dstLoc = op.getType().getLoc();
        if (srcLoc == TL::L1)
            allowedDstLocs = {TL::L0A, TL::L0B, TL::BT};
        else if (srcLoc == TL::L0C)
            allowedDstLocs = {TL::L1, TL::UB};
        else if (srcLoc == TL::UB)
            allowedDstLocs = {TL::L1};
        if (op->hasAttrOfType<UnitAttr>(attr::locationCast) &&
            (!llvm::is_contained(allowedSrcLocs, srcLoc) || !llvm::is_contained(allowedDstLocs, dstLoc))) {
            StringRef srcLocStr = stringifyTensorLocation(srcLoc);
            StringRef dstLocStr = stringifyTensorLocation(dstLoc);
            auto diag =
                op.emitError() << "Direct data transfer from " << srcLocStr << " to " << dstLocStr
                               << " is not supported. "
                               << "Please call copy() with explicit locations to fulfill the requested data flow.";
            diag.attachNote(op.getBase().getLoc()) << "source tensor with " << srcLocStr << " location defined here:";
            diag.attachNote(op.getResult().getLoc())
                << "result tensor is required to have " << dstLocStr << " location:";
            signalPassFailure();
            return;
        }
        if (badLoc(op, srcType, "input", allowedSrcLocs) || badLoc(op, op.getType(), "result", allowedDstLocs))
            signalPassFailure();
    });
    funcOp.walk([this](AccumulatorOp op) {
        if (op.getBias() && badLoc(op, op.getBias().getType(), "bias", TL::BT))
            signalPassFailure();
    });
    funcOp.walk([this](MatmulOp op) {
        if (badLoc(op, op.getMatrixA().getType(), "input", TL::L0A) ||
            badLoc(op, op.getMatrixB().getType(), "other", TL::L0B) || badLoc(op, op.getType(), "result", TL::L0C) ||
            op.getBias() && badLoc(op, op.getBias().getType(), "bias", TL::BT))
            signalPassFailure();
    });
    funcOp.walk([this](MatmulAccOp op) {
        if (badLoc(op, op.getMatrixA().getType(), "input", TL::L0A) ||
            badLoc(op, op.getMatrixB().getType(), "other", TL::L0B) ||
            badLoc(op, op.getAcc().getType(), "acc", TL::L0C))
            signalPassFailure();
    });
    funcOp.walk([this](TransposeOp op) {
        LocVector allow = {TL::UB};
        if (op.getType().getRank() == 2)
            allow.append({TL::L1, TL::L0A, TL::L0B});
        if (badLoc(op, op.getOperand().getType(), "input", allow))
            signalPassFailure();
    });
}

} // namespace

std::unique_ptr<Pass> mlir::asctile::createVerifyTensorLocationPass()
{
    return std::make_unique<VerifyTensorLocationPass>();
}
