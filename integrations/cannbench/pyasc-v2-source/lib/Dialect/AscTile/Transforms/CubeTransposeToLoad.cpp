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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_CUBETRANSPOSETOLOAD
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

constexpr const char* const fusibleAttr = "asctile.fusible";

void markTransposeCopy(asctile::TransposeOp op)
{
    auto tileLoc = op.getType().getLoc();
    if (tileLoc != TensorLocation::L0A && tileLoc != TensorLocation::L0B)
        return;
    auto copyOp = op.getOperand().getDefiningOp<asctile::CopyOp>();
    if (!copyOp || !copyOp->hasOneUse())
        return;
    if (copyOp.getType().getShape().size() != 2)
        return;
    auto* loadOp = copyOp.getBase().getDefiningOp();
    if (!loadOp)
        return;
    if (!loadOp->hasOneUse()) {
        for (auto* useLoad : loadOp->getUsers()) {
            auto copyFromLoad = dyn_cast<asctile::CopyOp>(useLoad);
            if (!copyFromLoad || !copyFromLoad->hasOneUse())
                return;
            if (!llvm::all_of(
                    copyFromLoad->getUsers(), [](auto* useCopy) { return isa<asctile::TransposeOp>(useCopy); }))
                return;
        }
    }
    op->setAttr(fusibleAttr, UnitAttr::get(op.getContext()));
}

void markTransposeLoad(asctile::TransposeOp op)
{
    auto tileLoc = op.getType().getLoc();
    if (tileLoc != TensorLocation::L1)
        return;
    auto loadOp = op.getOperand().getDefiningOp<asctile::LoadOp>();
    if (!loadOp || !loadOp->hasOneUse())
        return;
    if (loadOp.getType().getShape().size() != 2)
        return;
    if (!op->hasOneUse())
        return;
    auto copyOp = dyn_cast<asctile::CopyOp>(*op->user_begin());
    if (!copyOp)
        return;
    auto copyLoc = copyOp.getType().getLoc();
    if (copyLoc != TensorLocation::L0A && copyLoc != TensorLocation::L0B)
        return;
    op->setAttr(fusibleAttr, UnitAttr::get(op.getContext()));
}

struct CubeTransposeToLoad : OpRewritePattern<asctile::TransposeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::TransposeOp op, PatternRewriter& rewriter) const override
    {
        if (!op->hasAttrOfType<UnitAttr>(fusibleAttr))
            return failure();
        auto opType = op.getType();
        auto tileLoc = opType.getLoc();
        if (tileLoc == TensorLocation::L0A || tileLoc == TensorLocation::L0B) {
            auto copyOp = op.getOperand().getDefiningOp<asctile::CopyOp>();
            const auto* attr =
                tileLoc == TensorLocation::L0A ? asctile::attr::transposeAL0 : asctile::attr::transposeBL0;
            auto* loadOp = copyOp.getBase().getDefiningOp();
            rewriter.startOpModification(loadOp);
            loadOp->setAttr(attr, rewriter.getUnitAttr());
            rewriter.finalizeOpModification(loadOp);
            auto copyOpType = copyOp.getType();
            auto shape = copyOpType.getShape();
            Type newType = LocalTensorType::get({shape[1], shape[0]}, opType.getElementType(), copyOpType.getLoc());
            auto newCopyOp =
                rewriter.create<asctile::CopyOp>(op.getLoc(), newType, copyOp.getBase(), copyOp.getOffsets());
            rewriter.replaceOp(copyOp, newCopyOp);
            rewriter.startOpModification(newCopyOp);
            newCopyOp->setAttr(attr, rewriter.getUnitAttr());
            rewriter.finalizeOpModification(newCopyOp);
            rewriter.replaceOp(op, newCopyOp);
        } else if (tileLoc == TensorLocation::L1) {
            auto loadOp = op.getOperand().getDefiningOp<asctile::LoadOp>();
            auto copyOp = dyn_cast<asctile::CopyOp>(*op->user_begin());
            if (!copyOp)
                return failure();
            auto copyLoc = copyOp.getType().getLoc();
            const auto* attr =
                copyLoc == TensorLocation::L0A ? asctile::attr::transposeAL1 : asctile::attr::transposeBL1;
            auto loadType = loadOp.getType();
            auto shape = loadType.getShape();
            auto newLoadType = LocalTensorType::get({shape[1], shape[0]}, opType.getElementType(), loadType.getLoc());
            auto newLoadOp = rewriter.create<asctile::LoadOp>(
                op.getLoc(), newLoadType, loadOp.getBase(), loadOp.getOffsets(), loadOp.getPadValue(),
                loadOp.getRealShape());
            newLoadOp->setAttrs(loadOp->getAttrs());
            newLoadOp->setAttr(attr, rewriter.getUnitAttr());
            rewriter.startOpModification(copyOp);
            copyOp->setAttr(attr, rewriter.getUnitAttr());
            rewriter.finalizeOpModification(copyOp);
            rewriter.replaceOp(loadOp, newLoadOp);
            rewriter.replaceOp(op, newLoadOp);
        }
        return success();
    }
};

struct CubeTransposeToLoadPass : public asctile::impl::CubeTransposeToLoadBase<CubeTransposeToLoadPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp->walk(markTransposeCopy);
        funcOp->walk(markTransposeLoad);
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.insert<CubeTransposeToLoad>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed()) {
            signalPassFailure();
            return;
        }
        funcOp->walk([this](asctile::TransposeOp op) {
            auto location = op.getType().getLoc();
            if (location == TensorLocation::L0A || location == TensorLocation::L0B) {
                op.emitOpError() << "not supported on cube";
                signalPassFailure();
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createCubeTransposeToLoadPass()
{
    return std::make_unique<CubeTransposeToLoadPass>();
}
