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
#define GEN_PASS_DEF_SPLITCUBELOAD
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

struct ConvertLoadGMToL0 : OpRewritePattern<asctile::LoadOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::LoadOp op, PatternRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto tileLoc = opType.getLoc();
        if (tileLoc != TensorLocation::L0A && tileLoc != TensorLocation::L0B && tileLoc != TensorLocation::BT) {
            return failure();
        }
        auto l1Type = LocalTensorType::get(opType.getShape(), opType.getElementType(), TensorLocation::L1);
        Value l1Tile = rewriter.create<asctile::LoadOp>(
            op.getLoc(), l1Type, op.getBase(), op.getOffsets(), op.getPadValue(), op.getRealShape());
        if (tileLoc == TensorLocation::BT)
            l1Tile.getDefiningOp()->setAttr(attr::isBias, rewriter.getUnitAttr());
        Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32IntegerAttr(0));
        SmallVector<Value> offsets(opType.getShape().size(), zero);
        rewriter.replaceOpWithNewOp<asctile::CopyOp>(op, op.getType(), l1Tile, offsets);
        return success();
    }
};

struct MarkTileOperandInMmad : OpRewritePattern<asctile::LoadOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::LoadOp op, PatternRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto tileLoc = opType.getLoc();
        if (opType.getLoc() != TensorLocation::L1 || op->use_empty() || op->hasAttr(attr::isMatrixA)) {
            return failure();
        }

        std::optional<bool> isTensorA;
        for (auto& use : op->getUses()) {
            auto* user = use.getOwner();
            asctile::CopyOp copyOp;
            if (auto directCopy = dyn_cast<asctile::CopyOp>(user)) {
                copyOp = directCopy;
            } else if (auto transposeOp = dyn_cast<asctile::TransposeOp>(user)) {
                if (!transposeOp->hasOneUse()) {
                    op.emitError() << "TransposeOp after L1 load must have exactly one use.";
                    return failure();
                }
                copyOp = dyn_cast<asctile::CopyOp>(*transposeOp->user_begin());
                if (!copyOp) {
                    op.emitError() << "TransposeOp after L1 load must be used by CopyOp.";
                    return failure();
                }
            } else {
                op.emitError() << "L1 tensor is expected to be used for copy or transpose operations only.";
                return failure();
            }
            assert(copyOp && "copyOp must be initialized");
            auto l0TileLoc = copyOp.getType().getLoc();
            if (l0TileLoc != TensorLocation::L0A && l0TileLoc != TensorLocation::L0B &&
                l0TileLoc != TensorLocation::BT) {
                auto diag = op.emitError() << "L1 tensor copy to L0A/L0B/BT location is expected only.";
                diag.attachNote(copyOp->getLoc()) << "used here unexpectedly";
                return failure();
            }
            if (!isTensorA.has_value()) {
                isTensorA = l0TileLoc == TensorLocation::L0A;
            } else if (isTensorA.value() != (l0TileLoc == TensorLocation::L0A)) {
                auto diag = op.emitError()
                            << "The same L1 tensor should be copied only to tiles in same L0A/L0B location.";
                diag.attachNote(copyOp->getLoc()) << "copied here unexpectedly";
                return failure();
            }
        }
        if (isTensorA.value()) {
            rewriter.modifyOpInPlace(op, [&]() { op->setAttr(attr::isMatrixA, rewriter.getUnitAttr()); });
            return success();
        }

        return failure();
    }
};

struct SplitCubeLoadPass : public asctile::impl::SplitCubeLoadBase<SplitCubeLoadPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.insert<ConvertLoadGMToL0, MarkTileOperandInMmad>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createSplitCubeLoadPass() { return std::make_unique<SplitCubeLoadPass>(); }
