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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_TRANSFORMSTOREFIXPIPE
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

struct TransformCopyOp : OpRewritePattern<asctile::CopyOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::CopyOp op, PatternRewriter& rewriter) const override
    {
        auto base = op.getBase();
        if (base.getType().getLoc() != TensorLocation::L0C)
            return failure();
        rewriter.replaceOpWithNewOp<asctile::CopyFixpipeOp>(op, op.getType(), base, op.getOffsets());
        return success();
    }
};

struct TransformStoreOp : OpRewritePattern<asctile::StoreOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::StoreOp op, PatternRewriter& rewriter) const override
    {
        auto value = op.getValue();
        if (value.getType().getLoc() != TensorLocation::L0C)
            return failure();
        rewriter.replaceOpWithNewOp<asctile::StoreFixpipeOp>(
            op, value, op.getBase(), op.getOffsets(), op.getRealShape());
        return success();
    }
};

struct TransformFixpipeReluOp : OpRewritePattern<asctile::StoreFixpipeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::StoreFixpipeOp op, PatternRewriter& rewriter) const override
    {
        auto reluOp = op.getValue().getDefiningOp<asctile::ReluOp>();
        if (!reluOp || reluOp.getType().getLoc() != TensorLocation::L0C)
            return failure();
        auto operand = reluOp.getOperand();
        rewriter.startOpModification(op);
        op.getValueMutable().assign(operand);
        op.setRelu(true);
        rewriter.finalizeOpModification(op);
        return success();
    }
};

struct TransformFixpipeCastOp : OpRewritePattern<asctile::StoreFixpipeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::StoreFixpipeOp op, PatternRewriter& rewriter) const override
    {
        auto castOp = op.getValue().getDefiningOp<asctile::CastOp>();
        if (!castOp || castOp.getType().getLoc() != TensorLocation::L0C)
            return failure();
        rewriter.startOpModification(op);
        op.getValueMutable().assign(castOp.getIn());
        op.setQuantize(true);
        rewriter.finalizeOpModification(op);
        return success();
    }
};

struct TransformCopyFixpipeReluOp : OpRewritePattern<asctile::CopyFixpipeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::CopyFixpipeOp op, PatternRewriter& rewriter) const override
    {
        auto reluOp = op.getBase().getDefiningOp<asctile::ReluOp>();
        if (!reluOp || reluOp.getType().getLoc() != TensorLocation::L0C)
            return failure();
        auto operand = reluOp.getOperand();
        rewriter.startOpModification(op);
        op.getBaseMutable().assign(operand);
        op.setRelu(true);
        rewriter.finalizeOpModification(op);
        return success();
    }
};

struct TransformCopyFixpipeCastOp : OpRewritePattern<asctile::CopyFixpipeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::CopyFixpipeOp op, PatternRewriter& rewriter) const override
    {
        auto castOp = op.getBase().getDefiningOp<asctile::CastOp>();
        if (!castOp || castOp.getType().getLoc() != TensorLocation::L0C)
            return failure();
        rewriter.startOpModification(op);
        op.getBaseMutable().assign(castOp.getIn());
        op.setQuantize(true);
        rewriter.finalizeOpModification(op);
        return success();
    }
};

struct TransformStoreFixpipePass : public asctile::impl::TransformStoreFixpipeBase<TransformStoreFixpipePass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.insert<
            TransformCopyOp, TransformStoreOp, TransformFixpipeReluOp, TransformFixpipeCastOp,
            TransformCopyFixpipeReluOp, TransformCopyFixpipeCastOp>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed())
            signalPassFailure();
    }
};
} // namespace

std::unique_ptr<Pass> mlir::asctile::createTransformStoreFixpipePass()
{
    return std::make_unique<TransformStoreFixpipePass>();
}
