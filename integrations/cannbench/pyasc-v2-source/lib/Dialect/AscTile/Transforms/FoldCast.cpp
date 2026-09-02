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
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_FOLDCAST
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;

namespace {

bool isCastSupported(Type srcType, Type dstType)
{
    if (srcType == dstType)
        return true;
    auto getIntegerWidth = [](Type type) -> std::optional<unsigned> {
        if (auto intType = dyn_cast<IntegerType>(type))
            return intType.getWidth();
        return std::nullopt;
    };
    auto srcIntWidth = getIntegerWidth(srcType);
    auto dstIntWidth = getIntegerWidth(dstType);
    if (srcIntWidth && dstIntWidth) {
        unsigned sw = *srcIntWidth;
        unsigned dw = *dstIntWidth;
        return (sw == 8 && dw == 16) || (sw == 8 && dw == 32) || (sw == 16 && dw == 32) || (sw == 32 && dw == 16) ||
               (sw == 32 && dw == 64) || (sw == 64 && dw == 32);
    }
    if (srcIntWidth && isa<FloatType>(dstType)) {
        unsigned sw = *srcIntWidth;
        if (isa<Float16Type>(dstType) && (sw == 8 || sw == 16 || sw == 32))
            return true;
        if (isa<Float32Type>(dstType) && (sw == 16 || sw == 32 || sw == 64))
            return true;
        return false;
    }
    if (isa<FloatType>(srcType) && dstIntWidth) {
        unsigned dw = *dstIntWidth;
        if (isa<BFloat16Type>(srcType) && dw == 32)
            return true;
        if (isa<Float16Type>(srcType) && (dw == 8 || dw == 16 || dw == 32))
            return true;
        if (isa<Float32Type>(srcType) && (dw == 16 || dw == 32 || dw == 64))
            return true;
        return false;
    }
    if (isa<FloatType>(srcType) && isa<FloatType>(dstType)) {
        return (isa<BFloat16Type>(srcType) && isa<Float16Type, Float32Type>(dstType)) ||
               (isa<Float16Type>(srcType) && isa<BFloat16Type, Float32Type>(dstType)) ||
               (isa<Float32Type>(srcType) && isa<BFloat16Type, Float16Type, Float32Type>(dstType));
    }

    return false;
}

bool isCastSupported(asctile::LocalTensorType srcType, asctile::LocalTensorType dstType)
{
    return isCastSupported(getElementTypeOrSelf(srcType), getElementTypeOrSelf(dstType));
}

struct FoldCastPattern : OpRewritePattern<asctile::CastOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::CastOp op, PatternRewriter& rewriter) const override
    {
        auto defOp = op.getIn().getDefiningOp<asctile::CastOp>();
        if (!defOp)
            return failure();
        if (op.getRoundMode() != defOp.getRoundMode())
            return failure();
        if (!isCastSupported(defOp.getIn().getType(), op.getType()))
            return failure();
        rewriter.replaceOpWithNewOp<asctile::CastOp>(op, op.getType(), defOp.getIn(), defOp.getRoundMode());
        return success();
    }
};

struct FoldCastPass : public asctile::impl::FoldCastBase<FoldCastPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.insert<FoldCastPattern>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed()) {
            signalPassFailure();
            return;
        }
        bool hasError = false;
        funcOp.walk([&hasError](asctile::CastOp castOp) {
            if (!isCastSupported(castOp.getIn().getType(), castOp.getType())) {
                castOp.emitOpError() << "from " << getElementTypeOrSelf(castOp.getIn().getType()) << " to "
                                     << getElementTypeOrSelf(castOp.getType()) << " is not supported";
                hasError = true;
            }
        });
        if (hasError)
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createFoldCastPass() { return std::make_unique<FoldCastPass>(); }
