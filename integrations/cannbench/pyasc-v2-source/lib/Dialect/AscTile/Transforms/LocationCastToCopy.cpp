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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_LOCATIONCASTTOCOPY
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

using TL = TensorLocation;

struct CastToCopy : OpRewritePattern<tensor::CastOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::CastOp op, PatternRewriter& rewriter) const override
    {
        auto type = op.getType();
        auto base = op.getSource();
        if (type == base.getType())
            return failure();
        auto srcLoc = cast<LocalTensorType>(base.getType()).getLoc();
        auto dstLoc = cast<LocalTensorType>(type).getLoc();
        Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0L, 32U);
        SmallVector<Value, 2> offsets{static_cast<size_t>(type.getRank()), zero};
        rewriter.replaceOpWithNewOp<asctile::CopyOp>(op, type, base, offsets)
            ->setAttr(attr::locationCast, rewriter.getUnitAttr());
        return success();
    }
};

struct LocationCastToCopyPass : public asctile::impl::LocationCastToCopyBase<LocationCastToCopyPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<CastToCopy>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createLocationCastToCopyPass()
{
    return std::make_unique<LocationCastToCopyPass>();
}
