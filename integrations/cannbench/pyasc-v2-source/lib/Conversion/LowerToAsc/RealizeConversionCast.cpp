/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Conversion/LowerToAsc/Passes.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_REALIZECONVERSIONCAST
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

std::optional<std::pair<Value, Value>> unpack(Operation* op)
{
    if (op == nullptr || !isa<UnrealizedConversionCastOp>(op))
        return std::nullopt;
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        return std::nullopt;
    return std::make_pair(op->getOperand(0), op->getResult(0));
}

int64_t sizeInMemory(ascendc::LocalTensorType tensor)
{
    if (!tensor.hasStaticShape())
        return ShapedType::kDynamic;
    return tensor.getNumElements() * tensor.getElementType().getIntOrFloatBitWidth();
}

bool similarTensorTypes(Type t1, Type t2)
{
    auto tensor1 = dyn_cast<ascendc::LocalTensorType>(t1);
    auto tensor2 = dyn_cast<ascendc::LocalTensorType>(t2);
    return tensor1 && tensor2 && sizeInMemory(tensor1) == sizeInMemory(tensor2);
}

// Rewrites cast(cast(si -> x) -> ui) with cast(si -> ui)
// (x is generic shaped type; si and ui are either identical types or integer
// tensor types, which may or may not have different signedness)
struct RewriteReinterpretCast : public OpRewritePattern<UnrealizedConversionCastOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, PatternRewriter& rewriter) const override
    {
        if (auto unpacked0 = unpack(op)) {
            auto [src0, dst0] = *unpacked0;
            if (similarTensorTypes(dst0.getType(), src0.getType())) {
                rewriter.replaceOpWithNewOp<ascendc::LocalTensorReinterpretCastOp>(op, dst0.getType(), src0);
                return success();
            }
            if (auto unpacked1 = unpack(src0.getDefiningOp())) {
                auto [src1, dst1] = *unpacked1;
                if (similarTensorTypes(dst0.getType(), src1.getType())) {
                    rewriter.replaceOpWithNewOp<ascendc::LocalTensorReinterpretCastOp>(op, dst0.getType(), src1);
                    return success();
                }
            }
        }
        return failure();
    };
};

// Folds cast(cast(x -> y) -> x) and cast(cast(cast(x -> y) -> z) -> x) into x
struct FoldCastChain : public OpRewritePattern<UnrealizedConversionCastOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, PatternRewriter& rewriter) const override
    {
        if (auto unpacked0 = unpack(op)) {
            auto [src0, dst0] = *unpacked0;
            if (dst0.getType() == src0.getType()) {
                rewriter.replaceOp(op, src0);
                return success();
            }
            if (auto unpacked1 = unpack(src0.getDefiningOp())) {
                auto [src1, dst1] = *unpacked1;
                if (dst0.getType() == src1.getType()) {
                    rewriter.replaceOp(op, src1);
                    return success();
                }
                if (auto unpacked2 = unpack(src1.getDefiningOp())) {
                    auto [src2, dst2] = *unpacked2;
                    if (dst0.getType() == src2.getType()) {
                        rewriter.replaceOp(op, src2);
                        return success();
                    }
                }
            }
        }
        return failure();
    };
};

struct RealizeConversionCastPass : public asclower::impl::RealizeConversionCastBase<RealizeConversionCastPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.insert<RewriteReinterpretCast, FoldCastChain>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createRealizeConversionCastPass()
{
    return std::make_unique<RealizeConversionCastPass>();
}
