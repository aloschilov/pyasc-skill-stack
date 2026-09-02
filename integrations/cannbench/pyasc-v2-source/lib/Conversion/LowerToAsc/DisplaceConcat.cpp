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
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_DISPLACECONCAT
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

struct ConcatToNoop : public OpRewritePattern<tensor::ConcatOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::ConcatOp op, PatternRewriter& rewriter) const override
    {
        if (op.getDim() != 0)
            return failure();
        SmallVector<ascendc::LocalTensorAutoOp> allocs(llvm::map_range(op.getOperands(), [](Value tensor) {
            auto castOp = tensor.getDefiningOp<UnrealizedConversionCastOp>();
            if (!castOp || castOp.getNumOperands() != 1 || castOp->getNumResults() != 1)
                return ascendc::LocalTensorAutoOp{};
            return castOp.getOperand(0).getDefiningOp<ascendc::LocalTensorAutoOp>();
        }));
        if (!llvm::all_of(allocs, [](auto op) -> bool { return op; }))
            return failure();
        rewriter.setInsertionPointToStart(&op->getParentOfType<func::FuncOp>().getBody().front());
        auto tileType = op.getType();
        auto resultType = ascendc::LocalTensorType::get(tileType.getShape(), tileType.getElementType());
        Value result = rewriter.create<ascendc::LocalTensorAutoOp>(op.getLoc(), resultType);
        int64_t offset = 0;
        ascir::ConstantOpBuilder consts(rewriter);
        for (auto alloc : allocs) {
            auto tensor = rewriter.replaceOpWithNewOp<ascendc::LocalTensorSubIndexOp>(
                alloc, alloc.getType(), result, consts.i64(offset));
            offset += tensor.getType().getNumElements();
            rewriter.setInsertionPointAfter(tensor);
        }
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, tileType, result);
        return success();
    }
};

struct ConvertConcat : public ConvertOp<tensor::ConcatOp> {
    using ConvertOp::calCount;
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(tensor::ConcatOp op, ConvertRewriter& rewriter) const override
    {
        if (op.getDim() != 0)
            return failure();
        auto loc = op.getLoc();
        ascir::ConstantOpBuilder consts(rewriter);
        Value dst = createTensorOp(rewriter, loc, op.getType());
        int64_t offset = 0;
        for (auto opnd : op.getOperands()) {
            auto tensorType = typeConverter->convertType(opnd.getType());
            auto tensor = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, tensorType, dst, consts.i64(offset));
            Value src = rewriter.getRemappedValue(opnd);
            auto copyOp = rewriter.create<ascendc::DataCopyL2Op>(loc, tensor, src, consts.i64(calCount(src)));
            copyOp.setDirection(ascendc::TPosition::VECCALC, ascendc::TPosition::VECCALC);
            offset += tensor.getType().getNumElements();
        }
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct DisplaceConcatPass : public asclower::impl::DisplaceConcatBase<DisplaceConcatPass> {
    void tryConcatToNoop()
    {
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<ConcatToNoop>(context);
        if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)).failed())
            signalPassFailure();
    }

    void convertConcatFallback()
    {
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addIllegalOp<tensor::ConcatOp>();
        target.addLegalDialect<arith::ArithDialect, ascendc::AscendCDialect>();
        RewritePatternSet patterns(context);
        patterns.insert<ConvertConcat>(converter, context);
        if (applyPartialConversion(getOperation(), target, std::move(patterns)).failed())
            signalPassFailure();
    }

    void runOnOperation() override
    {
        tryConcatToNoop();
        convertConcatFallback();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createDisplaceConcatPass() { return std::make_unique<DisplaceConcatPass>(); }
