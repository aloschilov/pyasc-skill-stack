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
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERARITH
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

struct ConvertSplatConstant : ConvertOp<arith::ConstantOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(arith::ConstantOp op, ConvertRewriter& rewriter) const override
    {
        if (!isa_and_present<SplatElementsAttr>(op.getValue()))
            return failure();
        ascir::ConstantOpBuilder consts(rewriter);
        auto dense = dyn_cast<DenseElementsAttr>(op.getValue());
        Value scalar = rewriter.create<arith::ConstantOp>(op.getLoc(), dense.getSplatValue<TypedAttr>());
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        auto ifOp = rewriter.create<ascendc::IfAIVOp>(loc, TypeRange{}, ValueRange{});
        rewriter.setInsertionPointToStart(&ifOp.getRegion().emplaceBlock());
        rewriter.create<ascendc::DuplicateL2Op>(loc, dst, scalar, consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        rewriter.create<ascendc::YieldOp>(loc);
        return success();
    }
};

struct ConvertDenseConstant : ConvertOp<arith::ConstantOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(arith::ConstantOp op, ConvertRewriter& rewriter) const override
    {
        auto dense = dyn_cast<DenseElementsAttr>(op.getValue());
        if (!dense || dense.isSplat())
            return failure();
        ascir::ConstantOpBuilder consts(rewriter);
        Value dst = createTensorOp(rewriter, op.getLoc(), op.getType());
        for (auto [i, value] : llvm::enumerate(dense.getValues<TypedAttr>())) {
            Location uloc = rewriter.getUnknownLoc();
            Value cst = rewriter.create<arith::ConstantOp>(uloc, value);
            rewriter.create<ascendc::LocalTensorSetValueOp>(uloc, dst, consts.i32(static_cast<int32_t>(i)), cst);
        }
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertBitcast : public ConvertOp<arith::BitcastOp> {
    using ConvertOp::ConvertOp;

    LogicalResult matchAndRewrite(arith::BitcastOp op, ConvertRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<ascendc::LocalTensorReinterpretCastOp>(
            op, typeConverter->convertType(op.getType()), rewriter.getRemappedValue(op.getIn()));
        return success();
    }
};

struct ConvertNegF : public ConvertOp<arith::NegFOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(arith::NegFOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        auto src = rewriter.getRemappedValue(op.getOperand());
        auto floatTy = cast<FloatType>(cast<ShapedType>(op.getType()).getElementType());
        auto cm1 =
            rewriter.create<arith::ConstantFloatOp>(loc, llvm::APFloat(floatTy.getFloatSemantics(), "-1"), floatTy);
        rewriter.create<ascendc::MulsL2Op>(loc, dst, src, cm1, consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertSelect : ConvertOp<arith::SelectOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(arith::SelectOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto dst = createTensorOp(rewriter, loc, op.getType());
        auto sel = rewriter.getRemappedValue(op.getCondition());
        I1ReplacementType replType(op.getContext());
        sel = createReCastOp(rewriter, loc, sel, cast<ShapedType>(sel.getType()).getShape(), replType.uiType);
        auto src0 = rewriter.getRemappedValue(op.getTrueValue());
        auto src1 = rewriter.getRemappedValue(op.getFalseValue());
        auto zero = ascir::ConstantOpBuilder(rewriter).i64(0);
        rewriter.create<ascendc::SelectL0Op>(
            loc, dst, sel, src0, src1, ascendc::SELMODE::VSEL_TENSOR_TENSOR_MODE, zero, zero,
            rewriter.create<ascendc::ConstructOp>(loc, rewriter.getType<ascendc::BinaryRepeatParamsType>()));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct LowerArithPass : public asclower::impl::LowerArithBase<LowerArithPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addDynamicallyLegalOp<
            //
            arith::ConstantOp, arith::BitcastOp, arith::NegFOp, arith::SelectOp
            //
            >([&converter](Operation* op) { return converter.isLegal(op); });
        target.addLegalDialect<ascendc::AscendCDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertSplatConstant, ConvertDenseConstant, ConvertBitcast, ConvertNegF, ConvertSelect
            //
            >(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerArithPass() { return std::make_unique<LowerArithPass>(); }
