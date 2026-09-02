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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERMATH
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

template <typename MathOp, typename LibraryOp>
struct ConvertUnaryToLib : public ConvertOp<MathOp> {
    using ConvertOp<MathOp>::createTensorOp;
    using ConvertOp<MathOp>::ConvertOp;
    using ConvertOp<MathOp>::calCount;

    LogicalResult matchAndRewrite(MathOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        auto src = rewriter.getRemappedValue(op->getOperand(0));
        Value count = consts.i64(calCount(dst));
        rewriter.create<LibraryOp>(loc, dst, src, /*sharedTmpBuffer*/ Value{}, count, consts.i1(false));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

template <typename MathOp, typename L2Op>
struct ConvertUnaryToL2 : public ConvertOp<MathOp> {
    using ConvertOp<MathOp>::createTensorOp;
    using ConvertOp<MathOp>::ConvertOp;
    using ConvertOp<MathOp>::calCount;

    LogicalResult matchAndRewrite(MathOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        auto src = rewriter.getRemappedValue(op->getOperand(0));
        rewriter.create<L2Op>(loc, dst, src, consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct LowerMathPass : public asclower::impl::LowerMathBase<LowerMathPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addDynamicallyLegalDialect<math::MathDialect>(
            [&converter](Operation* op) { return converter.isLegal(op); });
        target.addLegalDialect<ascendc::AscendCDialect>();
        target.addLegalOp<arith::ConstantOp>();
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertUnaryToL2<math::LogOp, ascendc::LnL2Op>, ConvertUnaryToLib<math::ErfOp, ascendc::ErfOp>,
            ConvertUnaryToLib<math::AsinOp, ascendc::AsinOp>, ConvertUnaryToLib<math::AcosOp, ascendc::AcosOp>,
            ConvertUnaryToLib<math::CosOp, ascendc::CosOp>, ConvertUnaryToLib<math::SinOp, ascendc::SinOp>,
            ConvertUnaryToLib<math::TanOp, ascendc::TanOp>, ConvertUnaryToLib<math::SinhOp, ascendc::SinhOp>,
            ConvertUnaryToLib<math::CoshOp, ascendc::CoshOp>, ConvertUnaryToLib<math::TanhOp, ascendc::TanhOp>,
            ConvertUnaryToLib<math::CeilOp, ascendc::CeilOp>, ConvertUnaryToLib<math::FloorOp, ascendc::FloorOp>,
            ConvertUnaryToLib<math::RoundOp, ascendc::RoundOp>, ConvertUnaryToL2<math::AbsFOp, ascendc::AbsL2Op>,
            ConvertUnaryToL2<math::ExpOp, ascendc::ExpL2Op>, ConvertUnaryToL2<math::SqrtOp, ascendc::SqrtL2Op>,
            ConvertUnaryToL2<math::RsqrtOp, ascendc::RsqrtL2Op>, ConvertUnaryToLib<math::Log2Op, ascendc::Log2Op>,
            ConvertUnaryToL2<math::AbsIOp, ascendc::AbsL2Op>
            //
            >(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerMathPass() { return std::make_unique<LowerMathPass>(); }
