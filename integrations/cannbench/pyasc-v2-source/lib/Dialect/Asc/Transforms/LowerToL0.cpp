/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_LOWERTOL0
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace ascendc;

namespace {
template <typename L2Op, typename L0Op>
LogicalResult lowerUnaryOp(L2Op op, PatternRewriter& rewriter)
{
    ascir::ConstantOpBuilder consts(rewriter);
    auto loc = op.getLoc();
    auto zero = consts.i64(0);
    auto mask = rewriter.create<emitasc::MaskOp>(loc, zero, zero);
    auto repeatParams = rewriter.create<ConstructOp>(loc, rewriter.getType<UnaryRepeatParamsType>());
    rewriter.replaceOpWithNewOp<L0Op>(op, op.getDst(), op.getSrc(), mask, zero, repeatParams);
    return success();
}

template <typename L2Op, typename L0Op>
LogicalResult lowerBinaryOp(L2Op op, PatternRewriter& rewriter)
{
    ascir::ConstantOpBuilder consts(rewriter);
    auto loc = op.getLoc();
    auto zero = consts.i64(0);
    auto mask = rewriter.create<emitasc::MaskOp>(loc, zero, zero);
    auto repeatParams = rewriter.create<ConstructOp>(loc, rewriter.getType<BinaryRepeatParamsType>());
    rewriter.replaceOpWithNewOp<L0Op>(op, op.getDst(), op.getSrc0(), op.getSrc1(), mask, zero, repeatParams);
    return success();
}

template <typename L2Op, typename L0Op>
LogicalResult lowerDuplicateOp(L2Op op, PatternRewriter& rewriter)
{
    ascir::ConstantOpBuilder consts(rewriter);
    auto loc = op.getLoc();
    auto zero = consts.i64(0);
    auto mask = rewriter.create<emitasc::MaskOp>(loc, zero, zero);
    rewriter.replaceOpWithNewOp<L0Op>(op, op.getDst(), op.getScalar(), mask, zero, zero, zero);
    return success();
}

template <typename L2Op, typename L0Op>
LogicalResult lowerCastOp(L2Op op, PatternRewriter& rewriter)
{
    ascir::ConstantOpBuilder consts(rewriter);
    auto loc = op.getLoc();
    auto zero = consts.i64(0);
    auto mask = rewriter.create<emitasc::MaskOp>(loc, zero, zero);
    auto repeatParams = rewriter.create<ConstructOp>(loc, rewriter.getType<UnaryRepeatParamsType>());
    rewriter.replaceOpWithNewOp<L0Op>(op, op.getDst(), op.getSrc(), op.getRoundMode(), mask, zero, repeatParams);
    return success();
}

template <typename L2Op, typename L0Op>
LogicalResult lowerCompareOp(L2Op op, PatternRewriter& rewriter)
{
    ascir::ConstantOpBuilder consts(rewriter);
    auto loc = op.getLoc();
    auto zero = consts.i64(0);
    auto mask = rewriter.create<emitasc::MaskOp>(loc, zero, zero);
    auto repeatParams = rewriter.create<ConstructOp>(loc, rewriter.getType<BinaryRepeatParamsType>());
    rewriter.replaceOpWithNewOp<L0Op>(
        op, op.getDst(), op.getSrc0(), op.getSrc1(), op.getCmpMode(), mask, zero, repeatParams);
    return success();
}

template <typename L2Op, typename L0Op>
LogicalResult lowerVectorScalarOp(L2Op op, PatternRewriter& rewriter)
{
    ascir::ConstantOpBuilder consts(rewriter);
    auto loc = op.getLoc();
    auto zero = consts.i64(0);
    auto mask = rewriter.create<emitasc::MaskOp>(loc, zero, zero);
    auto repeatParams = rewriter.create<ConstructOp>(loc, rewriter.getType<UnaryRepeatParamsType>());
    rewriter.replaceOpWithNewOp<L0Op>(op, op.getDst(), op.getSrc(), op.getScalar(), mask, zero, repeatParams);
    return success();
}

struct LowerToL0Pass : public ascendc::impl::LowerToL0Base<LowerToL0Pass> {
    void runOnOperation() override
    {
        auto funcOp = getOperation();
        RewritePatternSet patterns(funcOp.getContext());
        populateLowerToL0Patterns(patterns);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed())
            signalPassFailure();
    }
};
} // namespace

namespace mlir {
namespace ascendc {
void populateLowerToL0Patterns(RewritePatternSet& patterns)
{
    patterns.add(lowerUnaryOp<AbsL2Op, AbsL0Op>);
    patterns.add(lowerUnaryOp<ExpL2Op, ExpL0Op>);
    patterns.add(lowerUnaryOp<LnL2Op, LnL0Op>);
    patterns.add(lowerUnaryOp<NotL2Op, NotL0Op>);
    patterns.add(lowerUnaryOp<ReciprocalL2Op, ReciprocalL0Op>);
    patterns.add(lowerUnaryOp<ReluL2Op, ReluL0Op>);
    patterns.add(lowerUnaryOp<RsqrtL2Op, RsqrtL0Op>);
    patterns.add(lowerUnaryOp<SqrtL2Op, SqrtL0Op>);
    patterns.add(lowerBinaryOp<AddL2Op, AddL0Op>);
    patterns.add(lowerBinaryOp<AddDeqReluL2Op, AddDeqReluL0Op>);
    patterns.add(lowerBinaryOp<AddReluL2Op, AddReluL0Op>);
    patterns.add(lowerBinaryOp<AndL2Op, AndL0Op>);
    patterns.add(lowerBinaryOp<DivL2Op, DivL0Op>);
    patterns.add(lowerBinaryOp<FusedMulAddL2Op, FusedMulAddL0Op>);
    patterns.add(lowerBinaryOp<FusedMulAddReluL2Op, FusedMulAddReluL0Op>);
    patterns.add(lowerBinaryOp<MaxL2Op, MaxL0Op>);
    patterns.add(lowerBinaryOp<MinL2Op, MinL0Op>);
    patterns.add(lowerBinaryOp<MulL2Op, MulL0Op>);
    patterns.add(lowerBinaryOp<MulAddDstL2Op, MulAddDstL0Op>);
    patterns.add(lowerBinaryOp<OrL2Op, OrL0Op>);
    patterns.add(lowerBinaryOp<SubL2Op, SubL0Op>);
    patterns.add(lowerBinaryOp<SubReluL2Op, SubReluL0Op>);
    // TODO: Add correct calculation for bitmasks in FillAscOperandsPass
    // patterns.add(lowerDuplicateOp<DuplicateL2Op, DuplicateL0Op>);
    patterns.add(lowerCompareOp<CompareL2Op, CompareL0Op>);
    patterns.add(lowerCastOp<CastL2Op, CastL0Op>);
    patterns.add(lowerVectorScalarOp<AddsL2Op, AddsL0Op>);
    patterns.add(lowerVectorScalarOp<DivsL2Op, DivsL0Op>);
    patterns.add(lowerVectorScalarOp<LeakyReluL2Op, LeakyReluL0Op>);
    patterns.add(lowerVectorScalarOp<MaxsL2Op, MaxsL0Op>);
    patterns.add(lowerVectorScalarOp<MinsL2Op, MinsL0Op>);
    patterns.add(lowerVectorScalarOp<MulsL2Op, MulsL0Op>);
    patterns.add(lowerVectorScalarOp<ShiftLeftL2Op, ShiftLeftL0Op>);
    patterns.add(lowerVectorScalarOp<ShiftRightL2Op, ShiftRightL0Op>);
    patterns.add(lowerVectorScalarOp<SubsL2Op, SubsL0Op>);
}

std::unique_ptr<Pass> createLowerToL0Pass() { return std::make_unique<LowerToL0Pass>(); }
} // namespace ascendc
} // namespace mlir
