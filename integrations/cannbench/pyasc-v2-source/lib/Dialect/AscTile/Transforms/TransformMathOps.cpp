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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_TRANSFORMMATHOPS
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;

namespace {

std::optional<TypedAttr> getSplatValue(arith::ConstantOp cstOp)
{
    if (!cstOp)
        return std::nullopt;
    auto dense = dyn_cast<DenseElementsAttr>(cstOp.getValue());
    if (!dense || !dense.isSplat())
        return std::nullopt;
    return dense.getSplatValue<TypedAttr>();
}

Value materializeSplatValue(OpBuilder& builder, Value cstTile)
{
    if (auto s = getSplatValue(cstTile.getDefiningOp<arith::ConstantOp>()))
        return builder.create<arith::ConstantOp>(cstTile.getLoc(), s.value());
    if (auto splat = cstTile.getDefiningOp<tensor::SplatOp>())
        return splat.getInput();
    return {};
}

template <typename AttrOrValue>
bool matchPatternZero(AttrOrValue value)
{
    return matchPattern(value, m_AnyZeroFloat()) || matchPattern(value, m_Zero());
}

bool isZero(Value value)
{
    if (auto cstOp = value.getDefiningOp<arith::ConstantOp>()) {
        if (matchPatternZero(value))
            return true;
        auto splat = getSplatValue(cstOp);
        return splat && matchPatternZero(*splat);
    }
    if (auto splatOp = value.getDefiningOp<tensor::SplatOp>())
        return matchPatternZero(splatOp.getInput());
    return false;
}

asctile::CompareMode getCmpMode(arith::CmpFPredicate pred)
{
    switch (pred) {
        case arith::CmpFPredicate::OEQ:
            return asctile::CompareMode::EQ;
        case arith::CmpFPredicate::ONE:
            return asctile::CompareMode::NE;
        case arith::CmpFPredicate::OLT:
            return asctile::CompareMode::LT;
        case arith::CmpFPredicate::OLE:
            return asctile::CompareMode::LE;
        case arith::CmpFPredicate::OGT:
            return asctile::CompareMode::GT;
        case arith::CmpFPredicate::OGE:
            return asctile::CompareMode::GE;
        default:
            llvm_unreachable("unexpected arith::CmpFPredicate");
    }
}

asctile::CompareMode getCmpMode(arith::CmpIPredicate pred)
{
    switch (pred) {
        case arith::CmpIPredicate::eq:
            return asctile::CompareMode::EQ;
        case arith::CmpIPredicate::ne:
            return asctile::CompareMode::NE;
        case arith::CmpIPredicate::slt:
            return asctile::CompareMode::LT;
        case arith::CmpIPredicate::sle:
            return asctile::CompareMode::LE;
        case arith::CmpIPredicate::sgt:
            return asctile::CompareMode::GT;
        case arith::CmpIPredicate::sge:
            return asctile::CompareMode::GE;
        default:
            llvm_unreachable("unexpected arith::CmpIPredicate");
    }
}

template <typename ArithOp, typename TileOp>
struct ScalarizeArithOp : OpRewritePattern<ArithOp> {
    using OpRewritePattern<ArithOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ArithOp op, PatternRewriter& rewriter) const override
    {
        if (!isa<asctile::LocalTensorType>(op.getType()))
            return failure();
        Value newLhs, newRhs;
        if (auto splat = materializeSplatValue(rewriter, op.getLhs())) {
            newLhs = op.getRhs();
            newRhs = splat;
        } else if (auto splat = materializeSplatValue(rewriter, op.getRhs())) {
            newLhs = op.getLhs();
            newRhs = splat;
        } else {
            return failure();
        }
        rewriter.replaceOpWithNewOp<TileOp>(op, op.getType(), newLhs, newRhs);
        return success();
    }
};

template <typename ArithOp, typename TileOp>
struct ScalarizeArithRhsOp : OpRewritePattern<ArithOp> {
    using OpRewritePattern<ArithOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ArithOp op, PatternRewriter& rewriter) const override
    {
        auto type = op.getType();
        if (!isa<asctile::LocalTensorType>(type))
            return failure();
        if (auto splat = materializeSplatValue(rewriter, op.getRhs())) {
            rewriter.replaceOpWithNewOp<TileOp>(op, type, op.getLhs(), splat);
            return success();
        }
        return failure();
    }
};

template <typename MaxOp>
struct MaxWithZeroToReluOp : OpRewritePattern<MaxOp> {
    using OpRewritePattern<MaxOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(MaxOp op, PatternRewriter& rewriter) const override
    {
        auto type = op.getType();
        if (!isa<asctile::LocalTensorType>(type))
            return failure();
        Value operand;
        if (isZero(op.getLhs())) {
            operand = op.getRhs();
        } else if (isZero(op.getRhs())) {
            operand = op.getLhs();
        }
        if (!operand) {
            return failure();
        }
        rewriter.replaceOpWithNewOp<asctile::ReluOp>(op, type, operand);
        return success();
    }
};

struct SelectMulToLeakyRelu : OpRewritePattern<arith::SelectOp> {
    using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

    static Value extractAlphaFromMul(Value mulResult, Value expectedX, OpBuilder& builder)
    {
        if (auto mulfOp = mulResult.getDefiningOp<arith::MulFOp>()) {
            Value other;
            if (mulfOp.getLhs() == expectedX)
                other = mulfOp.getRhs();
            else if (mulfOp.getRhs() == expectedX)
                other = mulfOp.getLhs();
            else
                return {};
            return materializeSplatValue(builder, other);
        }
        if (auto mulsOp = mulResult.getDefiningOp<asctile::MulSOp>()) {
            if (mulsOp.getBase() == expectedX)
                return mulsOp.getValue();
        }
        return {};
    }

    static std::pair<Value, Value> matchLeakyRelu(Value selMask, Value src0, Value src1, OpBuilder& builder)
    {
        Value cmpLhs, cmpRhs;
        asctile::CompareMode mode;
        if (auto cmpOp = selMask.getDefiningOp<arith::CmpFOp>()) {
            cmpLhs = cmpOp.getLhs();
            cmpRhs = cmpOp.getRhs();
            mode = getCmpMode(cmpOp.getPredicate());
        } else if (auto cmpSOp = selMask.getDefiningOp<asctile::CmpSOp>()) {
            cmpLhs = cmpSOp.getBase();
            cmpRhs = cmpSOp.getValue();
            mode = cmpSOp.getCmpMode();
        } else {
            return {};
        }
        bool lhsZero = isZero(cmpLhs);
        bool rhsZero = isZero(cmpRhs);
        if (!lhsZero && !rhsZero)
            return {};
        Value x = rhsZero ? cmpLhs : cmpRhs;
        bool isPositive = rhsZero ? (mode == asctile::CompareMode::GE || mode == asctile::CompareMode::GT) :
                                    (mode == asctile::CompareMode::LE || mode == asctile::CompareMode::LT);
        Value positiveSrc = isPositive ? src0 : src1;
        Value negativeSrc = isPositive ? src1 : src0;
        if (positiveSrc != x)
            return {};
        Value alpha = extractAlphaFromMul(negativeSrc, x, builder);
        if (!alpha)
            return {};
        return {x, alpha};
    }

    LogicalResult matchAndRewrite(arith::SelectOp op, PatternRewriter& rewriter) const override
    {
        auto type = op.getType();
        if (!isa<asctile::LocalTensorType>(type))
            return failure();
        auto elemType = cast<ShapedType>(type).getElementType();
        if (!elemType.isF16() && !elemType.isF32())
            return failure();
        auto [x, alpha] = matchLeakyRelu(op.getCondition(), op.getTrueValue(), op.getFalseValue(), rewriter);
        if (!x)
            return failure();
        rewriter.replaceOpWithNewOp<asctile::LeakyReluOp>(op, type, x, alpha);
        return success();
    }
};

template <typename CmpOp>
struct ScalarizeCompare : OpRewritePattern<CmpOp> {
    using OpRewritePattern<CmpOp>::OpRewritePattern;

    static asctile::CompareMode invertCmpMode(asctile::CompareMode mode)
    {
        switch (mode) {
            case asctile::CompareMode::EQ:
                return asctile::CompareMode::EQ;
            case asctile::CompareMode::NE:
                return asctile::CompareMode::NE;
            case asctile::CompareMode::LT:
                return asctile::CompareMode::GT;
            case asctile::CompareMode::LE:
                return asctile::CompareMode::GE;
            case asctile::CompareMode::GT:
                return asctile::CompareMode::LT;
            case asctile::CompareMode::GE:
                return asctile::CompareMode::LE;
        }
        llvm_unreachable("unexpected cmpmode");
    }

    LogicalResult matchAndRewrite(CmpOp op, PatternRewriter& rewriter) const override
    {
        Value newLhs, newRhs;
        auto mode = getCmpMode(op.getPredicate());
        if (auto splat = materializeSplatValue(rewriter, op.getLhs())) {
            newLhs = op.getRhs();
            newRhs = splat;
            mode = invertCmpMode(mode);
        } else if (auto splat = materializeSplatValue(rewriter, op.getRhs())) {
            newLhs = op.getLhs();
            newRhs = splat;
        } else {
            return failure();
        }
        rewriter.replaceOpWithNewOp<asctile::CmpSOp>(op, op.getType(), newLhs, newRhs, mode);
        return success();
    }
};

struct ScalarizeShL : OpRewritePattern<arith::ShLIOp> {
    using OpRewritePattern<arith::ShLIOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::ShLIOp op, PatternRewriter& rewriter) const override
    {
        if (!isa<asctile::LocalTensorType>(op.getType()))
            return failure();

        Value scalar = materializeSplatValue(rewriter, op.getRhs());
        if (!scalar)
            return failure();

        rewriter.replaceOpWithNewOp<asctile::ShLSOp>(op, op.getType(), op.getLhs(), scalar);

        return success();
    }
};

struct ScalarizeShR : OpRewritePattern<arith::ShRSIOp> {
    using OpRewritePattern<arith::ShRSIOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::ShRSIOp op, PatternRewriter& rewriter) const override
    {
        if (!isa<asctile::LocalTensorType>(op.getType()))
            return failure();

        Value scalar = materializeSplatValue(rewriter, op.getRhs());
        if (!scalar)
            return failure();

        rewriter.replaceOpWithNewOp<asctile::ShRSOp>(op, op.getType(), op.getLhs(), scalar);

        return success();
    }
};

class TransformMathOpsPass : public asctile::impl::TransformMathOpsBase<TransformMathOpsPass> {
public:
    TransformMathOpsPass() = default;

    void runOnOperation() override
    {
        auto op = getOperation();
        MLIRContext* context = op.getContext();
        RewritePatternSet patterns(context);
        patterns.add<
            //
            MaxWithZeroToReluOp<arith::MaximumFOp>, MaxWithZeroToReluOp<arith::MaxNumFOp>,
            MaxWithZeroToReluOp<arith::MaxSIOp>, SelectMulToLeakyRelu
            //
            >(context, /*benefit=*/2);
        patterns.add<
            //
            ScalarizeArithOp<arith::AddFOp, asctile::AddSOp>, ScalarizeArithOp<arith::AddIOp, asctile::AddSOp>,
            ScalarizeArithRhsOp<arith::SubFOp, asctile::SubSOp>, ScalarizeArithRhsOp<arith::SubIOp, asctile::SubSOp>,
            ScalarizeArithOp<arith::MulFOp, asctile::MulSOp>, ScalarizeArithOp<arith::MulIOp, asctile::MulSOp>,
            ScalarizeArithRhsOp<arith::DivFOp, asctile::DivSOp>, ScalarizeShL, ScalarizeShR,
            ScalarizeArithOp<arith::MaximumFOp, asctile::MaxSOp>, ScalarizeArithOp<arith::MaxSIOp, asctile::MaxSOp>,
            ScalarizeArithOp<arith::MinimumFOp, asctile::MinSOp>, ScalarizeArithOp<arith::MinSIOp, asctile::MinSOp>,
            ScalarizeCompare<arith::CmpFOp>, ScalarizeCompare<arith::CmpIOp>
            //
            >(context, /*benefit=*/1);
        if (applyPatternsAndFoldGreedily(op, std::move(patterns)).failed())
            signalPassFailure();
    }
};
} // namespace

std::unique_ptr<Pass> mlir::asctile::createTransformMathOpsPass() { return std::make_unique<TransformMathOpsPass>(); }
