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
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERARITHBINARY
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

template <typename ArithOp, typename L2Op>
struct ConvertToL2 : ConvertOp<ArithOp> {
    using ConvertOp<ArithOp>::ConvertOp;
    using ConvertOp<ArithOp>::createTensorOp;
    using ConvertOp<ArithOp>::calCount;

    LogicalResult matchAndRewrite(ArithOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        auto dst = createTensorOp(rewriter, loc, op.getType());
        rewriter.create<L2Op>(
            loc, dst, rewriter.getRemappedValue(op.getLhs()), rewriter.getRemappedValue(op.getRhs()),
            consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

template <typename Converter, typename CmpOp>
struct ConvertCmpBase : public ConvertOp<CmpOp> {
    using ConvertOp<CmpOp>::ConvertOp;
    using ConvertOp<CmpOp>::createReCastOp;
    using ConvertOp<CmpOp>::createTensorOp;
    using ConvertOp<CmpOp>::typeConverter;

    static FailureOr<std::pair<Value, Value>> rewriteInputs(CmpOp op, ConvertRewriter& rewriter)
    {
        Value src0 = rewriter.getRemappedValue(op.getLhs());
        Value src1 = rewriter.getRemappedValue(op.getRhs());
        return std::pair(src0, src1);
    }

    LogicalResult matchAndRewrite(CmpOp op, ConvertRewriter& rewriter) const override
    {
        auto inputs = Converter::rewriteInputs(op, rewriter);
        if (failed(inputs))
            return failure();
        auto srcType = typeConverter->template convertType<ShapedType>(op.getLhs().getType());
        I1ReplacementType replType(op.getContext());
        int64_t dstShape = llvm::divideCeilSigned(srcType.getNumElements(), replType.width);
        auto loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, dstShape, replType.iType);
        dst = createReCastOp(rewriter, loc, dst, dstShape, replType.uiType);
        Value zero = ascir::ConstantOpBuilder(rewriter).i64(0);
        rewriter.create<ascendc::CompareL0Op>(
            loc, dst, inputs->first, inputs->second, Converter::getCmpMode(op.getPredicate()), zero, zero,
            rewriter.create<ascendc::ConstructOp>(loc, rewriter.getType<ascendc::BinaryRepeatParamsType>()));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertCmpF : public ConvertCmpBase<ConvertCmpF, arith::CmpFOp> {
    using ConvertCmpBase::ConvertCmpBase;

    static ascendc::CMPMODE getCmpMode(arith::CmpFPredicate pred)
    {
        switch (pred) {
            case arith::CmpFPredicate::OEQ:
                return ascendc::CMPMODE::EQ;
            case arith::CmpFPredicate::ONE:
                return ascendc::CMPMODE::NE;
            case arith::CmpFPredicate::OLT:
                return ascendc::CMPMODE::LT;
            case arith::CmpFPredicate::OLE:
                return ascendc::CMPMODE::LE;
            case arith::CmpFPredicate::OGT:
                return ascendc::CMPMODE::GT;
            case arith::CmpFPredicate::OGE:
                return ascendc::CMPMODE::GE;
            default:
                llvm_unreachable("unexpected arith::CmpFPredicate");
        }
    }
};

struct ConvertCmpI : public ConvertCmpBase<ConvertCmpI, arith::CmpIOp> {
    using ConvertCmpBase::ConvertCmpBase;

    static ascendc::CMPMODE getCmpMode(arith::CmpIPredicate pred)
    {
        switch (pred) {
            case arith::CmpIPredicate::eq:
                return ascendc::CMPMODE::EQ;
            case arith::CmpIPredicate::ne:
                return ascendc::CMPMODE::NE;
            case arith::CmpIPredicate::slt:
                return ascendc::CMPMODE::LT;
            case arith::CmpIPredicate::sle:
                return ascendc::CMPMODE::LE;
            case arith::CmpIPredicate::sgt:
                return ascendc::CMPMODE::GT;
            case arith::CmpIPredicate::sge:
                return ascendc::CMPMODE::GE;
            default:
                llvm_unreachable("unexpected arith::CmpIPredicate");
        }
    }

    static FailureOr<std::pair<Value, Value>> rewriteInputs(arith::CmpIOp op, ConvertRewriter& rewriter)
    {
        Value src0 = rewriter.getRemappedValue(op.getLhs());
        Value src1 = rewriter.getRemappedValue(op.getRhs());
        auto srcType = cast<ShapedType>(src0.getType());
        if (ascendc::isTargetArchC310(op))
            return std::pair(src0, src1);
        unsigned bitWidth = srcType.getElementTypeBitWidth();
        if (bitWidth != 16 && bitWidth != 32)
            return op.emitOpError("can only be lowered with i16 or i32 tensor operands");
        auto castToType = bitWidth == 16 ? rewriter.getF16Type() : rewriter.getF32Type();
        auto loc = op.getLoc();
        Value src0Casted = createTensorOp(rewriter, loc, srcType.getShape(), castToType);
        Value src1Casted = createTensorOp(rewriter, loc, srcType.getShape(), castToType);
        Value zero = ascir::ConstantOpBuilder(rewriter).i64(0);
        rewriter.create<ascendc::CastL2Op>(loc, src0Casted, src0, ascendc::RoundMode::CAST_NONE, zero);
        rewriter.create<ascendc::CastL2Op>(loc, src1Casted, src1, ascendc::RoundMode::CAST_NONE, zero);
        return std::pair(src0Casted, src1Casted);
    }
};

struct LowerArithBinaryPass : public asclower::impl::LowerArithBinaryBase<LowerArithBinaryPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addLegalDialect<ascendc::AscendCDialect, arith::ArithDialect>();
        target.addDynamicallyLegalOp<
            //
            arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, arith::MulFOp, arith::MulIOp, arith::DivFOp,
            arith::DivSIOp, arith::MaxSIOp, arith::MinSIOp, arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
            arith::MinNumFOp, arith::AndIOp, arith::OrIOp, arith::CmpFOp, arith::CmpIOp
            //
            >([&converter](Operation* op) { return converter.isLegal(op); });
        target.addLegalOp<UnrealizedConversionCastOp>();
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertToL2<arith::AddFOp, ascendc::AddL2Op>, ConvertToL2<arith::AddIOp, ascendc::AddL2Op>,
            ConvertToL2<arith::SubFOp, ascendc::SubL2Op>, ConvertToL2<arith::SubIOp, ascendc::SubL2Op>,
            ConvertToL2<arith::MulFOp, ascendc::MulL2Op>, ConvertToL2<arith::MulIOp, ascendc::MulL2Op>,
            ConvertToL2<arith::DivFOp, ascendc::DivL2Op>, ConvertToL2<arith::DivSIOp, ascendc::DivL2Op>,
            ConvertToL2<arith::MaximumFOp, ascendc::MaxL2Op>, ConvertToL2<arith::MinimumFOp, ascendc::MinL2Op>,
            ConvertToL2<arith::MaxSIOp, ascendc::MaxL2Op>, ConvertToL2<arith::MinSIOp, ascendc::MinL2Op>,
            ConvertToL2<arith::MaxNumFOp, ascendc::MaxL2Op>, ConvertToL2<arith::MinNumFOp, ascendc::MinL2Op>,
            ConvertToL2<arith::AndIOp, ascendc::AndL2Op>, ConvertToL2<arith::OrIOp, ascendc::OrL2Op>, ConvertCmpF,
            ConvertCmpI
            //
            >(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerArithBinaryPass() { return std::make_unique<LowerArithBinaryPass>(); }
