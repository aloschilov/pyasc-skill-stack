/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/AscVF/Transforms/Passes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_LOWERTOREG
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

struct VFInfo {
    Value calCount, oneRepeatSize, repeatTimes;
    Type elemType;

    static Value createVecLen(ImplicitLocOpBuilder& builder, Operation* op)
    {
        if (auto vecLen = ascendc::getVecLen(op))
            return builder.create<arith::ConstantIndexOp>(*vecLen);
        return builder.create<ascendc::GetVecLenOp>(builder.getIndexType());
    }

    explicit VFInfo(ascvf::VecScopeOp op)
    {
        auto groupOp = op->getParentOfType<ascvf::VFGroupOp>();
        assert(groupOp && "ascvf.vec_scope op must be inside ascvf.vf_group op");
        auto builder = ImplicitLocOpBuilder::atBlockBegin(UnknownLoc::get(op.getContext()), op.getBody());
        auto vecLen = createVecLen(builder, op);
        elemType = groupOp.getGroupType();
        Value sizeIndex = builder.create<arith::ConstantIndexOp>(ascendc::getTypeSize(elemType));
        oneRepeatSize = builder.create<arith::DivSIOp>(vecLen, sizeIndex);
        calCount = groupOp.getCalCount();
        if (!isa<IndexType>(calCount.getType()))
            calCount = builder.create<arith::IndexCastOp>(builder.getIndexType(), calCount);
        repeatTimes = builder.create<arith::CeilDivSIOp>(builder.getIndexType(), calCount, oneRepeatSize);
    }
};

class RewriterAdaptor {
    using Rewriter = ConversionPatternRewriter;

    Rewriter& rewriter;

    template <size_t... indices>
    auto createRegTensorsImpl(Type elemType, std::index_sequence<indices...>)
    {
        return std::make_tuple((static_cast<void>(indices), createRegTensor(elemType))...);
    }

public:
    RewriterAdaptor(Rewriter& rewriter) : rewriter(rewriter) {}
    ~RewriterAdaptor() = default;

    Rewriter& operator*() { return rewriter; }
    Rewriter* operator->() { return &rewriter; }

    template <typename OpT, typename... Args>
    OpT create(Args&&... args)
    {
        return rewriter.create<OpT>(rewriter.getUnknownLoc(), args...);
    }

    ascendc::RegTensorOp createRegTensor(Type elemType)
    {
        return create<ascendc::RegTensorOp>(rewriter.getType<ascendc::RegTensorType>(elemType));
    }

    template <size_t count>
    auto createRegTensors(Type elemType)
    {
        return createRegTensorsImpl(elemType, std::make_index_sequence<count>{});
    }

    Value createUI32Variable(Value initValue)
    {
        return create<emitasc::VariableOp>(MemRefType::get(1, rewriter.getIntegerType(32U, false)), initValue);
    }

    Value createMaskOp(Type elemType, ascendc::MaskPattern pattern = ascendc::MaskPattern::ALL)
    {
        return create<ascendc::CreateMaskOp>(rewriter.getType<ascendc::MaskRegType>(), elemType, pattern);
    }

    Value updateMaskOp(Value calCount, Type elemType)
    {
        return create<ascendc::UpdateMaskOp>(rewriter.getType<ascendc::MaskRegType>(), calCount, elemType);
    }
};

template <typename OpT>
struct ConvertOp : public OpConversionPattern<OpT> {
    ConvertOp(MLIRContext* context, const VFInfo& vfInfo, PatternBenefit benefit = 1)
        : OpConversionPattern<OpT>::OpConversionPattern(context, benefit), vfInfo(vfInfo)
    {}

    virtual LogicalResult matchAndRewrite(OpT op, RewriterAdaptor& adaptor) const = 0;

    LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        RewriterAdaptor adaptor(rewriter);
        return matchAndRewrite(op, adaptor);
    }

protected:
    const VFInfo& vfInfo;
};

template <typename L2Op, typename RegOp>
struct ConvertBinaryL2 : public ConvertOp<L2Op> {
    using ConvertOp<L2Op>::ConvertOp;
    using ConvertOp<L2Op>::vfInfo;

    LogicalResult matchAndRewrite(L2Op op, RewriterAdaptor& adaptor) const override
    {
        auto [src0Reg, src1Reg, dstReg] = adaptor.createRegTensors<3>(vfInfo.elemType);
        Value calCount = adaptor.createUI32Variable(vfInfo.calCount);
        Value maskAll = adaptor.createMaskOp(vfInfo.elemType);
        auto loop = adaptor.create<ascvf::VFForOp>(vfInfo.repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());
        Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        Value mulOp = adaptor.create<arith::MulIOp>(loop.getInductionVar(), vfInfo.oneRepeatSize);
        adaptor.create<ascvf::LoadOp>(src0Reg, op.getSrc0(), mulOp);
        adaptor.create<ascvf::LoadOp>(src1Reg, op.getSrc1(), mulOp);
        adaptor.create<RegOp>(dstReg, src0Reg, src1Reg, maskAll);
        adaptor.create<ascvf::StoreOp>(op.getDst(), mulOp, dstReg, updateMask);
        adaptor->eraseOp(op);
        return success();
    }
};

template <typename L2Op, typename RegOp>
struct ConvertUnaryL2 : public ConvertOp<L2Op> {
    using ConvertOp<L2Op>::ConvertOp;
    using ConvertOp<L2Op>::vfInfo;

    LogicalResult matchAndRewrite(L2Op op, RewriterAdaptor& adaptor) const override
    {
        auto [srcReg, dstReg] = adaptor.createRegTensors<2>(vfInfo.elemType);
        Value calCount = adaptor.createUI32Variable(vfInfo.calCount);
        Value maskAll = adaptor.createMaskOp(vfInfo.elemType);
        auto loop = adaptor.create<ascvf::VFForOp>(vfInfo.repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());
        Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        Value mulOp = adaptor.create<arith::MulIOp>(loop.getInductionVar(), vfInfo.oneRepeatSize);
        adaptor.create<ascvf::LoadOp>(srcReg, op.getSrc(), mulOp);
        adaptor.create<RegOp>(dstReg, srcReg, maskAll);
        adaptor.create<ascvf::StoreOp>(op.getDst(), mulOp, dstReg, updateMask);
        adaptor->eraseOp(op);
        return success();
    }
};

template <typename L2Op, typename RegOp>
struct ConvertVecScalarL2 : public ConvertOp<L2Op> {
    using ConvertOp<L2Op>::ConvertOp;
    using ConvertOp<L2Op>::vfInfo;

    LogicalResult matchAndRewrite(L2Op op, RewriterAdaptor& adaptor) const override
    {
        auto [srcReg, dstReg] = adaptor.createRegTensors<2>(vfInfo.elemType);
        Value calCount = adaptor.createUI32Variable(vfInfo.calCount);
        auto loop = adaptor.create<ascvf::VFForOp>(vfInfo.repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());
        Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        Value mulOp = adaptor.create<arith::MulIOp>(loop.getInductionVar(), vfInfo.oneRepeatSize);
        adaptor.create<ascvf::LoadOp>(srcReg, op.getSrc(), mulOp);
        adaptor.create<RegOp>(dstReg, srcReg, op.getScalar(), updateMask);
        adaptor.create<ascvf::StoreOp>(op.getDst(), mulOp, dstReg, updateMask);
        adaptor->eraseOp(op);
        return success();
    }
};

template <typename L2Op, typename BinRegOp>
struct ConvertVecScalarWithDuplicateL2 : public ConvertOp<L2Op> {
    using ConvertOp<L2Op>::ConvertOp;
    using ConvertOp<L2Op>::vfInfo;

    LogicalResult matchAndRewrite(L2Op op, RewriterAdaptor& adaptor) const override
    {
        auto [srcReg, dupReg, dstReg] = adaptor.createRegTensors<3>(vfInfo.elemType);
        Value calCount = adaptor.createUI32Variable(vfInfo.calCount);
        Value maskAll = adaptor.createMaskOp(vfInfo.elemType);
        auto loop = adaptor.create<ascvf::VFForOp>(vfInfo.repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());
        Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        Value mulOp = adaptor.create<arith::MulIOp>(loop.getInductionVar(), vfInfo.oneRepeatSize);
        adaptor.create<ascvf::LoadOp>(srcReg, op.getSrc(), mulOp);
        adaptor.create<ascendc::DuplicateRegOp>(dupReg, op.getScalar(), maskAll);
        adaptor.create<BinRegOp>(dstReg, srcReg, dupReg, maskAll);
        adaptor.create<ascvf::StoreOp>(op.getDst(), mulOp, dstReg, updateMask);
        adaptor->eraseOp(op);
        return success();
    }
};

template <typename ReduceL2Op, typename AccumulateRegOp, typename ReduceRegOp>
struct ConvertReduceL2 : public ConvertOp<ReduceL2Op> {
    using ConvertOp<ReduceL2Op>::ConvertOp;
    using ConvertOp<ReduceL2Op>::vfInfo;

    template <typename ReduceOpType>
    static Value getNeutralElement(ascir::ConstantOpBuilder& consts, Type elemType)
    {
        if constexpr (std::is_same_v<ReduceOpType, ascendc::ReduceMaxL2Op>) {
            if (elemType.isF32())
                return consts.f32(-std::numeric_limits<float>::infinity());
            if (elemType.isF16())
                return consts.f16(-std::numeric_limits<float>::infinity());
            if (elemType.isInteger(32))
                return consts.i32(-std::numeric_limits<int>::infinity());
        }
        if constexpr (std::is_same_v<ReduceOpType, ascendc::ReduceSumL2Op>) {
            if (elemType.isF32())
                return consts.f32(0);
            if (elemType.isF16())
                return consts.f16(0);
            if (elemType.isInteger(32))
                return consts.i32(0);
        }
        if constexpr (std::is_same_v<ReduceOpType, ascendc::ReduceMinL2Op>) {
            if (elemType.isF32())
                return consts.f32(std::numeric_limits<float>::infinity());
            if (elemType.isF16())
                return consts.f16(std::numeric_limits<float>::infinity());
            if (elemType.isInteger(32))
                return consts.i32(std::numeric_limits<int>::infinity());
        }
        llvm_unreachable("unknown neutral element");
    }

    LogicalResult matchAndRewrite(ReduceL2Op op, RewriterAdaptor& adaptor) const override
    {
        auto [srcReg, dstReg, accReg, acc0Reg] = adaptor.createRegTensors<4>(vfInfo.elemType);
        ascir::ConstantOpBuilder consts(*adaptor);
        Value neutral = getNeutralElement<ReduceL2Op>(consts, vfInfo.elemType);
        adaptor.create<ascendc::DuplicateScalarRegOp>(accReg, neutral);
        Value maskAll = adaptor.createMaskOp(vfInfo.elemType);
        Value repeatTimes =
            adaptor.create<arith::DivSIOp>(adaptor->getIndexType(), vfInfo.calCount, vfInfo.oneRepeatSize);
        Value calCount = adaptor.createUI32Variable(vfInfo.calCount);
        auto loop = adaptor.create<ascvf::VFForOp>(repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());
        adaptor.updateMaskOp(calCount, vfInfo.elemType);
        Value mulOp = adaptor.create<arith::MulIOp>(loop.getInductionVar(), vfInfo.oneRepeatSize);
        adaptor.create<ascvf::LoadOp>(srcReg, op.getSrc(), mulOp);
        adaptor.create<AccumulateRegOp>(accReg, accReg, srcReg, maskAll);
        adaptor->setInsertionPointAfter(loop);
        Value remOp = adaptor.create<arith::RemSIOp>(vfInfo.calCount, vfInfo.oneRepeatSize);
        Value cmpOp = adaptor.create<arith::CmpIOp>(arith::CmpIPredicate::ne, remOp, consts.index(0));
        auto ifOp = adaptor->create<scf::IfOp>(adaptor->getUnknownLoc(), cmpOp, false);
        adaptor->setInsertionPointToStart(ifOp.getBody());
        adaptor.create<ascendc::DuplicateScalarRegOp>(acc0Reg, neutral);
        Value lastIter = adaptor.create<arith::MulIOp>(repeatTimes, vfInfo.oneRepeatSize);
        adaptor.create<ascvf::LoadOp>(srcReg, op.getSrc(), lastIter);
        Value tailMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        adaptor.create<ascendc::SelectRegOp>(acc0Reg, srcReg, acc0Reg, tailMask);
        adaptor.create<AccumulateRegOp>(accReg, accReg, acc0Reg, maskAll);
        adaptor->setInsertionPointAfter(ifOp);
        adaptor.create<ReduceRegOp>(dstReg, accReg, maskAll);
        Value maskOne = adaptor.createMaskOp(vfInfo.elemType, ascendc::MaskPattern::VL1);
        adaptor.create<ascvf::StoreOp>(op.getDst(), consts.index(0), dstReg, maskOne);
        adaptor->eraseOp(op);
        return success();
    }
};

struct ConvertDuplicateL2 : public ConvertOp<ascendc::DuplicateL2Op> {
    using ConvertOp::ConvertOp;

    LogicalResult matchAndRewrite(ascendc::DuplicateL2Op op, RewriterAdaptor& adaptor) const override
    {
        auto [srcReg, tmpReg] = adaptor.createRegTensors<2>(vfInfo.elemType);
        Value zero = adaptor.create<arith::ConstantIndexOp>(0);
        adaptor.create<ascvf::LoadOp>(srcReg, op.getScalar(), zero);
        Value maskAll = adaptor.createMaskOp(vfInfo.elemType);
        adaptor.create<ascendc::DuplicateRegOp>(tmpReg, srcReg, maskAll);
        Value calCount = adaptor.createUI32Variable(vfInfo.calCount);
        auto loop = adaptor.create<ascvf::VFForOp>(vfInfo.repeatTimes);
        adaptor->setInsertionPoint(loop.getBody()->getTerminator());
        Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        Value mulOp = adaptor.create<arith::MulIOp>(loop.getInductionVar(), vfInfo.oneRepeatSize);
        adaptor.create<ascvf::StoreOp>(op.getDst(), mulOp, tmpReg, updateMask);
        adaptor->eraseOp(op);
        return success();
    }
};

LogicalResult convertToReg(ascvf::VecScopeOp vecScopeOp)
{
    VFInfo vfInfo(vecScopeOp);
    MLIRContext* context = vecScopeOp.getContext();
    ConversionTarget target(*context);
    target.addDynamicallyLegalDialect<ascendc::AscendCDialect>([](Operation* op) {
        return llvm::none_of(op->getOperandTypes(), [](Type type) { return isa<ascendc::LocalTensorType>(type); });
    });
    target.addLegalDialect<arith::ArithDialect, ascvf::AscVFDialect, emitasc::EmitAscDialect, scf::SCFDialect>();
    RewritePatternSet patterns(context);
    patterns.add<
        // BinaryOp
        ConvertBinaryL2<ascendc::AddL2Op, ascendc::AddRegOp>, ConvertBinaryL2<ascendc::AndL2Op, ascendc::AndRegOp>,
        ConvertBinaryL2<ascendc::DivL2Op, ascendc::DivRegOp>, ConvertBinaryL2<ascendc::MaxL2Op, ascendc::MaxRegOp>,
        ConvertBinaryL2<ascendc::MinL2Op, ascendc::MinRegOp>, ConvertBinaryL2<ascendc::MulL2Op, ascendc::MulRegOp>,
        ConvertBinaryL2<ascendc::MulAddDstL2Op, ascendc::MulAddDstRegOp>,
        ConvertBinaryL2<ascendc::OrL2Op, ascendc::OrRegOp>, ConvertBinaryL2<ascendc::PreluL2Op, ascendc::PreluRegOp>,
        ConvertBinaryL2<ascendc::SubL2Op, ascendc::SubRegOp>,
        // UnaryOp
        ConvertUnaryL2<ascendc::AbsL2Op, ascendc::AbsRegOp>, ConvertUnaryL2<ascendc::ExpL2Op, ascendc::ExpRegOp>,
        ConvertUnaryL2<ascendc::LnL2Op, ascendc::LnRegOp>, ConvertUnaryL2<ascendc::NegL2Op, ascendc::NegRegOp>,
        ConvertUnaryL2<ascendc::NotL2Op, ascendc::NotRegOp>, ConvertUnaryL2<ascendc::ReluL2Op, ascendc::ReluRegOp>,
        ConvertUnaryL2<ascendc::SqrtL2Op, ascendc::SqrtRegOp>,
        // VecScalarOp
        ConvertVecScalarL2<ascendc::LeakyReluL2Op, ascendc::LeakyReluRegOp>,
        ConvertVecScalarL2<ascendc::ShiftLeftL2Op, ascendc::ShiftLeftsRegOp>,
        ConvertVecScalarL2<ascendc::ShiftRightL2Op, ascendc::ShiftRightsRegOp>,
        // VecScalarWithDuplicate
        ConvertVecScalarWithDuplicateL2<ascendc::AddsL2Op, ascendc::AddRegOp>,
        ConvertVecScalarWithDuplicateL2<ascendc::DivsL2Op, ascendc::DivRegOp>,
        ConvertVecScalarWithDuplicateL2<ascendc::MaxsL2Op, ascendc::MaxRegOp>,
        ConvertVecScalarWithDuplicateL2<ascendc::MinsL2Op, ascendc::MinRegOp>,
        ConvertVecScalarWithDuplicateL2<ascendc::MulsL2Op, ascendc::MulRegOp>,
        ConvertVecScalarWithDuplicateL2<ascendc::SubsL2Op, ascendc::SubRegOp>,
        // Reduce
        ConvertReduceL2<ascendc::ReduceMaxL2Op, ascendc::MaxRegOp, ascendc::ReduceMaxRegOp>,
        ConvertReduceL2<ascendc::ReduceSumL2Op, ascendc::AddRegOp, ascendc::ReduceSumRegOp>,
        ConvertReduceL2<ascendc::ReduceMinL2Op, ascendc::MinRegOp, ascendc::ReduceMinRegOp>,
        // Duplicate
        ConvertDuplicateL2>(context, vfInfo);
    return applyPartialConversion(vecScopeOp, target, std::move(patterns));
}

ascvf::VecScopeOp wrapInVecScope(ascvf::VFGroupOp op)
{
    OpBuilder builder(op);
    auto vecScope = builder.create<ascvf::VecScopeOp>(builder.getUnknownLoc());
    op.getBody()->moveBefore(&vecScope.getRegion(), vecScope.getRegion().end());
    builder.createBlock(&op.getRegion(), op.getRegion().end());
    auto yield = builder.create<ascvf::YieldOp>(builder.getUnknownLoc());
    vecScope->moveBefore(yield);
    return vecScope;
}

struct LowerToRegPass : public ascvf::impl::LowerToRegBase<LowerToRegPass> {
    void runOnOperation() override
    {
        getOperation().walk([this](ascvf::VFGroupOp op) {
            auto vecScope = wrapInVecScope(op);
            if (convertToReg(vecScope).failed())
                signalPassFailure();
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createLowerToRegPass() { return std::make_unique<LowerToRegPass>(); }
