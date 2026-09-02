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
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/EmitAsc/Utils/InitStructBuilder.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERASCTILETOBASIC
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

ascendc::CMPMODE getCmpMode(asctile::CompareMode mode)
{
    switch (mode) {
        case asctile::CompareMode::EQ:
            return ascendc::CMPMODE::EQ;
        case asctile::CompareMode::NE:
            return ascendc::CMPMODE::NE;
        case asctile::CompareMode::LT:
            return ascendc::CMPMODE::LT;
        case asctile::CompareMode::LE:
            return ascendc::CMPMODE::LE;
        case asctile::CompareMode::GT:
            return ascendc::CMPMODE::GT;
        case asctile::CompareMode::GE:
            return ascendc::CMPMODE::GE;
    }
    llvm_unreachable("unexpected cmpmode");
}

struct ConvertRelu : ConvertOp<asctile::ReluOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::ReluOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        Value src = rewriter.getRemappedValue(op.getOperand());
        rewriter.create<ascendc::ReluL2Op>(loc, dst, src, consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertCast : ConvertOp<asctile::CastOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    static FailureOr<ascendc::RoundMode> inferRoundMode(Type srcType, Type dstType)
    {
        bool srcIsInt = isa<IntegerType>(srcType);
        bool dstIsFloat = isa<FloatType>(dstType);
        bool srcIsFloat = isa<FloatType>(srcType);
        bool dstIsInt = isa<IntegerType>(dstType);
        if (srcIsInt && dstIsFloat) {
            unsigned srcWidth = cast<IntegerType>(srcType).getWidth();
            unsigned dstWidth = cast<FloatType>(dstType).getWidth();
            if ((srcWidth == 8 && dstWidth == 16) || (srcWidth == 16 && dstWidth == 32)) {
                return ascendc::RoundMode::CAST_NONE;
            }
            return ascendc::RoundMode::CAST_TRUNC;
        }
        if (srcIsFloat && dstIsInt)
            return ascendc::RoundMode::CAST_TRUNC;
        if (srcIsFloat && dstIsFloat) {
            unsigned srcWidth = cast<FloatType>(srcType).getWidth();
            unsigned dstWidth = cast<FloatType>(dstType).getWidth();
            if (srcWidth > dstWidth)
                return ascendc::RoundMode::CAST_RINT;
            if (srcWidth == 16 && dstWidth == 16 && srcType != dstType)
                return ascendc::RoundMode::CAST_RINT;
            return ascendc::RoundMode::CAST_NONE;
        }
        return ascendc::RoundMode::CAST_NONE;
    }

    static bool isRoundModeSupported(asctile::RoundMode mode, Type srcType, Type dstType)
    {
        if (mode == asctile::RoundMode::Default)
            return true;
        bool srcIsInt = isa<IntegerType>(srcType);
        bool srcIsFloat = isa<FloatType>(srcType);
        bool dstIsInt = isa<IntegerType>(dstType);
        bool dstIsFloat = isa<FloatType>(dstType);
        if (srcIsInt && dstIsInt)
            return mode == asctile::RoundMode::NoRound;
        if (srcIsFloat && dstIsFloat) {
            if (isa<Float32Type>(srcType) && isa<Float16Type>(dstType))
                return true;
            if (isa<Float16Type>(srcType) && isa<Float32Type>(dstType))
                return mode == asctile::RoundMode::NoRound;
            if (isa<BFloat16Type>(srcType) && isa<Float32Type>(dstType))
                return mode == asctile::RoundMode::NoRound;
            if (isa<Float32Type>(srcType) && isa<Float32Type>(dstType))
                return mode != asctile::RoundMode::Odd && mode != asctile::RoundMode::NoRound;
            return mode != asctile::RoundMode::Odd && mode != asctile::RoundMode::NoRound;
        }
        if (srcIsFloat && dstIsInt) {
            unsigned dstWidth = cast<IntegerType>(dstType).getWidth();
            if (isa<Float32Type>(srcType) && (dstWidth == 16 || dstWidth == 32 || dstWidth == 64))
                return mode != asctile::RoundMode::Odd && mode != asctile::RoundMode::NoRound;
            if (isa<Float16Type>(srcType) && (dstWidth == 8 || dstWidth == 16 || dstWidth == 32))
                return mode != asctile::RoundMode::Odd && mode != asctile::RoundMode::NoRound;
            if (isa<BFloat16Type>(srcType) && dstWidth == 32)
                return mode != asctile::RoundMode::Odd && mode != asctile::RoundMode::NoRound;
            return false;
        }
        if (srcIsInt && dstIsFloat) {
            unsigned srcWidth = cast<IntegerType>(srcType).getWidth();
            if (isa<Float16Type>(dstType) && srcWidth == 8)
                return mode == asctile::RoundMode::NoRound;
            if (isa<Float32Type>(dstType) && srcWidth == 16)
                return mode == asctile::RoundMode::NoRound;
            if (isa<Float16Type>(dstType) && (srcWidth == 16 || srcWidth == 32))
                return mode != asctile::RoundMode::Odd && mode != asctile::RoundMode::NoRound;
            if (isa<Float32Type>(dstType) && (srcWidth == 32 || srcWidth == 64))
                return mode != asctile::RoundMode::Odd && mode != asctile::RoundMode::NoRound;
            return false;
        }
        return false;
    }

    static ascendc::RoundMode convertRoundMode(asctile::RoundMode mode)
    {
        return static_cast<ascendc::RoundMode>(static_cast<uint32_t>(mode));
    }

    LogicalResult matchAndRewrite(asctile::CastOp op, ConvertRewriter& rewriter) const override
    {
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        ascir::ConstantOpBuilder consts(rewriter);
        Value src = rewriter.getRemappedValue(op.getIn());
        Type srcType = getElementTypeOrSelf(src);
        Type dstType = getElementTypeOrSelf(dst);
        ascendc::RoundMode roundMode;
        auto roundModeOpt = op.getRoundMode();
        if (roundModeOpt == asctile::RoundMode::Default) {
            roundMode = *inferRoundMode(srcType, dstType);
        } else {
            if (!isRoundModeSupported(roundModeOpt, srcType, dstType)) {
                return op.emitError() << "round_mode " << asctile::stringifyRoundMode(roundModeOpt)
                                      << " is not supported for cast from " << srcType << " to " << dstType;
            }
            roundMode = convertRoundMode(roundModeOpt);
        }
        rewriter.create<ascendc::CastL2Op>(loc, dst, src, roundMode, consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertMatmul : ConvertOp<asctile::MatmulOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::MatmulOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto dst = createTensorOp(rewriter, loc, op.getType());
        auto matrixA = rewriter.getRemappedValue(op.getMatrixA());
        auto matrixB = rewriter.getRemappedValue(op.getMatrixB());
        auto matrixATensorShape = cast<ascendc::LocalTensorType>(matrixA.getType()).getShape();
        auto matrixBTensorShape = cast<ascendc::LocalTensorType>(matrixB.getType()).getShape();
        assert(matrixATensorShape.size() == 2 && "matrix must be have dim = 2");
        assert(matrixBTensorShape.size() == 2 && "matrix must be have dim = 2");
        if (op.getHf32()) {
            rewriter.create<ascendc::SetHF32ModeOp>(loc, consts.i1(true));
            rewriter.create<ascendc::SetHF32TransModeOp>(loc, consts.i1(true));
        }
        auto initMmadParams = emitasc::InitStructBuilder(rewriter.getType<ascendc::MmadParamsType>())
                                  .addField("m", consts.i32(matrixATensorShape[0]))
                                  .addField("n", consts.i32(matrixBTensorShape[1]))
                                  .addField("k", consts.i32(matrixBTensorShape[0]));
        if (auto bias = op.getBias()) {
            initMmadParams.addField("cmatrixInitVal", consts.i1(false)).addField("cmatrixSource", consts.i1(true));
            Value mmadParams = initMmadParams.create(rewriter, loc);
            rewriter.create<ascendc::MmadWithBiasOp>(
                loc, dst, matrixA, matrixB, rewriter.getRemappedValue(bias), mmadParams);
        } else {
            Value mmadParams = initMmadParams.create(rewriter, loc);
            rewriter.create<ascendc::MmadOp>(loc, dst, matrixA, matrixB, mmadParams);
        }
        rewriter.replaceOp(op, dst);
        if (op.getHf32())
            rewriter.create<ascendc::SetHF32ModeOp>(loc, consts.i1(false));
        return success();
    }
};

template <typename TileOp, typename L2Op>
struct ConvertToL2 : ConvertOp<TileOp> {
    using ConvertOp<TileOp>::ConvertOp;
    using ConvertOp<TileOp>::calCount;
    using ConvertOp<TileOp>::createTensorOp;

    LogicalResult matchAndRewrite(TileOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        Value lhs = rewriter.getRemappedValue(op->getOperand(0));
        Value rhs = rewriter.getRemappedValue(op->getOperand(1));
        rewriter.create<L2Op>(loc, dst, lhs, rhs, consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

template <typename TileOp, typename VecScalarOp, typename VectorOp = void>
struct ConvertVecScalarToL2 : ConvertOp<TileOp> {
    using ConvertOp<TileOp>::ConvertOp;
    using ConvertOp<TileOp>::calCount;
    using ConvertOp<TileOp>::createTensorOp;

    LogicalResult matchAndRewrite(TileOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();

        Value dst = createTensorOp(rewriter, loc, op.getType());
        Value lhs = rewriter.getRemappedValue(op->getOperand(0));
        Value rhs = rewriter.getRemappedValue(op->getOperand(1));
        constexpr bool requireArchC310 = !std::is_same_v<VectorOp, void>;
        if (requireArchC310 && !ascendc::isTargetArchC310(op)) {
            Value dup = createTensorOp(rewriter, loc, op.getType());
            rewriter.create<ascendc::DuplicateL2Op>(loc, dup, rhs, consts.i64(calCount(dst)));
            rewriter.create<VectorOp>(loc, dst, lhs, dup, consts.i64(calCount(dst)));
        } else {
            rewriter.create<VecScalarOp>(loc, dst, lhs, rhs, consts.i64(calCount(dst)));
        }
        rewriter.replaceOp(op, dst);

        return success();
    }
};

struct ConvertReduceAs1d : ConvertOp<asctile::ReduceAs1dOp> {
    using ConvertOp::calCount;
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    static unsigned calculateFinalTensorSize(unsigned typeSize, int64_t calCount)
    {
        unsigned elementsPerBlock = ascendc::ubBlockSize / typeSize;
        unsigned elementsPerRepeat = ascendc::repeatBlockSize / typeSize;
        unsigned firstMaxRepeat = llvm::divideCeil(calCount, elementsPerRepeat);
        return llvm::divideCeil(firstMaxRepeat, elementsPerBlock) * elementsPerBlock;
    }

    LogicalResult matchAndRewrite(asctile::ReduceAs1dOp op, ConvertRewriter& rewriter) const override
    {
        Type elemType = getElementTypeOrSelf(op.getType());
        unsigned typeSize = ascendc::getTypeSize(elemType);
        unsigned finalSize = calculateFinalTensorSize(typeSize, calCount(op.getOperand()));
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, 1, elemType);
        Value src = rewriter.getRemappedValue(op.getOperand());
        Value tmpBuff = createTensorOp(rewriter, loc, static_cast<int64_t>(finalSize), elemType);
        Value count = consts.i64(calCount(op.getOperand()));
        auto kind = op.getKind();
        if (kind == asctile::ReduceKind::Sum)
            rewriter.create<ascendc::ReduceSumL2Op>(loc, dst, src, tmpBuff, count);
        else if (kind == asctile::ReduceKind::Max)
            rewriter.create<ascendc::ReduceMaxL2Op>(loc, dst, src, tmpBuff, count, consts.i64(0));
        else if (kind == asctile::ReduceKind::Min)
            rewriter.create<ascendc::ReduceMinL2Op>(loc, dst, src, tmpBuff, count, consts.i64(0));
        else
            return op.emitOpError() << "with " << asctile::stringifyReduceKind(kind) << " is not supported";
        if (isa<asctile::LocalTensorType>(op.getType()))
            rewriter.replaceOp(op, dst);
        else
            rewriter.replaceOpWithNewOp<ascendc::LocalTensorGetValueOp>(op, elemType, dst, consts.i64(0));
        return success();
    }
};

struct ConvertMatmulAcc : ConvertOp<asctile::MatmulAccOp> {
    using ConvertOp::ConvertOp;

    LogicalResult matchAndRewrite(asctile::MatmulAccOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto dst = rewriter.getRemappedValue(op.getAcc());
        auto matrixA = rewriter.getRemappedValue(op.getMatrixA());
        auto matrixB = rewriter.getRemappedValue(op.getMatrixB());
        auto matrixATensorShape = cast<ascendc::LocalTensorType>(matrixA.getType()).getShape();
        auto matrixBTensorShape = cast<ascendc::LocalTensorType>(matrixB.getType()).getShape();
        auto accTensorShape = cast<ascendc::LocalTensorType>(dst.getType()).getShape();
        assert(matrixATensorShape.size() == 2 && "matrix must be have dim = 2");
        assert(matrixBTensorShape.size() == 2 && "matrix must be have dim = 2");
        assert(accTensorShape.size() == 2 && "accumulator must be have dim = 2");
        if (op.getHf32()) {
            rewriter.create<ascendc::SetHF32ModeOp>(loc, consts.i1(true));
            rewriter.create<ascendc::SetHF32TransModeOp>(loc, consts.i1(true));
        }
        auto params = emitasc::InitStructBuilder(rewriter.getType<ascendc::MmadParamsType>())
                          .addField("m", consts.i32(matrixATensorShape[0]))
                          .addField("n", consts.i32(matrixBTensorShape[1]))
                          .addField("k", consts.i32(matrixBTensorShape[0]));
        bool hasBias = op->hasAttrOfType<UnitAttr>(asctile::attr::hasBias);
        params.addField("cmatrixInitVal", consts.i1(!hasBias));
        params.addField("cmatrixSource", consts.i1(hasBias));
        auto mmadParams = params.create(rewriter, loc);
        rewriter.create<ascendc::MmadOp>(loc, dst, matrixA, matrixB, mmadParams);
        rewriter.eraseOp(op);
        if (op.getHf32())
            rewriter.create<ascendc::SetHF32ModeOp>(loc, consts.i1(false));
        return success();
    }
};

struct ConvertTransposeUB : ConvertOp<asctile::TransposeOp> {
    using ConvertOp::ConvertOp;

    static constexpr int64_t transDataBlockSize = 16;

    static SmallVector<int32_t> fillArray(int64_t startOffset, bool evenOffset, int64_t rowStride)
    {
        SmallVector<int32_t> result;
        for (int64_t i = 0; i < transDataBlockSize; ++i) {
            int64_t offset = 0;
            if (!evenOffset) {
                offset = i * rowStride;
            } else {
                offset = (i / 2) * rowStride + (i % 2) * ascendc::ubBlockSize;
            }
            result.push_back(static_cast<int32_t>(startOffset + offset));
        }
        return result;
    }

    LogicalResult matchAndRewrite(asctile::TransposeOp op, ConvertRewriter& rewriter) const override
    {
        if (op.getOperand().getType().getLoc() != asctile::TensorLocation::UB ||
            op.getResult().getType().getLoc() != asctile::TensorLocation::UB)
            return failure();
        int64_t elementSize = ascendc::getElementTypeSize(op.getType());
        if (elementSize != sizeof(int16_t) && elementSize != sizeof(int32_t) && elementSize != sizeof(int8_t))
            return failure();

        auto loc = op.getLoc();
        Value src = rewriter.getRemappedValue(op.getOperand());
        auto srcShape = cast<ascendc::BaseTensorType>(src.getType()).getShape();
        if (srcShape.size() != 2)
            return op.emitError("Not supporting tiles with dims greater than 2");

        ascir::ConstantOpBuilder consts(rewriter);
        Value dst = createTensorOp(rewriter, loc, op.getType());

        int64_t width = srcShape[1];
        int64_t height = srcShape[0];
        bool axis = height >= width;
        // for int16 use 16x16 block, int32 use 8x16, int8 special case 32x32
        int64_t blockWidth = ascendc::ubBlockSize / elementSize;
        int64_t blockHeight = elementSize == 1 ? ascendc::ubBlockSize : transDataBlockSize;
        int64_t loopStep = axis ? blockWidth : blockHeight;
        int64_t loopCount = ((axis ? width : height) + loopStep - 1) / loopStep;

        auto i1Type = rewriter.getI1Type();
        auto ui8Type = rewriter.getIntegerType(8, false);
        auto ui16Type = rewriter.getIntegerType(16, false);

        for (int64_t i = 0; i < loopCount; ++i) {
            int64_t srcStride = width * elementSize;
            int64_t dstStride = height * elementSize;
            int64_t srcOffset = 0;
            int64_t dstOffset = 0;
            int64_t blockCount = 0;
            int64_t dstBlockStride = 0;
            int64_t srcBlockStride = 0;
            if (axis) {
                srcOffset = i * ascendc::ubBlockSize;
                dstOffset = i * height * elementSize * blockWidth;
                blockCount = (height + blockHeight - 1) / blockHeight;
                srcBlockStride = elementSize * width * blockHeight;
                dstBlockStride = blockHeight * elementSize;
            } else {
                srcOffset = i * width * elementSize * blockHeight;
                dstOffset = i * blockHeight * elementSize;
                blockCount = (width + blockWidth - 1) / blockWidth;
                srcBlockStride = blockWidth * elementSize;
                dstBlockStride = height * elementSize * blockWidth;
            }
            assert(blockCount > 0 && blockCount <= 255);
            assert(dstBlockStride % ascendc::ubBlockSize == 0);
            assert(srcBlockStride % ascendc::ubBlockSize == 0);
            if (elementSize != sizeof(int8_t)) {
                auto srcList = fillArray(srcOffset, false, srcStride);
                auto dstList = fillArray(dstOffset, elementSize == sizeof(uint32_t), dstStride);
                Value params = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::TransDataTo5HDParamsType>(),
                    ValueRange{
                        consts.i1(false), consts.i1(false), consts.i32(blockCount),
                        consts.i16(blockCount == 1 ? 0 : dstBlockStride / ascendc::ubBlockSize),
                        consts.i16(blockCount == 1 ? 0 : srcBlockStride / ascendc::ubBlockSize)},
                    rewriter.getTypeArrayAttr({i1Type, i1Type, ui8Type, ui16Type, ui16Type}));
                rewriter.create<ascendc::TransDataTo5HDTensorOp>(loc, dst, src, dstList, srcList, params);
            } else {
                int64_t srcOffsetH = srcOffset + width * transDataBlockSize;
                int64_t dstOffsetH = dstOffset + height * transDataBlockSize;
                auto srcListLow = fillArray(srcOffset, false, srcStride);
                auto srcListHigh = fillArray(srcOffsetH, false, srcStride);
                auto dstListLow = fillArray(dstOffset, false, dstStride);
                auto dstListHigh = fillArray(dstOffsetH, false, dstStride);
                // Use 4 calls for each 16x16 subtile inside 32x32
                for (int i = 0; i < 4; ++i) {
                    Value dstHighHalf = consts.i1(i / 2 != 0);
                    Value srcHighHalf = consts.i1(i % 2 != 0);
                    Value params = rewriter.create<ascendc::ConstructOp>(
                        loc, rewriter.getType<ascendc::TransDataTo5HDParamsType>(),
                        ValueRange{
                            dstHighHalf, srcHighHalf, consts.i32(blockCount),
                            consts.i16(blockCount == 1 ? 0 : dstBlockStride / ascendc::ubBlockSize),
                            consts.i16(blockCount == 1 ? 0 : srcBlockStride / ascendc::ubBlockSize)},
                        rewriter.getTypeArrayAttr({i1Type, i1Type, ui8Type, ui16Type, ui16Type}));
                    rewriter.create<ascendc::TransDataTo5HDTensorOp>(
                        loc, dst, src, i % 2 == 0 ? dstListLow : dstListHigh, i / 2 == 0 ? srcListLow : srcListHigh,
                        params);
                }
            }
        }
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertBitwiseNot : ConvertOp<asctile::BitwiseNotOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::BitwiseNotOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        Value src = rewriter.getRemappedValue(op.getOperand());
        rewriter.create<ascendc::NotL2Op>(loc, dst, src, consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertCmpS : ConvertOp<asctile::CmpSOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::CmpSOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto value = rewriter.getRemappedValue(op.getValue());
        auto base = rewriter.getRemappedValue(op.getBase());
        auto srcType = cast<ShapedType>(op.getBase().getType());
        if (isa<IntegerType>(srcType.getElementType())) {
            unsigned bitWidth = srcType.getElementTypeBitWidth();
            if (bitWidth != 8 && bitWidth != 16 && bitWidth != 32)
                return op.emitOpError("can only be lowered with i8, i16 or i32 tensor operands");
            auto castToType = bitWidth == 32 ? rewriter.getF32Type() : rewriter.getF16Type();
            auto baseCasted = createTensorOp(rewriter, loc, srcType.getShape(), castToType);
            rewriter.create<ascendc::CastL2Op>(
                loc, baseCasted, base, ascendc::RoundMode::CAST_NONE, consts.i64(srcType.getNumElements()));
            base = baseCasted;
            value = rewriter.create<arith::SIToFPOp>(loc, castToType, value);
        }
        I1ReplacementType replType(op.getContext());
        auto dstShape = llvm::divideCeilSigned(srcType.getNumElements(), replType.width);
        Value dst = createTensorOp(rewriter, loc, dstShape, replType.iType);
        dst = createReCastOp(rewriter, loc, dst, dstShape, replType.uiType);
        auto mode = getCmpMode(op.getCmpMode());
        Value zero = consts.i64(0);
        rewriter.create<ascendc::CompareScalarL0Op>(
            loc, dst, base, value, mode, zero, zero,
            rewriter.create<ascendc::ConstructOp>(loc, rewriter.getType<ascendc::UnaryRepeatParamsType>()));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct LowerAscTileToBasicPass : public asclower::impl::LowerAscTileToBasicBase<LowerAscTileToBasicPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addIllegalOp<
            //
            asctile::LeakyReluOp, asctile::ReluOp, asctile::CastOp, asctile::MatmulOp, asctile::AddSOp, asctile::SubSOp,
            asctile::MulSOp, asctile::DivSOp, asctile::MinSOp, asctile::MaxSOp, asctile::ShLSOp, asctile::ShRSOp,
            asctile::CmpSOp, asctile::ReduceAs1dOp, asctile::MatmulAccOp, asctile::TransposeOp
            //
            >();
        target.addLegalDialect<ascendc::AscendCDialect, arith::ArithDialect, emitasc::EmitAscDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertRelu, ConvertCast, ConvertMatmul, ConvertBitwiseNot, ConvertMatmulAcc, ConvertReduceAs1d,
            ConvertTransposeUB, ConvertCmpS, ConvertToL2<asctile::AddSOp, ascendc::AddsL2Op>,
            ConvertVecScalarToL2<asctile::SubSOp, ascendc::SubsL2Op, ascendc::SubL2Op>,
            ConvertToL2<asctile::MulSOp, ascendc::MulsL2Op>, ConvertToL2<asctile::LeakyReluOp, ascendc::LeakyReluL2Op>,
            ConvertVecScalarToL2<asctile::DivSOp, ascendc::DivsL2Op, ascendc::DivL2Op>,
            ConvertToL2<asctile::MinSOp, ascendc::MinsL2Op>, ConvertToL2<asctile::MaxSOp, ascendc::MaxsL2Op>,
            ConvertToL2<asctile::ShLSOp, ascendc::ShiftLeftL2Op>, ConvertToL2<asctile::ShRSOp, ascendc::ShiftRightL2Op>
            //
            >(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerAscTileToBasicPass()
{
    return std::make_unique<LowerAscTileToBasicPass>();
}
