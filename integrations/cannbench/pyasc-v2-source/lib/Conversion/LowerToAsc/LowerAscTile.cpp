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
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/EmitAsc/Utils/InitStructBuilder.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERASCTILE
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

constexpr int oneBlkFloatNum = 8;
constexpr int totalUbSize = 256 * 1024;
constexpr int maxRepeat = 255;
constexpr int basicBlkBslength = 8;
constexpr int halfSizeInByte = 2;

std::pair<int, int> unpack2DShape(ArrayRef<int64_t> shape)
{
    assert(shape.size() == 1 || shape.size() == 2);
    return {shape.size() == 2 ? shape[0] : 1, shape.back()};
}

bool check1D2DShape(Operation* op, ArrayRef<int64_t> shape)
{
    if (shape.size() != 1 && shape.size() != 2) {
        op->emitError() << "invalid dimension of input tensor";
        return false;
    }
    return true;
}

struct ConvertTensor : public ConvertOp<asctile::TensorOp> {
    using ConvertOp::ConvertOp;

    LogicalResult matchAndRewrite(asctile::TensorOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        Value tensor = rewriter.create<ascendc::GlobalTensorOp>(loc, typeConverter->convertType(op.getType()));
        rewriter.create<ascendc::GlobalTensorSetGlobalBufferOp>(loc, tensor, op.getBase(), /*size*/ Value{});
        rewriter.replaceOp(op, tensor);
        return success();
    }
};

struct ConvertAccumulator : ConvertOp<asctile::AccumulatorOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::AccumulatorOp op, ConvertRewriter& rewriter) const override
    {
        auto type = op.getType();
        assert(type.getLoc() == asctile::TensorLocation::L0C && "accumulator should be have tensor location L0C");
        auto loc = op.getLoc();
        auto dst = createTensorOp(rewriter, loc, type);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertSoftmax : ConvertOp<asctile::SoftmaxOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    static int64_t getSoftMaxMinTmpSize(int64_t srcM, int64_t srcK, int64_t dataTypeSize)
    {
        // Formula from
        // https://gitcode.com/cann/asc-devkit/blob/b0fb6c8f89686dadc9b99fd114d9635eb5564dbf/impl/adv_api/detail/activation/softmax/regbase/3510/softmax_impl.h#L1188
        constexpr int64_t softmaxFloatSize = 4;
        int64_t b32DataNumPerBlock = 32 / dataTypeSize;
        int64_t offset = ((srcM + b32DataNumPerBlock - 1) / b32DataNumPerBlock) * b32DataNumPerBlock;
        return (offset * 2 + srcM * srcK) * softmaxFloatSize;
    }

    LogicalResult matchAndRewrite(asctile::SoftmaxOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto tensorType = op.getType();
        auto shape = tensorType.getShape();
        if (!check1D2DShape(op, shape))
            return failure();
        auto [height, width] = unpack2DShape(shape);
        auto src = rewriter.getRemappedValue(op.getOperand());
        auto dst = createTensorOp(rewriter, loc, tensorType);
        auto elemType = tensorType.getElementType();
        width = static_cast<int>(llvm::alignTo(width, ascendc::ubBlockSize / ascendc::getElementTypeSize(tensorType)));
        int64_t bufferSize = getSoftMaxMinTmpSize(height, width, ascendc::getElementTypeSize(tensorType));
        auto sharedBufTensor = createTensorOp(rewriter, loc, bufferSize, rewriter.getIntegerType(8, false));
        auto tiling = rewriter.create<ascendc::ConstructOp>(loc, rewriter.getType<ascendc::SoftMaxTilingType>());
        Value shapeInfo = emitasc::InitStructBuilder(rewriter.getType<ascendc::SoftMaxShapeInfoType>())
                              .addField("srcM", consts.i32(height))
                              .addField("srcK", consts.i32(width))
                              .addField("oriSrcM", consts.i32(height))
                              .addField("oriSrcK", consts.i32(width))
                              .create(rewriter, loc);
        rewriter.create<ascendc::SoftMaxOp>(
            loc, false, false, false, dst, Value{}, Value{}, src, sharedBufTensor, tiling, shapeInfo);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertRmsNorm : ConvertOp<asctile::RmsNormOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    struct RmsNormTiling {
        int bLength;
        int sLength;
        int hLength;
        int originalHLength;
        float reciprocalOfHLength;
        int mainBshLength;
        int mainBsLength;
        int mainBsLengthAlign;
        int loopRound;
        int tailBshLength;
        int inputTailPos;
        int tailBsLength;
    };

    static int alignToBlock(const int inputValue, const int typeSize)
    {
        int alignUnit = static_cast<int>(ascendc::ubBlockSize) / typeSize;
        return (inputValue + alignUnit - 1) / alignUnit * alignUnit;
    }

    // TODO: Refactor
    static RmsNormTiling getRmsNormTiling(ArrayRef<int64_t> shape, bool isBasicBlock, int typeSize)
    {
        auto [bLength, inHLength] = unpack2DShape(shape);
        auto sLength = 1;
        auto hLength = alignToBlock(inHLength, typeSize);
        auto bshLength = bLength * sLength * hLength;
        auto originalHLength = inHLength;
        auto reciprocalOfHLength = 1.0F / static_cast<float>(originalHLength);
        auto oneTmpSize = totalUbSize / typeSize;
        auto alignBsLength = oneBlkFloatNum;
        auto halfCoeff = (typeSize == sizeof(float) ? 1 : 2);
        while (oneTmpSize > alignBsLength * hLength * halfCoeff + alignBsLength) {
            alignBsLength += oneBlkFloatNum;
        }
        alignBsLength = alignBsLength == oneBlkFloatNum ? oneBlkFloatNum : alignBsLength - oneBlkFloatNum;
        oneTmpSize =
            (typeSize == halfSizeInByte) ? (oneTmpSize - alignBsLength) / halfCoeff : (oneTmpSize - alignBsLength);
        auto inputXSize = bLength * sLength * hLength;
        oneTmpSize = std::min(oneTmpSize, inputXSize);
        auto bsLength = oneTmpSize / hLength;
        if (isBasicBlock) {
            bsLength = bsLength < basicBlkBslength ? 1 : bsLength / basicBlkBslength * basicBlkBslength;
        } else if (bsLength > maxRepeat) {
            bsLength = maxRepeat;
        }
        oneTmpSize = bsLength * hLength;
        auto mainBshLength = oneTmpSize;
        auto mainBsLength = oneTmpSize / hLength;
        auto mainBsLengthAlign = alignToBlock(oneTmpSize / hLength, typeSize);
        auto loopRound = inputXSize / oneTmpSize;
        auto inputTailSize = inputXSize % oneTmpSize;
        auto tailBshLength = inputTailSize;
        auto inputTailPos = inputXSize - inputTailSize;
        auto tailBsLength = inputTailSize / hLength;
        return {bLength,      sLength,           hLength,   originalHLength, reciprocalOfHLength, mainBshLength,
                mainBsLength, mainBsLengthAlign, loopRound, tailBshLength,   inputTailPos,        tailBsLength};
    }

    LogicalResult matchAndRewrite(asctile::RmsNormOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto tensorType = op.getType();
        auto shape = tensorType.getShape();
        if (!check1D2DShape(op, shape))
            return failure();
        auto src = rewriter.getRemappedValue(op.getInput());
        auto gammaTensor = rewriter.getRemappedValue(op.getGamma());
        auto epsilon = rewriter.getRemappedValue(op.getEpsilon());
        auto dst = createTensorOp(rewriter, loc, tensorType);
        constexpr bool isBasicBlock = false;
        auto typeSize = static_cast<int>(ascendc::getElementTypeSize(tensorType));
        RmsNormTiling tilingStruct = getRmsNormTiling(shape, isBasicBlock, typeSize);
        auto sharedBufTensor =
            createTensorOp(rewriter, loc, ascendc::getTypeSize(tensorType) * 2, rewriter.getIntegerType(8, false));
        Value tiling = emitasc::InitStructBuilder(rewriter.getType<ascendc::RmsNormTilingType>())
                           .addField("bLength", consts.i32(tilingStruct.bLength))
                           .addField("sLength", consts.i32(tilingStruct.sLength))
                           .addField("hLength", consts.i32(tilingStruct.hLength))
                           .addField("originalHLength", consts.i32(tilingStruct.originalHLength))
                           .addField("reciprocalOfHLength", consts.f32(tilingStruct.reciprocalOfHLength))
                           .addField("mainBshLength", consts.i32(tilingStruct.mainBshLength))
                           .addField("mainBsLength", consts.i32(tilingStruct.mainBsLength))
                           .addField("mainBsLengthAlign", consts.i32(tilingStruct.mainBsLengthAlign))
                           .addField("loopRound", consts.i32(tilingStruct.loopRound))
                           .addField("tailBshLength", consts.i32(tilingStruct.tailBshLength))
                           .addField("inputTailPos", consts.i32(tilingStruct.inputTailPos))
                           .addField("tailBsLength", consts.i32(tilingStruct.tailBsLength))
                           .create(rewriter, loc);
        rewriter.create<ascendc::RmsNormOp>(loc, isBasicBlock, dst, src, gammaTensor, epsilon, tiling, sharedBufTensor);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertLayerNorm : ConvertOp<asctile::LayerNormOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    static std::pair<Value, int> createLayerNormSeparateTiling(
        ConvertRewriter& rewriter, Location loc, ArrayRef<int64_t> shape, int typeSize, int rLengthWithPadding,
        int gammaTypeSize)
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto [aLength, rLength] = unpack2DShape(shape);
        auto inputXSize = aLength * rLengthWithPadding;
        auto meanVarSize = aLength;
        auto rValueBack = 1.0F / static_cast<float>(rLength);
        uint32_t k = 0;
        uint32_t temp = rLength;
        while (temp > 1) {
            temp >>= 1;
            k++;
        }
        auto k2Rec = 1.0F / static_cast<float>(1 << k);
        auto k2RRec = static_cast<float>(1 << k) / static_cast<float>(rLength);
        uint32_t rHeadLength = rLength <= 128 ? rLength : 1 << k;
        constexpr uint32_t sregLower = 64;
        uint32_t halfAddCount = (rHeadLength / 2 + sregLower - 1) / sregLower;
        uint32_t halfAddRepeatTimes = 0;
        uint32_t tempCount = halfAddCount;
        while (tempCount > sregLower) {
            tempCount = (tempCount + sregLower - 1) / sregLower;
            halfAddRepeatTimes++;
        }
        auto halfCoeff = (typeSize == sizeof(float) ? 1 : 2);
        auto tmpBufSize = totalUbSize / static_cast<int>(sizeof(float));
        auto numberOfTmpBuf = 3;
        auto oneTmpSize = tmpBufSize / numberOfTmpBuf;
        if (typeSize == halfSizeInByte) {
            oneTmpSize = (oneTmpSize * 2) / halfCoeff;
        }
        auto inputXSizeFloat = (typeSize == halfSizeInByte) ? inputXSize / 2 : inputXSize;
        oneTmpSize = std::min(oneTmpSize, inputXSizeFloat);
        auto aCurLength = std::min(static_cast<int>(oneTmpSize / rLengthWithPadding), maxRepeat);
        if (aCurLength == 0) {
            aCurLength = 1;
        }
        oneTmpSize = aCurLength * rLengthWithPadding;
        auto oneTmpSizeOriginal = (typeSize == halfSizeInByte) ? oneTmpSize * 2 : oneTmpSize;
        auto arCurLength = oneTmpSize;
        auto inputRoundSize = oneTmpSizeOriginal;
        auto loopRound = inputXSize / oneTmpSizeOriginal;
        auto inputTailSize = inputXSize % oneTmpSizeOriginal;
        auto inputTailPos = inputXSize - inputTailSize;
        auto meanVarRoundSize = aCurLength;
        auto meanVarTailSize = inputTailSize / rLengthWithPadding;
        auto meanVarTailPos = meanVarSize - meanVarTailSize;
        auto firstTmpStartPos = 0;
        auto secondTmpStartPos = oneTmpSize;
        auto thirdTmpStartPos = oneTmpSize * 2;
        auto varianceTmpTensorPos = oneTmpSize * numberOfTmpBuf;
        auto varianceTmpTensorSize = aCurLength;
        auto stage1FloatSize = oneTmpSize * numberOfTmpBuf + aCurLength;
        int stage2FloatSize =
            gammaTypeSize == sizeof(float) ? 2 * aLength * rLengthWithPadding : (2 * aLength + 2) * rLengthWithPadding;
        auto sharedBufFloatSize = std::max(stage1FloatSize, stage2FloatSize);
        auto tilingValue = emitasc::InitStructBuilder(rewriter.getType<ascendc::LayerNormSeparateTilingType>())
                               .addField("aLength", consts.i32(aLength))
                               .addField("rLength", consts.i32(rLength))
                               .addField("halfAddRepeatTimes", consts.i32(halfAddRepeatTimes))
                               .addField("rHeadLength", consts.i32(rHeadLength))
                               .addField("k2Rec", consts.f32(k2Rec))
                               .addField("k2RRec", consts.f32(k2RRec))
                               .addField("inputXSize", consts.i32(inputXSize))
                               .addField("meanVarSize", consts.i32(meanVarSize))
                               .addField("numberOfTmpBuf", consts.i32(numberOfTmpBuf))
                               .addField("varianceTmpTensorPos", consts.i32(varianceTmpTensorPos))
                               .addField("varianceTmpTensorSize", consts.i32(varianceTmpTensorSize))
                               .addField("tmpBufSize", consts.i32(sharedBufFloatSize))
                               .addField("oneTmpSize", consts.i32(oneTmpSize))
                               .addField("firstTmpStartPos", consts.i32(firstTmpStartPos))
                               .addField("secondTmpStartPos", consts.i32(secondTmpStartPos))
                               .addField("thirdTmpStartPos", consts.i32(thirdTmpStartPos))
                               .addField("loopRound", consts.i32(loopRound))
                               .addField("inputRoundSize", consts.i32(inputRoundSize))
                               .addField("inputTailSize", consts.i32(inputTailSize))
                               .addField("inputTailPos", consts.i32(inputTailPos))
                               .addField("meanVarRoundSize", consts.i32(meanVarRoundSize))
                               .addField("meanVarTailSize", consts.i32(meanVarTailSize))
                               .addField("meanVarTailPos", consts.i32(meanVarTailPos))
                               .addField("arCurLength", consts.i32(arCurLength))
                               .addField("aCurLength", consts.i32(aCurLength))
                               .addField("rValueBack", consts.f32(rValueBack))
                               .create(rewriter, loc);
        return {tilingValue, sharedBufFloatSize};
    }

    static Value createLayerNormPara(
        ConvertRewriter& rewriter, Location loc, int aLength, int rLength, int rLengthWithPadding)
    {
        ascir::ConstantOpBuilder consts(rewriter);
        return emitasc::InitStructBuilder(rewriter.getType<ascendc::LayerNormParaType>())
            .addField("aLength", consts.i32(aLength))
            .addField("rLength", consts.i32(rLength))
            .addField("rLengthWithPadding", consts.i32(rLengthWithPadding))
            .create(rewriter, loc);
    }

    LogicalResult matchAndRewrite(asctile::LayerNormOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto tensorType = cast<asctile::LocalTensorType>(op.getOutput().getType());
        auto shape = tensorType.getShape();
        if (!check1D2DShape(op, shape))
            return failure();
        auto src = rewriter.getRemappedValue(op.getInput());
        auto gammaTensor = rewriter.getRemappedValue(op.getGamma());
        auto betaTensor = rewriter.getRemappedValue(op.getBeta());
        auto epsilon = rewriter.getRemappedValue(op.getEpsilon());
        auto dst = createTensorOp(rewriter, loc, tensorType);
        auto typeSize = static_cast<int>(ascendc::getElementTypeSize(tensorType));
        auto gammaType = cast<asctile::LocalTensorType>(op.getGamma().getType());
        auto gammaTypeSize = static_cast<int>(ascendc::getElementTypeSize(gammaType));
        auto [aLength, rLength] = unpack2DShape(shape);
        auto rLengthWithPadding = static_cast<int>(llvm::alignTo<ascendc::ubBlockSize>(rLength * typeSize) / typeSize);
        auto meanVarSize = aLength;
        auto dstMeanFloat = createTensorOp(rewriter, loc, meanVarSize, rewriter.getF32Type());
        auto dstVarRstdFloat = createTensorOp(rewriter, loc, meanVarSize, rewriter.getF32Type());
        auto [separateTiling, sharedBufFloatSize] =
            createLayerNormSeparateTiling(rewriter, loc, shape, typeSize, rLengthWithPadding, gammaTypeSize);
        Value para = createLayerNormPara(rewriter, loc, aLength, rLength, rLengthWithPadding);
        auto sharedBufSize = sharedBufFloatSize * static_cast<int>(sizeof(float));
        auto sharedBufTensor = createTensorOp(rewriter, loc, sharedBufSize, rewriter.getIntegerType(8, false));
        auto lnOp = rewriter.create<ascendc::LayerNormOp>(
            loc, dst, dstMeanFloat, dstVarRstdFloat, src, gammaTensor, betaTensor, epsilon, separateTiling, para,
            sharedBufTensor);
        lnOp.setOutputRstdAttr(op.getOutputRstdAttr());
        rewriter.replaceOp(op, {dst, dstMeanFloat, dstVarRstdFloat});
        return success();
    }
};

struct ConvertReshape : ConvertOp<asctile::ReshapeOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::ReshapeOp op, ConvertRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<ascendc::LocalTensorReinterpretCastOp>(
            op, typeConverter->convertType(op.getType()), rewriter.getRemappedValue(op.getIn()));
        return success();
    }
};

struct ConvertBroadcast : ConvertOp<asctile::BroadcastOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::BroadcastOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto dstType = op.getResult().getType();
        auto src = rewriter.getRemappedValue(op.getOperand());
        auto dst = createTensorOp(rewriter, loc, dstType.getShape(), dstType.getElementType());
        auto srcType = op.getOperand().getType();
        if (srcType.getNumElements() == 1) {
            Value dupSrc = src;
            if (!ascendc::isTargetArchC310(op)) {
                dupSrc =
                    rewriter.create<ascendc::LocalTensorGetValueOp>(loc, srcType.getElementType(), src, consts.i64(0));
            }
            rewriter.create<ascendc::DuplicateL2Op>(loc, dst, dupSrc, consts.i64(0));
            rewriter.replaceOp(op, dst);
            return success();
        }
        auto srcShapeVec = srcType.getShape();
        auto dstShapeVec = dstType.getShape();
        if (srcShapeVec.size() > dstShapeVec.size() || srcShapeVec.empty() || dstShapeVec.empty())
            return op.emitError("Incompatible tensor shapes for Broadcast: [")
                .append(srcShapeVec)
                .append("] and [")
                .append(dstShapeVec)
                .append("]");
        SmallVector<Value> dstShape, srcShape;
        // Workaround: when dim<3 old Broadcast algorithm used (it need aligned data)
        for (size_t i = 0; i < 3U - dstShapeVec.size(); ++i) {
            dstShape.push_back(consts.i32(1));
        }
        // Pad srcShape with `1` to match dstShape
        for (size_t i = srcShapeVec.size(); i < dstShapeVec.size() + dstShape.size(); ++i) {
            srcShape.push_back(consts.i32(1));
        }
        for (int64_t i : srcShapeVec) {
            srcShape.push_back(consts.i32(i));
        }
        for (int64_t i : dstShapeVec) {
            dstShape.push_back(consts.i32(i));
        }
        assert(srcShape.size() == dstShape.size());
        rewriter.create<ascendc::BroadcastOp>(loc, dst, src, dstShape, srcShape);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertReduce : ConvertOp<asctile::ReduceOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    static std::optional<ascendc::ReducePattern> findPattern(size_t length, uint64_t mask)
    {
        struct Pattern {
            size_t length;
            uint64_t mask;
            ascendc::ReducePattern value;
        };
        static constexpr Pattern patterns[] = {
            {1, 0b1, ascendc::ReducePattern::R},
            {2, 0b01, ascendc::ReducePattern::RA},
            {2, 0b10, ascendc::ReducePattern::AR},
            {3, 0b010, ascendc::ReducePattern::ARA},
            {3, 0b101, ascendc::ReducePattern::RAR},
            {4, 0b1010, ascendc::ReducePattern::ARAR},
            {4, 0b0101, ascendc::ReducePattern::RARA},
            {5, 0b10101, ascendc::ReducePattern::RARAR},
            {5, 0b01010, ascendc::ReducePattern::ARARA},
            {6, 0b101010, ascendc::ReducePattern::RARARA},
            {6, 0b010101, ascendc::ReducePattern::ARARAR},
            {7, 0b1010101, ascendc::ReducePattern::RARARAR},
            {7, 0b0101010, ascendc::ReducePattern::ARARARA},
            {8, 0b10101010, ascendc::ReducePattern::RARARARA},
            {8, 0b01010101, ascendc::ReducePattern::ARARARAR},
            {9, 0b010101010, ascendc::ReducePattern::ARARARARA},
        };
        for (const auto& p : patterns) {
            if (p.length == length && p.mask == mask)
                return p.value;
        }
        return std::nullopt;
    }

    static std::pair<SmallVector<int64_t>, std::optional<ascendc::ReducePattern>> getReductionParams(
        ArrayRef<int64_t> tensorShape, ArrayRef<int64_t> dims)
    {
        if (tensorShape.empty() || dims.empty())
            return {};
        SmallVector<bool> reduceDims(tensorShape.size(), false);
        for (auto dim : dims)
            reduceDims[dim] = true;
        SmallVector<int64_t> shape;
        bool reduceCurrent = reduceDims[0];
        int64_t accum = 1;
        uint64_t mask = 0;
        for (size_t i = 0; i <= tensorShape.size(); ++i) {
            if (i < tensorShape.size() && reduceDims[i] == reduceCurrent) {
                accum *= tensorShape[i];
                continue;
            }
            if (reduceCurrent)
                mask |= (1 << shape.size());
            shape.push_back(accum);
            if (i < tensorShape.size()) {
                accum = tensorShape[i];
                reduceCurrent = reduceDims[i];
            }
        }
        return std::pair(shape, findPattern(shape.size(), mask));
    }

    LogicalResult matchAndRewrite(asctile::ReduceOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        SmallVector<int64_t> reduceDims;
        for (auto attr : op.getDims()) {
            reduceDims.push_back(cast<IntegerAttr>(attr).getValue().getSExtValue());
        }
        auto srcType = op.getOperand().getType();
        auto [shape, pattern] = getReductionParams(srcType.getShape(), reduceDims);
        if (shape.empty() || !pattern)
            return emitError(loc, "Tensor of shape [")
                .append(srcType.getShape())
                .append("] have wrong reduction dimensions: ")
                .append(reduceDims);
        SmallVector<Value> srcShape;
        for (auto size : shape)
            srcShape.push_back(consts.i32(size));
        Value dst = createTensorOp(rewriter, loc, op.getType());
        Value src = rewriter.getRemappedValue(op.getOperand());
        Value tmpBuff = createTensorOp(rewriter, loc, srcType.getNumElements() * 4, rewriter.getIntegerType(8, false));
        auto kind = op.getKind();
        Operation* reduceOp = nullptr;
        bool reuseSource = op->hasAttr(asctile::attr::reuseSource);
        if (kind == asctile::ReduceKind::Sum)
            reduceOp = rewriter.create<ascendc::ReduceSumOp>(loc, dst, src, tmpBuff, srcShape, *pattern, reuseSource);
        else if (kind == asctile::ReduceKind::Max)
            reduceOp = rewriter.create<ascendc::ReduceMaxOp>(loc, dst, src, tmpBuff, srcShape, *pattern, reuseSource);
        else if (kind == asctile::ReduceKind::Min)
            reduceOp = rewriter.create<ascendc::ReduceMinOp>(loc, dst, src, tmpBuff, srcShape, *pattern, reuseSource);
        else if (kind == asctile::ReduceKind::Prod)
            reduceOp = rewriter.create<ascendc::ReduceProdOp>(loc, dst, src, tmpBuff, srcShape, *pattern, reuseSource);
        else
            return op.emitOpError() << "with " << asctile::stringifyReduceKind(kind) << " is not supported";
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertInlineVF : ConvertOp<asctile::InlineVFOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::InlineVFOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        SmallVector<Value> inputs;
        if (rewriter.getRemappedValues(op.getInputs(), inputs).failed())
            return op.emitOpError("has unsupported inputs");
        ascir::ConstantOpBuilder consts(rewriter);
        auto vfGroup = rewriter.create<ascvf::VFGroupOp>(loc, ValueRange{dst}, inputs, consts.i32(0));
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&vfGroup.getRegion().emplaceBlock());
            auto vecScope = rewriter.create<ascvf::VecScopeOp>(loc);
            {
                OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(&vecScope.getRegion().emplaceBlock());
                inputs.insert(inputs.begin(), dst);
                rewriter.create<emitasc::VerbatimOp>(loc, op.getCodeAttr(), inputs);
                rewriter.create<ascvf::YieldOp>(loc);
            }
            rewriter.create<ascvf::YieldOp>(loc);
        }
        rewriter.replaceOp(op, dst);
        return success();
    }
};

template <typename GroupOp, typename IfOp>
struct ConvertCVGroup : ConvertOp<GroupOp> {
    using ConvertOp<GroupOp>::ConvertOp;
    using ConvertOp<GroupOp>::createTensorOp;
    using ConvertOp<GroupOp>::typeConverter;

    LogicalResult matchAndRewrite(GroupOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        SmallVector<Value> srcList;
        if (rewriter.getRemappedValues(op.getOperands(), srcList).failed())
            return op.emitOpError("has unsupported operands");
        SmallVector<Type> resultTypes;
        if (typeConverter->convertTypes(op.getResultTypes(), resultTypes).failed())
            return op.emitOpError("failed to convert result types");
        Region& srcRegion = op.getRegion();
        auto ifOp = rewriter.create<IfOp>(loc, resultTypes, srcList);
        Region& thenRegion = ifOp.getRegion();
        thenRegion.getBlocks().splice(thenRegion.end(), srcRegion.getBlocks());
        auto yieldOp = cast<asctile::YieldOp>(thenRegion.begin()->getTerminator());
        SmallVector<Value> convertedOperands;
        if (rewriter.getRemappedValues(yieldOp.getOperands(), convertedOperands).failed())
            return yieldOp.emitOpError("has unsupported operands");
        rewriter.setInsertionPoint(yieldOp);
        rewriter.replaceOpWithNewOp<ascendc::YieldOp>(yieldOp, convertedOperands);
        rewriter.replaceOp(op, ifOp.getResults());
        return success();
    }
};

struct LowerAscTilePass : public asclower::impl::LowerAscTileBase<LowerAscTilePass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addIllegalOp<
            //
            asctile::TensorOp, asctile::AccumulatorOp, asctile::SoftmaxOp, asctile::ReshapeOp, asctile::BroadcastOp,
            asctile::ReduceOp, asctile::InlineVFOp, asctile::CubeGroupOp, asctile::VectorGroupOp
            //
            >();
        target.addLegalDialect<
            ascendc::AscendCDialect, ascvf::AscVFDialect, arith::ArithDialect, emitasc::EmitAscDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertTensor, ConvertAccumulator, ConvertReshape, ConvertBroadcast, ConvertSoftmax, ConvertRmsNorm,
            ConvertLayerNorm, ConvertReduce, ConvertInlineVF, ConvertCVGroup<asctile::CubeGroupOp, ascendc::IfAICOp>,
            ConvertCVGroup<asctile::VectorGroupOp, ascendc::IfAIVOp>
            //
            >(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerAscTilePass() { return std::make_unique<LowerAscTilePass>(); }
