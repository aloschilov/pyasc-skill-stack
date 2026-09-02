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
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERASCTILEDATATRANSFER
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

constexpr int64_t cubeKBlockBytes = ascendc::ubBlockSize;
constexpr int64_t fractalNum = 2;

SmallVector<Value> getTensorShape(OpBuilder& builder, asctile::TensorOp tensorOp)
{
    ascir::ConstantOpBuilder consts(builder);
    auto type = tensorOp.getType();
    auto dynamicSizes = tensorOp.getSizes();
    size_t dynamicSizeIndex = 0;
    SmallVector<Value> tensorShape;
    for (auto dim : type.getShape()) {
        if (ShapedType::isDynamic(dim))
            tensorShape.push_back(dynamicSizes[dynamicSizeIndex++]);
        else
            tensorShape.push_back(consts.i32(dim));
    }
    return tensorShape;
}

Value linearizeOffset(OpBuilder& builder, Location loc, ArrayRef<Value> tensorShape, ValueRange offsets)
{
    ascir::ConstantOpBuilder consts(builder);
    assert(offsets.size() == tensorShape.size() && "must be one offset for each dimension");
    Value linearOffset = consts.i32(0);
    Value stride = consts.i32(1);
    for (size_t i = tensorShape.size(); i-- > 0;) {
        Value next = builder.create<arith::MulIOp>(loc, offsets[i], stride);
        linearOffset = builder.create<arith::AddIOp>(loc, linearOffset, next);
        stride = builder.create<arith::MulIOp>(loc, tensorShape[i], stride);
    }
    return linearOffset;
}

SmallVector<Value> getStaticShape(OpBuilder& builder, ShapedType type)
{
    ascir::ConstantOpBuilder consts(builder);
    SmallVector<Value> shape;
    for (auto dim : type.getShape())
        shape.push_back(consts.i32(dim));
    return shape;
}

struct TensorInfo {
    SmallVector<Value> shape;
    Value tensor;
    ascendc::BaseTensorType type;
};

template <typename T>
SmallVector<T> applyPermute(const SmallVectorImpl<T>& input, ArrayRef<int64_t> permute)
{
    SmallVector<T> result;
    for (auto i : permute) {
        result.push_back(input[i]);
    }
    return result;
}

template <typename T, typename C>
SmallVector<T> reversePermute(const C& input, ArrayRef<int64_t> permute)
{
    assert(input.size() == permute.size());
    SmallVector<T> result;
    result.assign(permute.size(), T{});
    for (size_t i = 0; i < permute.size(); ++i) {
        result[permute[i]] = static_cast<T>(input[i]);
    }
    return result;
}

SmallVector<Value> getStrides(
    ConvertRewriter& rewriter, Location loc, ascir::ConstantOpBuilder& consts, ValueRange shape)
{
    SmallVector<Value> result;
    result.assign(shape.size(), Value{});
    Value stride = consts.i32(1);
    for (size_t i = shape.size() - 1; i + 1 >= 1; --i) {
        result[i] = stride;
        if (i > 0) {
            stride = rewriter.create<arith::MulIOp>(loc, stride, shape[i]);
        }
    }
    return result;
}

SmallVector<int64_t> getStrides(ArrayRef<int64_t> shape)
{
    SmallVector<int64_t> result;
    result.assign(shape.size(), 0);
    int64_t stride = 1;
    for (size_t i = shape.size() - 1; i + 1 >= 1; --i) {
        result[i] = stride;
        stride *= shape[i];
    }
    return result;
}

TensorInfo prepareTensorInfo(ConvertRewriter& rewriter, Location loc, Value base, ValueRange offsets = {})
{
    auto tensorOp = base.getDefiningOp<asctile::TensorOp>();
    assert(tensorOp && "tensor must be created by asctile.tensor op");
    SmallVector<Value> shape = getTensorShape(rewriter, tensorOp);
    Value tensor = rewriter.getRemappedValue(base);
    auto type = cast<ascendc::BaseTensorType>(tensor.getType());
    if (!offsets.empty()) {
        Value linearOffset = linearizeOffset(rewriter, loc, shape, offsets);
        tensor = rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, type, tensor, linearOffset);
    }
    return {shape, tensor, type};
}

std::optional<ascendc::QuantMode> getQuantizeMode(
    ascendc::BaseTensorType srcType, ascendc::BaseTensorType dstType, ConvertRewriter& rewriter)
{
    auto srcElType = srcType.getElementType();
    auto dstElType = dstType.getElementType();
    auto floatType = rewriter.getF32Type();
    auto halfType = rewriter.getF16Type();
    auto int32Type = rewriter.getIntegerType(32);
    auto int8Type = rewriter.getIntegerType(8);
    auto uint8Type = rewriter.getIntegerType(8, false);
    if (srcElType == floatType && dstElType == floatType) {
        return ascendc::QuantMode::NoQuant;
    }
    if (srcElType == floatType && dstElType == halfType) {
        return ascendc::QuantMode::F322F16;
    }
    if (srcElType == floatType && dstElType == rewriter.getBF16Type()) {
        return ascendc::QuantMode::F322BF16;
    }
    if (srcElType == int32Type && dstElType == halfType) {
        return ascendc::QuantMode::DEQF16;
    }
    if (srcElType == floatType && (dstElType == int8Type || dstElType == uint8Type)) {
        return ascendc::QuantMode::QF322B8_PRE;
    }
    if (srcElType == int32Type && (dstElType == int8Type || dstElType == uint8Type)) {
        return ascendc::QuantMode::REQ8;
    }
    // TODO: Add support for VDEQF16, VQF322B8_PRE, VREQ8
    return std::nullopt;
}

Value calculateCopyCount(
    ConvertRewriter& rewriter, Location& loc, ArrayRef<int64_t> ubShape, ValueRange gmShape, ValueRange offsets,
    ValueRange realShape, size_t dim)
{
    ascir::ConstantOpBuilder consts(rewriter);
    Value tailElementsCount = rewriter.create<arith::SubIOp>(loc, gmShape[dim], offsets[dim]);
    tailElementsCount = rewriter.create<arith::MaxSIOp>(loc, tailElementsCount, consts.i32(0));
    return rewriter.create<arith::MinSIOp>(
        loc, realShape.empty() ? consts.i32(ubShape[dim]) : realShape[dim], tailElementsCount);
}

void setCopyDirection(ascendc::DataCopyOp op, Value src, Value dst)
{
    auto srcPos = ascendc::TPosition::GM;
    auto dstPos = ascendc::TPosition::GM;
    if (auto tensor = src.getDefiningOp<ascendc::LocalTensorAutoOp>())
        srcPos = tensor.getPosition();
    if (auto tensor = dst.getDefiningOp<ascendc::LocalTensorAutoOp>())
        dstPos = tensor.getPosition();
    op.setDirection(srcPos, dstPos);
}

void setCopyDirection(ascendc::DataCopyOp op) { setCopyDirection(op, op.getSrc(), op.getDst()); }

struct ConvertLoadToUB : ConvertOp<asctile::LoadOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::LoadOp op, ConvertRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto dstLoc = opType.getLoc();
        if (dstLoc != asctile::TensorLocation::UB || op->hasAttr(asctile::attr::transposeDims))
            return failure();
        auto loc = op.getLoc();
        auto offsets = op.getOffsets();
        TensorInfo srcInfo = prepareTensorInfo(rewriter, loc, op.getBase(), offsets);
        auto dst = createTensorOp(rewriter, loc, opType, locationToPosition(dstLoc)).getResult();
        auto dstType = dst.getType();
        auto dstShape = dstType.getShape();
        ascir::ConstantOpBuilder consts(rewriter);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        auto padValue = rewriter.getRemappedValue(op.getPadValue());
        auto typeSizeValue = consts.i32(ascendc::getElementTypeSize(dstType));
        Value dstLastDim = consts.i32(dstShape.back());
        Value srcLastDim = srcInfo.shape.back();
        Value tailElementsLastDim = rewriter.create<arith::SubIOp>(loc, srcLastDim, offsets.back());
        auto tailNegCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tailElementsLastDim, const0);
        Value tailElements = rewriter.create<arith::SelectOp>(loc, tailNegCond, const0, tailElementsLastDim);
        Value minTailElements = rewriter.create<arith::MinSIOp>(loc, dstLastDim, tailElements);
        Value blockLen, srcStrideElements;
        auto realShape = op.getRealShape();
        auto hasRealShape = !realShape.empty();
        if (hasRealShape) {
            Value realLastDim = rewriter.getRemappedValue(realShape.back());
            Value realTailElements = rewriter.create<arith::MinSIOp>(loc, realLastDim, tailElements);
            blockLen = rewriter.create<arith::MulIOp>(loc, realTailElements, typeSizeValue);
            srcStrideElements = rewriter.create<arith::SubIOp>(loc, srcLastDim, realTailElements);
        } else {
            blockLen = rewriter.create<arith::MulIOp>(loc, minTailElements, typeSizeValue);
            srcStrideElements = rewriter.create<arith::SubIOp>(loc, srcLastDim, minTailElements);
        }
        auto ubBlockSizeValue = consts.i32(ascendc::ubBlockSize);
        auto blockLenRemainder = rewriter.create<arith::RemSIOp>(loc, blockLen, ubBlockSizeValue);
        auto remainderIsZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, blockLenRemainder, const0);
        auto rightPadBytes = rewriter.create<arith::SelectOp>(
            loc, remainderIsZero, const0, rewriter.create<arith::SubIOp>(loc, ubBlockSizeValue, blockLenRemainder));
        auto alignedBlockSize = rewriter.create<arith::AddIOp>(loc, blockLen, rightPadBytes);
        auto rowSizeBytes = rewriter.create<arith::MulIOp>(loc, dstLastDim, typeSizeValue);
        if (hasRealShape) {
            auto hasGap =
                rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, rowSizeBytes, alignedBlockSize);
            auto ifHasGap = rewriter.create<scf::IfOp>(loc, hasGap, false);
            ConvertRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ifHasGap.thenBlock());
            rewriter.create<ascendc::DuplicateL2Op>(loc, dst, padValue, const0);
        }
        Value blockCount = const1;
        if (dstShape.size() > 1) {
            // innerRows = max(0, min(srcRows, dstRows + offsetRows) - offsetRows)
            // If realShape is provided, further limit innerRows to realShape[0]
            auto srcRows = srcInfo.shape[0];
            auto dstRows = consts.i32(dstShape[0]);
            auto offsetRows = offsets[0];
            auto endPos = rewriter.create<arith::AddIOp>(loc, dstRows, offsetRows);
            auto clampedEnd = rewriter.create<arith::MinSIOp>(loc, srcRows, endPos);
            auto rowCount = rewriter.create<arith::SubIOp>(loc, clampedEnd, offsetRows);
            Value innerRows = rewriter.create<arith::MaxSIOp>(loc, const0, rowCount);
            if (!realShape.empty()) {
                Value realRows = rewriter.getRemappedValue(realShape[0]);
                innerRows = rewriter.create<arith::MinSIOp>(loc, innerRows, realRows);
            }
            blockCount = innerRows;
            if (hasRealShape) {
                auto padRows = rewriter.create<arith::SubIOp>(loc, dstRows, innerRows);
                auto padRowsIsPositive =
                    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, padRows, const0);
                auto ifPadRows = rewriter.create<scf::IfOp>(loc, padRowsIsPositive, false);
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(ifPadRows.thenBlock());
                auto dstCols = consts.i32(dstShape.back());
                auto padOffset = rewriter.create<arith::MulIOp>(loc, innerRows, dstCols);
                auto padTensor = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dstType, dst, padOffset);
                auto padCalCount = rewriter.create<arith::MulIOp>(loc, padRows, dstCols);
                auto dupOp = rewriter.create<ascendc::DuplicateL2Op>(loc, padTensor, padValue, padCalCount);
                dupOp->setAttr(ascendc::attr::calCountSet, rewriter.getUnitAttr());
            }
        }
        auto srcStride = rewriter.create<arith::MulIOp>(loc, srcStrideElements, typeSizeValue);
        auto rightPadElements = rewriter.create<arith::DivSIOp>(loc, rightPadBytes, typeSizeValue);
        auto dstStrideBytes = rewriter.create<arith::SubIOp>(loc, rowSizeBytes, alignedBlockSize);
        auto dstStrideDataBlocks = rewriter.create<arith::DivSIOp>(loc, dstStrideBytes, ubBlockSizeValue);
        auto ui32Type = rewriter.getIntegerType(32, false);
        auto extParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
            ValueRange{blockCount, blockLen, srcStride, dstStrideDataBlocks, const0},
            rewriter.getTypeArrayAttr({rewriter.getIntegerType(16, false), ui32Type, ui32Type, ui32Type, ui32Type}));
        auto padExtParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyPadExtParamsType>(getElementTypeOrSelf(dstType)),
            ValueRange{const1, const0, rightPadElements, padValue},
            rewriter.getTypeArrayAttr(
                {rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getIntegerType(8, false), padValue.getType()}));
        auto copyOp = rewriter.create<ascendc::DataCopyPadExtL0Op>(loc, dst, srcInfo.tensor, extParams, padExtParams);
        setCopyDirection(copyOp);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertLoadToUBWithTranspose : ConvertOp<asctile::LoadOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    static void createTilePadding(
        ConvertRewriter& rewriter, Location loc, Value value, ValueRange copyCount, ArrayRef<int64_t> tensorShape,
        Value padValue)
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto const0 = consts.i32(0);
        Value needDuplicateFull = consts.i1(false);
        for (size_t i = copyCount.size() - 1; i + 1 > 1; --i) {
            Value padCount = rewriter.create<arith::SubIOp>(loc, consts.i32(tensorShape[i]), copyCount[i]);
            Value needPadding = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, padCount, const0);
            needDuplicateFull = rewriter.create<arith::OrIOp>(loc, needDuplicateFull, needPadding);
        }
        Value firstDimPadding = rewriter.create<arith::SubIOp>(loc, consts.i32(tensorShape.front()), copyCount.front());
        Value needDuplicateLast =
            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, firstDimPadding, const0);
        auto ifDuplicateFullOp = rewriter.create<scf::IfOp>(loc, needDuplicateFull, true);
        {
            ConvertRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ifDuplicateFullOp.thenBlock());
            rewriter.create<ascendc::DuplicateL2Op>(loc, value, padValue, const0);
        }
        {
            ConvertRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ifDuplicateFullOp.elseBlock());
            auto ifDuplicateLast = rewriter.create<scf::IfOp>(loc, needDuplicateLast, false);
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(ifDuplicateLast.thenBlock());
                auto strides = getStrides(tensorShape);
                Value firstDimStride = consts.i32(strides.front());
                Value padOffset = rewriter.create<arith::MulIOp>(loc, firstDimStride, copyCount.front());
                auto padTensor =
                    rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, value.getType(), value, padOffset);
                auto padCalCount = rewriter.create<arith::MulIOp>(loc, firstDimStride, firstDimPadding);
                auto dupOp = rewriter.create<ascendc::DuplicateL2Op>(loc, padTensor, padValue, padCalCount);
                dupOp->setAttr(ascendc::attr::calCountSet, rewriter.getUnitAttr());
            }
        }
    }

    LogicalResult matchAndRewrite(asctile::LoadOp op, ConvertRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto dstLoc = opType.getLoc();
        if (dstLoc != asctile::TensorLocation::UB)
            return failure();
        auto transposeAttrs = op->getAttrOfType<DenseI32ArrayAttr>(asctile::attr::transposeDims);
        if (!transposeAttrs)
            return failure();
        auto loc = op.getLoc();
        TensorInfo srcInfo = prepareTensorInfo(rewriter, loc, op.getBase(), op.getOffsets());
        auto dst = createTensorOp(rewriter, loc, opType, locationToPosition(dstLoc)).getResult();
        auto dstShape = dst.getType().getShape();
        ascir::ConstantOpBuilder consts(rewriter);
        auto const1 = consts.i32(1);
        auto padValue = rewriter.getRemappedValue(op.getPadValue());
        SmallVector<int64_t> dimsOrder{static_cast<ArrayRef<int32_t>>(transposeAttrs)};
        auto dimCount = dimsOrder.size();

        SmallVector<int64_t> readShape = reversePermute<int64_t>(dstShape, dimsOrder);
        SmallVector<Value> copyCount;
        for (size_t i = 0; i < dimCount; ++i) {
            copyCount.push_back(
                calculateCopyCount(rewriter, loc, readShape, srcInfo.shape, op.getOffsets(), op.getRealShape(), i));
        }
        SmallVector<Value> srcStrides = getStrides(rewriter, loc, consts, srcInfo.shape);
        SmallVector<int32_t> dstStrides = reversePermute<int32_t>(getStrides(dstShape), dimsOrder);
        SmallVector<int32_t> padLeft(dimCount, 0);
        SmallVector<int32_t> padRight(dimCount, 0);

        if (!op.getRealShape().empty()) {
            createTilePadding(rewriter, loc, dst, applyPermute(copyCount, dimsOrder), dstShape, padValue);
        }
        auto paramsType = rewriter.getType<ascendc::NdDmaParamsType>(srcInfo.type.getElementType(), dimCount);
        auto params = rewriter.create<ascendc::NdDmaParamsOp>(
            loc, paramsType, dimCount, padValue, copyCount, srcStrides, rewriter.getI32ArrayAttr(dstStrides),
            rewriter.getI32ArrayAttr(padLeft), rewriter.getI32ArrayAttr(padRight));
        auto copyOp = rewriter.create<ascendc::DataCopyNdDmaOp>(loc, dst, srcInfo.tensor, params.getResult(), dimCount);
        setCopyDirection(copyOp);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertLoadToL1 : ConvertOp<asctile::LoadOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::LoadOp op, ConvertRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto dstLoc = opType.getLoc();
        if (dstLoc != asctile::TensorLocation::L1)
            return failure();
        auto loc = op.getLoc();
        TensorInfo srcInfo = prepareTensorInfo(rewriter, loc, op.getBase(), op.getOffsets());
        auto dst = createTensorOp(rewriter, loc, opType, locationToPosition(dstLoc)).getResult();
        auto dstShape = dst.getType().getShape();
        ascir::ConstantOpBuilder consts(rewriter);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        if (op->hasAttrOfType<UnitAttr>(asctile::attr::isBias)) {
            if (dstShape.size() != 1)
                return op.emitError() << "L1 load must be 1D for bias";
            auto calCount = consts.i32(dstShape[0]);
            auto copyOp = rewriter.create<ascendc::DataCopyL2Op>(loc, dst, srcInfo.tensor, calCount);
            setCopyDirection(copyOp);
            rewriter.replaceOp(op, dst);
            return success();
        }
        auto ui16Type = rewriter.getIntegerType(16, false);
        auto ui32Type = rewriter.getIntegerType(32, false);
        auto ui64Type = rewriter.getIntegerType(64, false);
        auto argTypes = rewriter.getTypeArrayAttr(
            TypeRange{ui16Type, ui16Type, ui32Type, ui64Type, ui32Type, ui16Type, ui16Type, ui64Type});
        bool isMatrixA = op->hasAttr(asctile::attr::isMatrixA);
        bool isTransposeAL0 = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeAL0);
        bool isTransposeAL1 = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeAL1);
        bool isTransposeBL0 = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeBL0);
        bool isTransposeBL1 = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeBL1);
        bool isL1Transpose = isTransposeAL1 || isTransposeBL1;
        auto dstShapeCols = rewriter.create<arith::MinSIOp>(
            loc, srcInfo.shape[1], consts.i32(isL1Transpose ? dstShape[0] : dstShape[1]));
        auto dstShapeRows = rewriter.create<arith::MinSIOp>(
            loc, srcInfo.shape[0], consts.i32(isL1Transpose ? dstShape[1] : dstShape[0]));
        auto offsets = op.getOffsets();
        assert(offsets.size() == 2);
        Value availableRows = rewriter.create<arith::MaxSIOp>(
            loc, const0, rewriter.create<arith::SubIOp>(loc, srcInfo.shape[0], offsets[0]));
        Value availableCols = rewriter.create<arith::MaxSIOp>(
            loc, const0, rewriter.create<arith::SubIOp>(loc, srcInfo.shape[1], offsets[1]));
        Value nValue = rewriter.create<arith::MinSIOp>(loc, dstShapeRows, availableRows);
        Value dValue = rewriter.create<arith::MinSIOp>(loc, dstShapeCols, availableCols);
        if (isMatrixA && isTransposeAL1) {
            auto dstType = dst.getType();
            auto dstNzC0Stride = consts.i32(dstShape[0]);
            auto dn2NzParams = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::Dn2NzParamsType>(),
                ValueRange{const1, dValue, nValue, const0, srcInfo.shape[1], dstNzC0Stride, const1, const0}, argTypes);
            auto copyOp = rewriter.create<ascendc::DataCopyL2Op>(loc, dst, srcInfo.tensor, dn2NzParams);
            setCopyDirection(copyOp);
        } else {
            constexpr int64_t maxSrcDValue = 65535;
            auto dstRowStride = consts.i32(isTransposeBL1 ? dstShape[1] : dstShape[0]);
            auto needsLoop = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, srcInfo.shape[1], consts.i32(maxSrcDValue));
            auto ifOp = rewriter.create<scf::IfOp>(loc, needsLoop, /*withElseRegion=*/true);
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(ifOp.thenBlock());
                auto dstType = dst.getType();
                auto srcRows = srcInfo.shape[0];
                auto offsetRows = op.getOffsets().empty() ? const0 : op.getOffsets()[0];
                auto availableRows = rewriter.create<arith::SubIOp>(loc, srcRows, offsetRows);
                auto availableRowsPos = rewriter.create<arith::MaxSIOp>(loc, const0, availableRows);
                auto actualNValue = rewriter.create<arith::MinSIOp>(loc, nValue, availableRowsPos);
                auto forOp = rewriter.create<scf::ForOp>(loc, const0, actualNValue, const1);
                rewriter.setInsertionPointToStart(forOp.getBody());
                auto rowIdx = forOp.getInductionVar();
                auto srcRowOffset = rewriter.create<arith::MulIOp>(loc, rowIdx, srcInfo.shape[1]);
                auto srcTensorWithOffset =
                    rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, srcInfo.type, srcInfo.tensor, srcRowOffset);
                auto dstRowOffset = rewriter.create<arith::MulIOp>(loc, rowIdx, consts.i32(ascendc::cubeBlockSize));
                auto dstTensorWithOffset =
                    rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dstType, dst, dstRowOffset);
                auto nd2NzParams = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::Nd2NzParamsType>(),
                    ValueRange{const1, const1, dValue, const0, dValue, dstRowStride, const1, const0}, argTypes);
                auto copyOp =
                    rewriter.create<ascendc::DataCopyL2Op>(loc, dstTensorWithOffset, srcTensorWithOffset, nd2NzParams);
                setCopyDirection(copyOp, srcInfo.tensor, dst);
                rewriter.setInsertionPointAfter(forOp);
            }
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(ifOp.elseBlock());
                auto dstNzC0Stride = dstShape[0] == 1 && isMatrixA ?
                                         dstRowStride :
                                         consts.i32(
                                             static_cast<int64_t>(llvm::alignTo(
                                                 isTransposeBL1 ? dstShape[1] : dstShape[0], ascendc::cubeBlockSize)));
                auto nd2NzParams = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::Nd2NzParamsType>(),
                    ValueRange{const1, nValue, dValue, const0, srcInfo.shape[1], dstNzC0Stride, const1, const0},
                    argTypes);
                auto copyOp = rewriter.create<ascendc::DataCopyL2Op>(loc, dst, srcInfo.tensor, nd2NzParams);
                setCopyDirection(copyOp);
            }
        }
        auto elemType = dst.getType().getElementType();
        auto constArgTypes = rewriter.getTypeArrayAttr(TypeRange{ui16Type, ui16Type, ui16Type, elemType});
        auto elementSize = ascendc::getElementTypeSize(opType);
        int64_t cubeKBlockSize = cubeKBlockBytes / elementSize;
        auto c0Size = consts.i32(cubeKBlockSize);
        auto elemSizeVal = consts.i32(elementSize);
        auto blockSizeVal = consts.i32(cubeKBlockBytes);
        if ((isMatrixA && (isTransposeAL0 || isTransposeAL1)) || (!isMatrixA && (!isTransposeBL0 && !isTransposeBL1))) {
            auto totalBytes = consts.i32(dstShape[0] * dstShape[1] * elementSize);
            auto validBytes = rewriter.create<arith::MulIOp>(
                loc, nValue, consts.i32((isTransposeAL1 ? dstShape[0] : dstShape[1]) * elementSize));
            auto padBytes = rewriter.create<arith::SubIOp>(loc, totalBytes, validBytes);
            auto padBlocks = rewriter.create<arith::DivSIOp>(loc, padBytes, blockSizeVal);
            auto needsRowPad = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, padBlocks, const0);
            auto rowPadIf = rewriter.create<scf::IfOp>(loc, needsRowPad);
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(rowPadIf.thenBlock());
                auto colBlocks = consts.i32((isTransposeAL1 ? dstShape[0] : dstShape[1]) / cubeKBlockSize);
                auto rowOffset = rewriter.create<arith::MulIOp>(loc, nValue, c0Size);
                auto blockNum =
                    rewriter.create<arith::SubIOp>(loc, consts.i32(isTransposeAL1 ? dstShape[1] : dstShape[0]), nValue);
                auto padTensor = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dst.getType(), dst, rowOffset);
                auto params = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::InitConstValueParamsType>(),
                    ValueRange{colBlocks, blockNum, nValue, const0}, constArgTypes);
                rewriter.create<ascendc::FillOp>(loc, padTensor, params.getResult());
            }
        } else {
            auto dValueBytes = rewriter.create<arith::MulIOp>(loc, dValue, elemSizeVal);
            auto totalBytes = consts.i32((isTransposeBL1 ? dstShape[0] : dstShape[1]) * elementSize);
            auto padBytes = rewriter.create<arith::SubIOp>(loc, totalBytes, dValueBytes);
            auto padBlocks = rewriter.create<arith::DivSIOp>(loc, padBytes, blockSizeVal);
            auto needsColPad = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, padBlocks, const0);
            auto colPadIf = rewriter.create<scf::IfOp>(loc, needsColPad);
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(colPadIf.thenBlock());
                auto colOffset =
                    rewriter.create<arith::MulIOp>(loc, dValue, consts.i32(isTransposeBL1 ? dstShape[1] : dstShape[0]));
                auto blockNum = rewriter.create<arith::MulIOp>(
                    loc, padBlocks, consts.i32(isTransposeBL1 ? dstShape[1] : dstShape[0]));
                auto padTensor = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dst.getType(), dst, colOffset);
                auto params = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::InitConstValueParamsType>(),
                    ValueRange{const1, blockNum, const0, const0}, constArgTypes);
                rewriter.create<ascendc::FillOp>(loc, padTensor, params.getResult());
            }
        }
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertStore : ConvertOp<asctile::StoreOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    static bool needLoopMode(ArrayRef<int64_t> shape)
    {
        for (size_t dim = 0; dim + 2 < shape.size(); ++dim) {
            if (shape[dim] > 1)
                return true;
        }
        return false;
    }
    static bool checkShape(ArrayRef<int64_t> shape)
    {
        for (size_t dim = 0; dim + 4 < shape.size(); ++dim) {
            if (shape[dim] > 1)
                return false;
        }
        return true;
    }
    static SmallVector<Value> makeLoopModeValues(
        ConvertRewriter& rewriter, Location loc, ascir::ConstantOpBuilder& consts, ValueRange dstShape,
        ArrayRef<int64_t> srcShape, Value typeSize, ValueRange offsets, ValueRange realShape)
    {
        Value loop1Size =
            calculateCopyCount(rewriter, loc, srcShape, dstShape, offsets, realShape, srcShape.size() - 3);
        Value loop2Size =
            srcShape.size() == 3 ?
                consts.i32(1) :
                calculateCopyCount(rewriter, loc, srcShape, dstShape, offsets, realShape, srcShape.size() - 4);
        Value dim0SrcStride = rewriter.create<arith::MulIOp>(loc, consts.i32(srcShape[srcShape.size() - 1]), typeSize);
        Value dim1SrcStride =
            rewriter.create<arith::MulIOp>(loc, dim0SrcStride, consts.i32(srcShape[srcShape.size() - 2]));
        Value dim2SrcStride =
            rewriter.create<arith::MulIOp>(loc, dim1SrcStride, consts.i32(srcShape[srcShape.size() - 3]));

        Value dim0DstStride = rewriter.create<arith::MulIOp>(loc, dstShape[dstShape.size() - 1], typeSize);
        Value dim1DstStride = rewriter.create<arith::MulIOp>(loc, dim0DstStride, dstShape[dstShape.size() - 2]);
        Value dim2DstStride = rewriter.create<arith::MulIOp>(loc, dim1DstStride, dstShape[dstShape.size() - 3]);
        return SmallVector<Value>{loop1Size, loop2Size, dim1SrcStride, dim1DstStride, dim2SrcStride, dim2DstStride};
    }

    LogicalResult matchAndRewrite(asctile::StoreOp op, ConvertRewriter& rewriter) const override
    {
        auto value = op.getValue();
        assert(value.getType().getLoc() == asctile::TensorLocation::UB && "tensor must be located in UB");
        Value src = rewriter.getRemappedValue(value);
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        SmallVector<Value> srcShape = getStaticShape(rewriter, srcType);
        if (op->hasAttrOfType<DenseI32ArrayAttr>(asctile::attr::transposeDims))
            return failure();
        if (!checkShape(srcType.getShape()))
            return op.emitError("Store tensors with dim > 4 not implemented");
        auto loc = op.getLoc();
        auto offsets = op.getOffsets();
        ascir::ConstantOpBuilder consts(rewriter);
        TensorInfo dstInfo = prepareTensorInfo(rewriter, loc, op.getBase(), offsets);
        auto const0 = consts.i32(0);
        auto ui32Type = rewriter.getIntegerType(32, false);
        auto ui64Type = rewriter.getIntegerType(64, false);
        Value typeSize = consts.i32(ascendc::getElementTypeSize(srcType));
        Value srcLastDim = srcShape.back();
        Value dstLastDim = dstInfo.shape.back();
        Value tailElementsLastDim = rewriter.create<arith::SubIOp>(loc, dstLastDim, offsets.back());
        auto tailNegCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tailElementsLastDim, const0);
        Value tailElements = rewriter.create<arith::SelectOp>(loc, tailNegCond, const0, tailElementsLastDim);
        Value minTailElements;
        auto realShape = op.getRealShape();
        if (realShape.empty()) {
            minTailElements = rewriter.create<arith::MinSIOp>(loc, srcLastDim, tailElements);
        } else {
            Value realLastDim = realShape.back();
            minTailElements = rewriter.create<arith::MinSIOp>(loc, realLastDim, tailElements);
        }
        Value blockLen = rewriter.create<arith::MulIOp>(loc, minTailElements, typeSize);
        Value srcStrideElements = rewriter.create<arith::SubIOp>(loc, srcLastDim, minTailElements);
        Value dstStrideElements = rewriter.create<arith::SubIOp>(loc, dstLastDim, minTailElements);
        Value blockCount = consts.i32(1);
        if (srcShape.size() > 1) {
            blockCount = calculateCopyCount(
                rewriter, loc, srcType.getShape(), dstInfo.shape, offsets, realShape, srcShape.size() - 2);
        }
        Value dataBlockSize = consts.i32(ascendc::ubBlockSize);
        Value srcStrideBytes = rewriter.create<arith::MulIOp>(loc, srcStrideElements, typeSize);
        Value srcStride = rewriter.create<arith::DivSIOp>(loc, srcStrideBytes, dataBlockSize);
        Value dstStride = rewriter.create<arith::MulIOp>(loc, dstStrideElements, typeSize);

        bool multiDim = needLoopMode(srcType.getShape());
        if (multiDim) {
            auto params = makeLoopModeValues(
                rewriter, loc, consts, dstInfo.shape, srcType.getShape(), typeSize, SmallVector<Value>{offsets},
                SmallVector<Value>{realShape});
            auto paramsOp = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::LoopModeParamsType>(), params,
                rewriter.getTypeArrayAttr({ui32Type, ui32Type, ui64Type, ui64Type, ui64Type, ui64Type}));
            auto setParamsOp =
                rewriter.create<ascendc::SetLoopModeParaOp>(loc, paramsOp, ascendc::DataCopyMVType::UB_TO_OUT);
        }
        auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
            ValueRange{blockCount, blockLen, srcStride, dstStride, const0},
            rewriter.getTypeArrayAttr({rewriter.getIntegerType(16, false), ui32Type, ui32Type, ui32Type, ui32Type}));
        auto copyOp =
            rewriter.replaceOpWithNewOp<ascendc::DataCopyPadExtL2Op>(op, dstInfo.tensor, src, dataCopyExtParams);
        copyOp.setDirection(ascendc::TPosition::VECCALC, ascendc::TPosition::GM);
        if (multiDim) {
            rewriter.create<ascendc::ResetLoopModeParaOp>(loc, ascendc::DataCopyMVType::UB_TO_OUT);
        }
        return success();
    }
};

struct ConvertStoreWithTranspose : ConvertOp<asctile::StoreOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;
    // Do transpose to copy_out in GM. Works only for 3d/4d tensor if last dim unchanged by permute

    LogicalResult matchAndRewrite(asctile::StoreOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto offsets = op.getOffsets();
        auto value = op.getValue();
        TensorInfo dstInfo = prepareTensorInfo(rewriter, loc, op.getBase(), offsets);
        Value src = rewriter.getRemappedValue(value);
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        assert(value.getType().getLoc() == asctile::TensorLocation::UB && "tensor must be located in UB");
        ascir::ConstantOpBuilder consts(rewriter);
        SmallVector<int64_t> srcShape{srcType.getShape()};

        if (srcType.getShape().size() != 3 && srcType.getShape().size() != 4)
            return failure();
        if (!op->hasAttrOfType<DenseI32ArrayAttr>(asctile::attr::transposeDims))
            return failure();
        auto transposeAttrs = op->getAttrOfType<DenseI32ArrayAttr>(asctile::attr::transposeDims);
        assert(transposeAttrs.size() == srcShape.size());
        SmallVector<int64_t> dimsOrder{transposeAttrs.asArrayRef()};
        assert(transposeAttrs[transposeAttrs.size() - 1] == transposeAttrs.size() - 1);
        auto const0 = consts.i32(0);
        Value typeSize = consts.i32(ascendc::getElementTypeSize(srcType));

        SmallVector<Value> copyCount;
        SmallVector<Value> dstShapeRev = reversePermute<Value>(dstInfo.shape, dimsOrder);
        SmallVector<Value> offsetsRev = reversePermute<Value>(ValueRange{offsets}, dimsOrder);
        SmallVector<Value> realShapeRev =
            op.getRealShape().empty() ? SmallVector<Value>{} : reversePermute<Value>(op.getRealShape(), dimsOrder);
        SmallVector<Value> dstStrides =
            reversePermute<Value>(getStrides(rewriter, loc, consts, dstInfo.shape), dimsOrder);
        SmallVector<int64_t> srcStrides = getStrides(srcShape);
        for (size_t i = 0; i < srcShape.size(); ++i) {
            copyCount.push_back(calculateCopyCount(rewriter, loc, srcShape, dstShapeRev, offsetsRev, realShapeRev, i));
        }

        auto ui32Type = rewriter.getIntegerType(32, false);
        Value blockSize = consts.i32(ascendc::ubBlockSize);
        Value blockCount = copyCount[copyCount.size() - 2];
        Value blockLen = copyCount.back();

        blockLen = rewriter.create<arith::MulIOp>(loc, blockLen, typeSize);
        Value innerSrcStride = consts.i32(srcShape.back());
        innerSrcStride = rewriter.create<arith::SubIOp>(loc, innerSrcStride, copyCount.back());
        innerSrcStride = rewriter.create<arith::MulIOp>(loc, innerSrcStride, typeSize);
        innerSrcStride = rewriter.create<arith::DivSIOp>(loc, innerSrcStride, blockSize);
        Value innerDstStride = dstStrides[dstStrides.size() - 2];
        innerDstStride = rewriter.create<arith::SubIOp>(loc, innerDstStride, copyCount.back());
        innerDstStride = rewriter.create<arith::MulIOp>(loc, innerDstStride, typeSize);

        size_t secondDim = srcShape.size() - 3;
        Value dim0SrcStride = rewriter.create<arith::MulIOp>(loc, consts.i32(srcStrides[secondDim]), typeSize);
        Value dim0DstStride = rewriter.create<arith::MulIOp>(loc, dstStrides[secondDim], typeSize);
        Value dim0CopyCount = copyCount[secondDim];
        Value dim1SrcStride;
        Value dim1DstStride;
        Value dim1CopyCount;
        if (srcShape.size() == 4) {
            size_t firstDim = srcShape.size() - 4;
            dim1SrcStride = rewriter.create<arith::MulIOp>(loc, consts.i32(srcStrides[firstDim]), typeSize);
            dim1DstStride = rewriter.create<arith::MulIOp>(loc, dstStrides[firstDim], typeSize);
            dim1CopyCount = copyCount[firstDim];
        } else {
            dim1SrcStride = blockSize;
            dim1DstStride = blockSize;
            dim1CopyCount = consts.i32(1);
        }
        auto ui64Type = rewriter.getIntegerType(64, false);
        auto params = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::LoopModeParamsType>(),
            ValueRange{dim0CopyCount, dim1CopyCount, dim0SrcStride, dim0DstStride, dim1SrcStride, dim1DstStride},
            rewriter.getTypeArrayAttr({ui32Type, ui32Type, ui64Type, ui64Type, ui64Type, ui64Type}));
        auto setParamsOp = rewriter.create<ascendc::SetLoopModeParaOp>(loc, params, ascendc::DataCopyMVType::UB_TO_OUT);
        auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
            ValueRange{blockCount, blockLen, innerSrcStride, innerDstStride, const0},
            rewriter.getTypeArrayAttr({rewriter.getIntegerType(16, false), ui32Type, ui32Type, ui32Type, ui32Type}));
        auto copyOp =
            rewriter.replaceOpWithNewOp<ascendc::DataCopyPadExtL2Op>(op, dstInfo.tensor, src, dataCopyExtParams);
        copyOp.setDirection(ascendc::TPosition::VECCALC, ascendc::TPosition::GM);
        rewriter.create<ascendc::ResetLoopModeParaOp>(loc, ascendc::DataCopyMVType::UB_TO_OUT);
        return success();
    }
};

struct ConvertStoreFixpipe : ConvertOp<asctile::StoreFixpipeOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::StoreFixpipeOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto offsets = op.getOffsets();
        auto value = op.getValue();
        TensorInfo dstInfo = prepareTensorInfo(rewriter, loc, op.getBase(), offsets);
        Value src = rewriter.getRemappedValue(value);
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        assert(value.getType().getLoc() == asctile::TensorLocation::L0C && "tensor must be located in L0C");
        ascir::ConstantOpBuilder consts(rewriter);
        SmallVector<Value> srcShape = getStaticShape(rewriter, srcType);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        Value srcLastDim = srcShape.back();
        Value dstLastDim = dstInfo.shape.back();
        Value tailElementsLastDim = rewriter.create<arith::SubIOp>(loc, dstLastDim, offsets.back());
        auto tailNegCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tailElementsLastDim, const0);
        Value tailElements = rewriter.create<arith::SelectOp>(loc, tailNegCond, const0, tailElementsLastDim);
        Value nSize = rewriter.create<arith::MinSIOp>(loc, srcLastDim, tailElements);
        Value mSize = srcShape[0];
        if (auto realShape = op.getRealShape(); !realShape.empty()) {
            Value realCols = rewriter.getRemappedValue(realShape.back());
            nSize = rewriter.create<arith::MinSIOp>(loc, nSize, realCols);
            Value realRows = rewriter.getRemappedValue(realShape[0]);
            mSize = rewriter.create<arith::MinSIOp>(loc, mSize, realRows);
        } else {
            Value tailRows = rewriter.create<arith::SubIOp>(loc, dstInfo.shape[0], offsets[0]);
            Value availableRows = rewriter.create<arith::MaxSIOp>(loc, const0, tailRows);
            mSize = rewriter.create<arith::MinSIOp>(loc, mSize, availableRows);
        }
        auto srcStride = static_cast<int32_t>(llvm::alignTo<ascendc::cubeBlockSize>(srcType.getShape()[0]));
        auto paramsBuilder = emitasc::InitStructBuilder(rewriter.getType<ascendc::FixpipeParamsV220Type>())
                                 .addField("nSize", nSize)
                                 .addField("mSize", mSize)
                                 .addField("srcStride", consts.i32(srcStride))
                                 .addField("dstStride", dstInfo.shape[1]);
        if (op.getRelu())
            paramsBuilder.addField("reluEn", const1);
        if (op.getQuantize()) {
            auto mode = getQuantizeMode(srcType, dstInfo.type, rewriter);
            if (!mode) {
                return op.emitError() << "Unsupported quant mode from " << srcType.getElementType() << " to "
                                      << dstInfo.type.getElementType();
            }
            auto quantMode = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::QuantModesType>(), ValueRange{consts.i32(static_cast<int32_t>(*mode))},
                rewriter.getTypeArrayAttr(rewriter.getType<ascendc::QuantModesType>()), true, true);
            paramsBuilder.addField("quantPre", quantMode);
        }
        Value params = paramsBuilder.create(rewriter, loc);
        Value layout = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::CO2LayoutType>(),
            ValueRange{consts.i32(static_cast<int32_t>(ascendc::CO2Layout::ROW_MAJOR))}, ArrayAttr{}, true, true);
        auto fixPipeConfig = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::FixpipeConfigType>(), ValueRange{layout}, ArrayAttr{}, true, true);
        auto copyOp = rewriter.replaceOpWithNewOp<ascendc::FixpipeOp>(op, dstInfo.tensor, src, params, fixPipeConfig);
        copyOp.setDirection(ascendc::TPosition::CO2, ascendc::TPosition::GM);
        return success();
    }
};

struct ConvertCopyFixpipe : ConvertOp<asctile::CopyFixpipeOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::CopyFixpipeOp op, ConvertRewriter& rewriter) const override
    {
        auto base = op.getResult();
        auto loc = op.getLoc();
        auto value = op.getBase();
        Value src = rewriter.getRemappedValue(value);
        Value dst = createTensorOp(rewriter, loc, op.getType());
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        assert(value.getType().getLoc() == asctile::TensorLocation::L0C && "tensor must be located in L0C.");
        auto dstType = cast<ascendc::BaseTensorType>(dst.getType());
        auto dstLoc = op.getType().getLoc();
        bool isToUB = dstLoc == asctile::TensorLocation::UB;
        assert((isToUB || dstLoc == asctile::TensorLocation::L1) && "dst should be in L1 or UB");
        assert((isToUB || dstType.getElementType() != rewriter.getF32Type()) && "dst type in L1 shouldn't be float32");
        ascir::ConstantOpBuilder consts(rewriter);
        SmallVector<Value> srcShape = getStaticShape(rewriter, srcType);
        SmallVector<Value> dstShape = getStaticShape(rewriter, dstType);
        auto const1 = consts.i32(1);
        Value linearOffset = linearizeOffset(rewriter, loc, dstShape, op.getOffsets());
        src = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, linearOffset);
        auto co2Layout = isToUB ? ascendc::CO2Layout::ROW_MAJOR : ascendc::CO2Layout::NZ;
        Value srcStride, dstStride;
        if (isToUB) {
            srcStride = consts.i32(static_cast<int32_t>(llvm::alignTo<ascendc::cubeBlockSize>(srcType.getShape()[0])));
            dstStride = dstShape[1];
        } else {
            srcStride = srcShape[0];
            dstStride = rewriter.create<arith::MulIOp>(
                loc, srcShape[0], consts.i32(cubeKBlockBytes / ascendc::getElementTypeSize(op.getType())));
        }
        auto paramsBuilder = emitasc::InitStructBuilder(
                                 ascendc::FixpipeParamsC310Type::get(
                                     op.getContext(), ascendc::CO2LayoutAttr::get(op.getContext(), co2Layout)))
                                 .addField("nSize", srcShape[1])
                                 .addField("mSize", srcShape[0])
                                 .addField("srcStride", srcStride)
                                 .addField("dstStride", dstStride);
        if (op.getRelu())
            paramsBuilder.addField("reluEn", const1);
        if (op.getQuantize()) {
            auto mode = getQuantizeMode(srcType, dstType, rewriter);
            if (!mode) {
                return op.emitError() << "Unsupported quant mode from " << srcType.getElementType() << " to "
                                      << dstType.getElementType();
            }
            auto quantMode = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::QuantModesType>(), ValueRange{consts.i32(static_cast<int32_t>(*mode))},
                rewriter.getTypeArrayAttr(rewriter.getType<ascendc::QuantModesType>()), true, true);
            paramsBuilder.addField("quantPre", quantMode);
        }
        Value params = paramsBuilder.create(rewriter, loc);
        Value layout = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::CO2LayoutType>(), ValueRange{consts.i32(static_cast<int32_t>(co2Layout))},
            ArrayAttr{}, true, true);
        Value fixPipeConfig = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::FixpipeConfigType>(), ValueRange{layout, consts.i1(isToUB)}, ArrayAttr{},
            true, true);
        auto copyOp = rewriter.create<ascendc::FixpipeOp>(loc, dst, src, params, fixPipeConfig);
        copyOp.setDirection(ascendc::TPosition::CO2, locationToPosition(dstLoc));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

Value buildLoadData2DV2Params(
    OpBuilder& builder, Location loc, ascir::ConstantOpBuilder& consts, bool isTensorA, bool isTransposeAL0,
    bool isTransposeBL0, bool isTransposeBL1, int64_t cubeKBlockSize, ArrayRef<int64_t> srcShape,
    ArrayRef<int64_t> dstShape)
{
    int64_t mStep, kStep, srcStride, dstStride;
    bool ifTranspose;
    bool isTransposeB = isTransposeBL0 || isTransposeBL1;
    if ((isTensorA && !isTransposeAL0) || (!isTensorA && isTransposeB)) {
        auto mAlignL0 = llvm::alignTo(isTransposeB ? dstShape[1] : dstShape[0], ascendc::cubeBlockSize);
        auto mAlignL1 = llvm::alignTo(isTransposeBL1 ? srcShape[1] : srcShape[0], ascendc::cubeBlockSize);
        auto kAlignL1 = llvm::alignTo(isTransposeB ? dstShape[0] : dstShape[1], cubeKBlockSize);
        mStep = llvm::divideCeilSigned(mAlignL0, ascendc::cubeBlockSize);
        kStep = llvm::divideCeilSigned(kAlignL1, cubeKBlockSize);
        srcStride = llvm::divideCeilSigned(mAlignL1, ascendc::cubeBlockSize);
        dstStride = llvm::divideCeilSigned(mAlignL0, ascendc::cubeBlockSize);
        ifTranspose = false;
    } else {
        auto mAlignL1 = llvm::alignTo(isTransposeAL0 ? dstShape[0] : dstShape[1], ascendc::cubeBlockSize);
        auto kaAlignL0 = llvm::alignTo(isTransposeAL0 ? dstShape[1] : dstShape[0], ascendc::cubeBlockSize);
        auto kaAlignL1 = llvm::alignTo(srcShape[0], ascendc::cubeBlockSize);
        mStep = llvm::divideCeilSigned(kaAlignL0, ascendc::cubeBlockSize);
        kStep = llvm::divideCeilSigned(mAlignL1, cubeKBlockSize);
        srcStride = llvm::divideCeilSigned(kaAlignL1, ascendc::cubeBlockSize);
        dstStride = llvm::divideCeilSigned(mAlignL1, ascendc::cubeBlockSize);
        ifTranspose = true;
    }
    auto paramsType = builder.getType<ascendc::LoadData2DParamsV2Type>();
    return emitasc::InitStructBuilder(paramsType)
        .addField("mStep", consts.i32(mStep))
        .addField("kStep", consts.i32(kStep))
        .addField("srcStride", consts.i32(srcStride))
        .addField("dstStride", consts.i32(dstStride))
        .addField("ifTranspose", consts.i1(ifTranspose))
        .create(builder, loc);
}

struct ConvertCopy : ConvertOp<asctile::CopyOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::CopyOp op, ConvertRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto dstPos = opType.getLoc();
        if (dstPos != asctile::TensorLocation::L0A && dstPos != asctile::TensorLocation::L0B &&
            dstPos != asctile::TensorLocation::BT && dstPos != asctile::TensorLocation::L1)
            return op.emitOpError("has invalid location of the result tensor");
        auto loc = op.getLoc();
        auto base = op.getBase();
        Value src = rewriter.getRemappedValue(base);
        auto srcType = src.getType();
        auto srcShape = base.getType().getShape();
        auto offsets = op.getOffsets();
        ascir::ConstantOpBuilder consts(rewriter);
        if (dstPos == asctile::TensorLocation::L1) {
            if (srcShape.size() != 2)
                return op.emitOpError("only supports 2D tensor");
            auto srcLoc = base.getType().getLoc();
            if (srcLoc != asctile::TensorLocation::UB)
                return op.emitError("L1 destination requires UB source");
            auto dst = createTensorOp(rewriter, loc, opType, locationToPosition(dstPos)).getResult();
            auto srcTensorType = cast<ascendc::BaseTensorType>(srcType);
            auto elemType = srcTensorType.getElementType();
            int64_t elementSize = ascendc::getElementTypeSize(srcTensorType);
            auto dstShape = opType.getShape();
            int64_t height = dstShape[0];
            int64_t width = dstShape[1];
            int64_t srcWidth = srcShape[1];
            if (!offsets.empty()) {
                Value linearOffset = linearizeOffset(rewriter, loc, getStaticShape(rewriter, srcTensorType), offsets);
                src = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcTensorType, src, linearOffset);
            }
            int64_t cubeKBlockSize = static_cast<int64_t>(cubeKBlockBytes) / elementSize;
            int64_t colBlocks = width / cubeKBlockSize;
            int64_t srcColBlocks = srcWidth / cubeKBlockSize;
            int64_t totalElements = height * width;
            auto const0 = consts.i32(0);
            auto const1 = consts.i32(1);
            auto tempUB = createTensorOp(rewriter, loc, {totalElements}, elemType).getResult();
            auto dataCopyParams = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::DataCopyParamsType>(),
                ValueRange{consts.i32(height), const1, consts.i32(srcColBlocks - 1), const0});
            auto tempType = cast<ascendc::BaseTensorType>(tempUB.getType());
            auto forOp = rewriter.create<scf::ForOp>(loc, const0, consts.i32(colBlocks), const1);
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(forOp.getBody());
                auto colIdx = forOp.getInductionVar();
                auto srcOffset = rewriter.create<arith::MulIOp>(loc, colIdx, consts.i32(cubeKBlockSize));
                auto srcView = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcTensorType, src, srcOffset);
                auto dstOffset = rewriter.create<arith::MulIOp>(loc, colIdx, consts.i32(height * cubeKBlockSize));
                auto dstView = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, tempType, tempUB, dstOffset);
                auto innerCopyOp = rewriter.create<ascendc::DataCopyL2Op>(loc, dstView, srcView, dataCopyParams);
                innerCopyOp.setDirection(ascendc::TPosition::VECCALC, ascendc::TPosition::VECCALC);
            }
            rewriter.setInsertionPointAfter(forOp);
            auto outerCopyOp = rewriter.create<ascendc::DataCopyL2Op>(loc, dst, tempUB, consts.i32(totalElements));
            setCopyDirection(outerCopyOp);
            rewriter.replaceOp(op, dst);
            return success();
        }
        if (dstPos == asctile::TensorLocation::BT) {
            auto srcLoc = base.getType().getLoc();
            if (srcLoc != asctile::TensorLocation::L1) {
                op.emitError() << "BT destination requires L1 source for bias copy";
                return failure();
            }
            auto dst = createTensorOp(rewriter, loc, opType).getResult();
            auto dstShape = opType.getShape();
            if (srcShape.size() != 1)
                return op.emitError() << "bias must have 1D shape";
            if (!offsets.empty()) {
                src = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, offsets[0]);
            }
            int64_t typeSize = ascendc::getElementTypeSize(base.getType());
            int64_t blockLen = (dstShape[0] * typeSize) / cubeKBlockBytes;
            auto dataCopyParams = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::DataCopyParamsType>(),
                ValueRange{consts.i32(1), consts.i32(blockLen), consts.i32(0), consts.i32(0)});
            auto copyOp = rewriter.create<ascendc::DataCopyL0Op>(loc, dst, src, dataCopyParams);
            setCopyDirection(copyOp);
            rewriter.replaceOp(op, dst);
            return success();
        }
        assert(srcShape.size() == 2 && "supported only tensorShape with 2 dims");
        assert(offsets.size() == srcShape.size() && "must be one offset for each dimension");
        auto dst = createTensorOp(rewriter, loc, opType).getResult();
        auto dstType = dst.getType();
        auto dstShape = dstType.getShape();
        bool isTensorA = dstPos == asctile::TensorLocation::L0A;
        bool isTransposeAL0 = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeAL0);
        bool isTransposeBL0 = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeBL0);
        bool isTransposeBL1 = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeBL1);
        bool isFloat32 = isa<Float32Type>(opType.getElementType());
        const int64_t cubeKBlockSize = cubeKBlockBytes / ascendc::getElementTypeSize(opType);
        int64_t dstNzC0StrideElements =
            static_cast<int64_t>(llvm::alignTo(isTransposeBL1 ? srcShape[1] : srcShape[0], cubeKBlockSize));
        int64_t dValue = cubeKBlockSize;
        Value colOffset = rewriter.create<arith::MulIOp>(
            loc, consts.i32(dstNzC0StrideElements), isTransposeBL1 ? offsets[0] : offsets[1]);
        Value rowOffset =
            rewriter.create<arith::MulIOp>(loc, isTransposeBL1 ? offsets[1] : offsets[0], consts.i32(dValue));
        Value linearOffset = rewriter.create<arith::AddIOp>(loc, colOffset, rowOffset);
        src = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, linearOffset);
        Value params = buildLoadData2DV2Params(
            rewriter, loc, consts, isTensorA, isTransposeAL0, isTransposeBL0, isTransposeBL1, cubeKBlockSize, srcShape,
            dstShape);
        auto copyOp = rewriter.create<ascendc::LoadDataL0V2Op>(loc, dst, src, params);
        setCopyDirection(copyOp);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertGetValue : ConvertOp<asctile::GetValueOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::GetValueOp op, ConvertRewriter& rewriter) const override
    {
        auto base = op.getBase();
        auto loc = op.getLoc();
        auto tensorOp = base.getDefiningOp<asctile::TensorOp>();
        assert(tensorOp && "tensor must be created by asctile.tensor op");
        SmallVector<Value> srcShape = getTensorShape(rewriter, tensorOp);
        Value linearOffset = linearizeOffset(rewriter, loc, srcShape, op.getOffsets());
        Value src = rewriter.getRemappedValue(base);
        rewriter.replaceOpWithNewOp<ascendc::GlobalTensorGetValueOp>(op, op.getType(), src, linearOffset);
        return success();
    }
};

struct ConvertSetValue : ConvertOp<asctile::SetValueOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::SetValueOp op, ConvertRewriter& rewriter) const override
    {
        auto base = op.getBase();
        auto loc = op.getLoc();
        Value src = rewriter.getRemappedValue(op.getValue());
        if (auto srcType = dyn_cast<ascendc::LocalTensorType>(src.getType())) {
            ascir::ConstantOpBuilder consts(rewriter);
            src = rewriter.create<ascendc::LocalTensorGetValueOp>(loc, srcType.getElementType(), src, consts.i64(0));
        }
        auto tensorOp = base.getDefiningOp<asctile::TensorOp>();
        assert(tensorOp && "tensor must be created by asctile.tensor op");
        SmallVector<Value> dstShape = getTensorShape(rewriter, tensorOp);
        Value linearOffset = linearizeOffset(rewriter, loc, dstShape, op.getOffsets());
        Value dst = rewriter.getRemappedValue(base);
        Value offset = rewriter.create<emitc::CastOp>(loc, rewriter.getIntegerType(64, false), linearOffset);
        rewriter.replaceOpWithNewOp<ascendc::GlobalTensorSetValueOp>(op, dst, offset, src);
        return success();
    }
};

struct ConvertGather : ConvertOp<asctile::GatherOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::GatherOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto src = op.getSrc();
        auto srcShape = src.getType().getShape();
        auto indexShape = op.getIndex().getType().getShape();
        auto resultShape = op.getType().getShape();
        auto i32Type = rewriter.getIntegerType(32);
        auto ui32Type = rewriter.getIntegerType(32, false);
        ascir::ConstantOpBuilder consts(rewriter);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        size_t dim = static_cast<size_t>(op.getDim());
        size_t elementsPerBlock = ascendc::ubBlockSize / ascendc::getElementTypeSize(src.getType());
        auto elementSize = ascendc::getElementTypeSize(op.getType());
        auto elementType = src.getType().getElementType();
        auto indexType = op.getIndex().getType().getElementType();

        if (dim >= srcShape.size())
            return op.emitOpError("has 'dim' is incompatible with 'src' tensor rank");
        if (indexShape.size() != 1)
            return op.emitError("'index' have rank != 1");
        if (resultShape.back() * elementSize % ascendc::ubBlockSize != 0)
            return op.emitError("'result' inner dimension not aligned to block size");

        auto elementsCount = op.getType().getNumElements();
        auto result = createTensorOp(rewriter, loc, elementsCount, elementType).getResult();
        auto indexTensor = rewriter.getRemappedValue(op.getIndex());
        auto srcTensor = rewriter.getRemappedValue(src);

        SmallVector<Value> offsets{op.getOffsets()};
        offsets.insert(offsets.end(), srcShape.size() - offsets.size(), const0);
        auto srcInfo = prepareTensorInfo(rewriter, loc, src, offsets);

        auto ubStrides = getStrides(resultShape);
        auto gmStrides = getStrides(srcShape.drop_front(dim));

        Value dstStride = consts.i32(ubStrides.front()); // Stride in elements between writes to ub
        Value srcStride = consts.i32(gmStrides.front()); // Stride in elements between reads from gm

        int64_t blockLen = srcShape.back() * elementSize;         // Size of last dimension in bytes
        int64_t blockCount = gmStrides.front() / srcShape.back(); // Size of dimensions from dim+1 except last
        size_t padElements = (elementsPerBlock - srcShape.back() % elementsPerBlock) % elementsPerBlock;
        Value padValue;
        if (op.getPadValue()) {
            padValue = rewriter.getRemappedValue(op.getPadValue());
        } else {
            if (auto intType = dyn_cast<IntegerType>(elementType)) {
                padValue = consts.create(intType, 0);
            } else if (auto floatType = dyn_cast<FloatType>(elementType)) {
                padValue = consts.create(floatType, 0.0);
            } else {
                return op.emitOpError("doesn't support element type") << elementType;
            }
        }
        assert((blockLen + padElements * elementSize) % ascendc::ubBlockSize == 0);
        Value repeatsCount = op.getNumIndices() ? rewriter.create<arith::MinSIOp>(
                                                      loc, op.getNumIndices(), consts.i32(indexShape.front())) :
                                                  consts.i32(indexShape.front());
        Value maxIndex = rewriter.create<arith::SubIOp>(loc, srcInfo.shape[dim], op.getOffsets().back());
        auto forOp = rewriter.create<scf::ForOp>(loc, const0, repeatsCount, const1);
        {
            ConvertRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(forOp.getBody());
            auto stepIndex = forOp.getInductionVar();
            Value indexValue = rewriter.create<ascendc::LocalTensorGetValueOp>(loc, indexType, indexTensor, stepIndex);
            if (auto typeSize = ascendc::getTypeSize(indexValue.getType()); typeSize < 4)
                indexValue = rewriter.create<arith::ExtSIOp>(loc, i32Type, indexValue);
            else if (typeSize > 4)
                indexValue = rewriter.create<arith::TruncIOp>(loc, i32Type, indexValue);
            Value valueInBounds = consts.i1(true);
            if (op.getCheckBounds()) {
                auto check1 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, indexValue, const0);
                auto check2 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, indexValue, maxIndex);
                valueInBounds = rewriter.create<arith::AndIOp>(loc, check1, check2);
            }

            auto writeOffset = rewriter.create<arith::MulIOp>(loc, stepIndex, dstStride);
            auto writeTensor =
                rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, result.getType(), result, writeOffset);
            auto ifIndexValid = rewriter.create<scf::IfOp>(loc, valueInBounds, true);
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(ifIndexValid.thenBlock());
                auto readOffset = rewriter.create<arith::MulIOp>(loc, indexValue, srcStride);
                auto readTensor =
                    rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, srcInfo.type, srcInfo.tensor, readOffset);

                auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
                    ValueRange{consts.i32(blockCount), consts.i32(blockLen), const0, const0, const0},
                    rewriter.getTypeArrayAttr(
                        {rewriter.getIntegerType(16, false), ui32Type, ui32Type, ui32Type, ui32Type}));
                auto dataCopyPadExtParams = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::DataCopyPadExtParamsType>(padValue.getType()),
                    ValueRange{const1, const0, consts.i32(static_cast<int64_t>(padElements)), padValue},
                    rewriter.getTypeArrayAttr(
                        {i32Type, i32Type, rewriter.getIntegerType(8, false), padValue.getType()}));
                auto copyOp = rewriter.create<ascendc::DataCopyPadExtL0Op>(
                    loc, writeTensor, readTensor, dataCopyExtParams, dataCopyPadExtParams);
                copyOp.setDirection(ascendc::TPosition::GM, ascendc::TPosition::VECCALC);
            }
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(ifIndexValid.elseBlock());
                auto fillOp = rewriter.create<ascendc::DuplicateL2Op>(loc, writeTensor, padValue, dstStride);
                fillOp->setAttr(ascendc::attr::calCountSet, rewriter.getUnitAttr());
            }
        }
        rewriter.setInsertionPointAfter(forOp);
        rewriter.replaceOp(op, result);
        return success();
    }
};

struct ConvertScatter : ConvertOp<asctile::ScatterOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::ScatterOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto dstType = op.getDst().getType();
        auto dstShape = dstType.getShape();
        auto indexShape = op.getIndex().getType().getShape();
        auto srcShape = op.getSrc().getType().getShape();
        auto i32Type = rewriter.getIntegerType(32);
        auto ui32Type = rewriter.getIntegerType(32, false);
        ascir::ConstantOpBuilder consts(rewriter);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        size_t dim = static_cast<size_t>(op.getDim());
        size_t elementsPerBlock = ascendc::ubBlockSize / ascendc::getElementTypeSize(dstType);
        auto elementSize = ascendc::getElementTypeSize(dstType);
        auto elementType = dstType.getElementType();
        auto indexType = op.getIndex().getType().getElementType();

        if (dim >= dstShape.size())
            return op.emitError("'dim' out of 'operand' rank");
        if (indexShape.size() != 1)
            return op.emitError("'index' have rank != 1");
        if (srcShape.size() != dstShape.size() - dim)
            return op.emitError("'src' have rank of ") << srcShape.size() << " should be " << dstShape.size() - dim;
        if (op.getSrc().getType().getElementType() != dstType.getElementType())
            return op.emitError("'base' and 'src' mismatch types: ")
                   << op.getSrc().getType().getElementType() << " and " << dstType.getElementType();
        if (indexShape.front() > srcShape.front())
            return op.emitError("'src' and 'index' shape mismatch");

        auto indexTensor = rewriter.getRemappedValue(op.getIndex());
        auto srcTensor = rewriter.getRemappedValue(op.getSrc());

        SmallVector<Value> offsets{op.getOffsets()};
        offsets.insert(offsets.end(), dstShape.size() - offsets.size(), const0);
        auto dstInfo = prepareTensorInfo(rewriter, loc, op.getDst(), offsets);

        auto ubStrides = getStrides(srcShape);
        auto gmStrides = getStrides(dstShape.drop_front(dim));

        Value srcStride = consts.i32(ubStrides.front()); // Stride in elements between writes to ub
        Value dstStride = consts.i32(gmStrides.front()); // Stride in elements between reads from gm

        int64_t blockLen = std::min(dstShape.back(), srcShape.back()) * elementSize; // Size of last dimension in bytes
        int64_t blockCount = gmStrides.front() / dstShape.back(); // Size of dimensions from dim+1 except last

        Value repeatsCount = op.getNumIndices() ? rewriter.create<arith::MinSIOp>(
                                                      loc, op.getNumIndices(), consts.i32(indexShape.front())) :
                                                  consts.i32(indexShape.front());
        Value maxIndex = rewriter.create<arith::SubIOp>(loc, dstInfo.shape[dim], op.getOffsets().back());
        auto forOp = rewriter.create<scf::ForOp>(loc, const0, repeatsCount, const1);
        {
            ConvertRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(forOp.getBody());
            auto stepIndex = forOp.getInductionVar();
            Value indexValue = rewriter.create<ascendc::LocalTensorGetValueOp>(loc, indexType, indexTensor, stepIndex);
            if (auto typeSize = ascendc::getTypeSize(indexValue.getType()); typeSize < 4)
                indexValue = rewriter.create<arith::ExtSIOp>(loc, i32Type, indexValue);
            else if (typeSize > 4)
                indexValue = rewriter.create<arith::TruncIOp>(loc, i32Type, indexValue);
            Value valueInBounds = consts.i1(true);
            if (op.getCheckBounds()) {
                auto check1 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, indexValue, const0);
                auto check2 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, indexValue, maxIndex);
                valueInBounds = rewriter.create<arith::AndIOp>(loc, check1, check2);
            }

            auto srcOffset = rewriter.create<arith::MulIOp>(loc, stepIndex, srcStride);
            auto readTensor =
                rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcTensor.getType(), srcTensor, srcOffset);
            auto ifIndexValid = rewriter.create<scf::IfOp>(loc, valueInBounds, false);
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(ifIndexValid.thenBlock());
                auto writeOffset = rewriter.create<arith::MulIOp>(loc, indexValue, dstStride);
                auto writeTensor =
                    rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, dstInfo.type, dstInfo.tensor, writeOffset);
                auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
                    ValueRange{consts.i32(blockCount), consts.i32(blockLen), const0, const0, const0},
                    rewriter.getTypeArrayAttr(
                        {rewriter.getIntegerType(16, false), ui32Type, ui32Type, ui32Type, ui32Type}));
                auto copyOp =
                    rewriter.create<ascendc::DataCopyPadExtL2Op>(loc, writeTensor, readTensor, dataCopyExtParams);
                copyOp.setDirection(ascendc::TPosition::VECCALC, ascendc::TPosition::GM);
            }
        }
        rewriter.eraseOp(op);
        rewriter.setInsertionPointAfter(forOp);
        return success();
    }
};

struct LowerAscTileDataTransferPass
    : public asclower::impl::LowerAscTileDataTransferBase<LowerAscTileDataTransferPass> {
    void runOnOperation() override
    {
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addIllegalOp<
            //
            asctile::LoadOp, asctile::StoreOp, asctile::CopyOp, asctile::StoreFixpipeOp, asctile::GetValueOp,
            asctile::SetValueOp, asctile::CopyFixpipeOp, asctile::GatherOp, asctile::ScatterOp
            //
            >();
        target.addLegalDialect<
            ascendc::AscendCDialect, arith::ArithDialect, emitasc::EmitAscDialect, emitc::EmitCDialect,
            scf::SCFDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertLoadToUB, ConvertLoadToUBWithTranspose, ConvertLoadToL1, ConvertStore, ConvertStoreFixpipe,
            ConvertCopy, ConvertGetValue, ConvertSetValue, ConvertCopyFixpipe, ConvertStoreWithTranspose, ConvertGather,
            ConvertScatter
            //
            >(converter, context);
        auto op = getOperation();
        if (applyPartialConversion(op, target, std::move(patterns)).failed())
            signalPassFailure();
        op.walk([this](ascendc::DataCopyOp op) {
            if (!op.getDirection()) {
                op.emitOpError("doesn't have a direction set");
                signalPassFailure();
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerAscTileDataTransferPass()
{
    return std::make_unique<LowerAscTileDataTransferPass>();
}
