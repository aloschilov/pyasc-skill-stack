/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/Utils/CVGroupCanonicalization.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::asctile;

namespace {

template <typename OpT>
OpFoldResult foldCastLike(OpT op, bool allowTransitCast = true)
{
    Value in = op.getIn();
    Type resultType = op.getResult().getType();
    if (in.getType() == resultType)
        return in;
    if (auto defOp = in.getDefiningOp<OpT>()) {
        Value defIn = defOp.getIn();
        if (resultType == defIn.getType())
            return defIn;
        if (!allowTransitCast)
            return {};
        op.setOperand(defIn);
        return op.getResult();
    }
    return {};
}

Type getI1SameShape(Type type)
{
    auto i1Type = IntegerType::get(type.getContext(), 1);
    if (auto shapedType = llvm::dyn_cast<ShapedType>(type))
        return shapedType.cloneWith(std::nullopt, i1Type);
    return i1Type;
}

LogicalResult verifyCVGroupOp(Operation* op)
{
    auto& block = op->getRegion(0).front();
    if (block.getNumArguments() != 0)
        return op->emitOpError("block is not allowed to have arguments");
    auto yieldOperands = cast<asctile::YieldOp>(block.getTerminator()).getOperands();
    auto resultTypes = op->getResultTypes();
    if (yieldOperands.size() != resultTypes.size()) {
        return op->emitOpError("number of yield operands (")
               << yieldOperands.size() << ") must match number of results (" << resultTypes.size() << ")";
    }
    for (auto [i, it] : llvm::enumerate(llvm::zip(yieldOperands.getTypes(), resultTypes))) {
        auto [yieldType, resultType] = it;
        if (yieldType != resultType) {
            return op->emitOpError("yield operand type at index ")
                   << i << " (" << yieldType << ") does not match result type (" << resultType << ")";
        }
    }
    if (op->getParentOfType<CubeGroupOp>() || op->getParentOfType<VectorGroupOp>())
        return op->emitOpError("is not allowed to be nested to other cube/vector group");
    return success();
}

LogicalResult verifyDataAlignment(Operation* op, LocalTensorType type)
{
    if (type.getRank() < 2)
        return success();
    auto itemSize = type.getElementTypeBitWidth() / CHAR_BIT;
    if (itemSize < 1)
        return success();
    if (type.getShape().back() % (ascendc::ubBlockSize / itemSize) != 0)
        return op->emitError() << "Last dimension of a tensor must be aligned by " << ascendc::ubBlockSize
                               << " bytes, got " << type.getShape().back() << " x " << itemSize << " bytes";
    return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// AccumulatorOp
//===----------------------------------------------------------------------===//

LogicalResult AccumulatorOp::canonicalize(AccumulatorOp op, PatternRewriter& rewriter)
{
    return ascir::eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

OpFoldResult BroadcastOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    SplatElementsAttr attr;
    if (!matchPattern(getOperand(), m_Constant(&attr)))
        return {};
    return SplatElementsAttr::get(getType(), attr.getSplatValue<Attribute>());
}

//===----------------------------------------------------------------------===//
// TensorOp
//===----------------------------------------------------------------------===//

LogicalResult TensorOp::verify()
{
    if (getType().getNumDynamicDims() != getSizes().size())
        return emitOpError("must have value in 'sizes' for each dynamic dimension");
    return success();
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

OpFoldResult DimOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    auto index = getIndex();
    auto type = getBase().getType();
    auto dim = type.getDimSize(index);
    if (!ShapedType::isDynamic(dim))
        return IntegerAttr::get(IntegerType::get(getContext(), 32), dim);
    auto tensorOp = getBase().getDefiningOp<TensorOp>();
    if (!tensorOp)
        return OpFoldResult{};
    auto dynamicIndex = type.getDynamicDimIndex(index);
    assert(dynamicIndex < tensorOp.getSizes().size() && "dim index must be less than number of dynamic sizes");
    return tensorOp.getSizes()[dynamicIndex];
}

LogicalResult DimOp::verify()
{
    if (getIndex() >= getBase().getType().getRank())
        return emitOpError("'index' must not exceed the tensor rank");
    return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult CopyOp::canonicalize(CopyOp op, PatternRewriter& rewriter) { return ascir::eraseUnusedOp(op, rewriter); }

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::canonicalize(LoadOp op, PatternRewriter& rewriter) { return ascir::eraseUnusedOp(op, rewriter); }

LogicalResult LoadOp::verify()
{
    auto type = getType();
    if (type.getLoc() == TensorLocation::UB && verifyDataAlignment(getOperation(), type).failed())
        return failure();
    SmallVector<Value> realShape = getRealShape();
    if (realShape.empty())
        return success();
    auto tileShape = type.getShape();
    if (tileShape.size() != realShape.size())
        return emitOpError() << "real_shape must have same size as tensor shape";

    if (auto attr = getOperation()->getAttrOfType<DenseI32ArrayAttr>(asctile::attr::transposeDims)) {
        ArrayRef<int32_t> transposeDims = attr;
        if (transposeDims.size() != tileShape.size())
            return emitOpError() << "transpose_dims must have same rank as tensor";
        SmallVector<Value> tmp;
        for (size_t i = 0; i < tileShape.size(); ++i) {
            auto dim = transposeDims[i];
            tmp.push_back(realShape[dim]);
        }
        tmp.swap(realShape);
    }
    for (auto [realDimValue, tileDim] : llvm::zip_equal(realShape, tileShape)) {
        APInt realDim;
        if (!matchPattern(realDimValue, m_ConstantInt(&realDim)))
            continue;
        if (realDim.getSExtValue() > tileDim)
            return emitOpError() << "real_shape exceeds tensor shape";
    }
    return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify()
{
    auto srcType = getValue().getType();
    if (srcType.getLoc() == TensorLocation::UB && verifyDataAlignment(getOperation(), srcType).failed())
        return failure();
    SmallVector<Value> realShape = getRealShape();
    if (realShape.empty())
        return success();
    auto tileShape = srcType.getShape();
    if (tileShape.size() != realShape.size())
        return emitOpError() << "real_shape must have same size as tensor shape";

    if (auto attr = getOperation()->getAttrOfType<DenseI32ArrayAttr>(asctile::attr::transposeDims)) {
        ArrayRef<int32_t> transposeDims = attr;
        if (transposeDims.size() != tileShape.size())
            return emitOpError() << "transpose_dims must have same rank as tensor";
        SmallVector<Value> tmp;
        for (size_t i = 0; i < tileShape.size(); ++i) {
            auto dim = transposeDims[i];
            tmp.push_back(realShape[dim]);
        }
        tmp.swap(realShape);
    }
    for (auto [realDimValue, tileDim] : llvm::zip_equal(realShape, tileShape)) {
        APInt realDim;
        if (!matchPattern(realDimValue, m_ConstantInt(&realDim)))
            continue;
        if (realDim.getSExtValue() > tileDim)
            return emitOpError() << "real_shape exceeds tensor shape";
    }
    return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;
    auto inType = dyn_cast<LocalTensorType>(inputs.front());
    auto outType = dyn_cast<LocalTensorType>(outputs.front());
    return inType && outType && inType.getLoc() == outType.getLoc() && inType.getShape() == outType.getShape() &&
           inType.getElementType().isIntOrFloat() && outType.getElementType().isIntOrFloat();
}

OpFoldResult CastOp::fold(FoldAdaptor) { return foldCastLike(*this, false); }

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

bool ReshapeOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;
    auto inType = dyn_cast<LocalTensorType>(inputs.front());
    auto outType = dyn_cast<LocalTensorType>(outputs.front());
    return inType && outType && inType.getLoc() == outType.getLoc() &&
           inType.getElementType() == outType.getElementType() && inType.getNumElements() == outType.getNumElements();
}

OpFoldResult ReshapeOp::fold(FoldAdaptor) { return foldCastLike(*this); }

//===----------------------------------------------------------------------===//
// StoreFixpipeOp
//===----------------------------------------------------------------------===//

LogicalResult StoreFixpipeOp::verify()
{
    if (!getQuantize() && getElementTypeOrSelf(getBase()) != getElementTypeOrSelf(getValue())) {
        return emitOpError("failed to verify that all of {base, value} have same element type");
    }
    if (auto realShape = getRealShape(); !realShape.empty()) {
        auto tileShape = getValue().getType().getShape();
        if (tileShape.size() != realShape.size())
            return emitOpError() << "real_shape must have same size as tensor shape";
        for (auto [realDimValue, tileDim] : llvm::zip_equal(realShape, tileShape)) {
            APInt realDim;
            if (!matchPattern(realDimValue, m_ConstantInt(&realDim)))
                continue;
            if (realDim.getSExtValue() > tileDim)
                return emitOpError() << "real_shape exceeds tensor shape";
        }
    }
    return success();
}

//===----------------------------------------------------------------------===//
// CopyFixpipeOp
//===----------------------------------------------------------------------===//

LogicalResult CopyFixpipeOp::canonicalize(CopyFixpipeOp op, PatternRewriter& rewriter)
{
    return ascir::eraseUnusedOp(op, rewriter);
}

LogicalResult CopyFixpipeOp::verify()
{
    if (!getQuantize() && getElementTypeOrSelf(getBase()) != getElementTypeOrSelf(getResult())) {
        return emitOpError("failed to verify that all of {base, result} have same element type");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// AccumulatorOp
//===----------------------------------------------------------------------===//

LogicalResult AccumulatorOp::verify()
{
    auto bias = getBias();
    if (!bias)
        return success();
    auto resultShape = getType().getShape();
    if (resultShape.size() != 2)
        return emitOpError("result must be a 2D tensor");
    if (bias.getType().getShape()[0] != resultShape[1])
        return emitOpError("bias shape must match result's second dimension");
    return success();
}

//===----------------------------------------------------------------------===//
// CubeGroupOp
//===----------------------------------------------------------------------===//

void CubeGroupOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context)
{
    results.add<
        ascir::EraseEmptyGroup<CubeGroupOp, YieldOp>, ascir::EraseUnusedOperands<CubeGroupOp, YieldOp>,
        ascir::EraseUnusedResults<CubeGroupOp, YieldOp>>(context);
}

LogicalResult CubeGroupOp::verify() { return verifyCVGroupOp(*this); }

//===----------------------------------------------------------------------===//
// VectorGroupOp
//===----------------------------------------------------------------------===//

void VectorGroupOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context)
{
    results.add<
        ascir::EraseEmptyGroup<VectorGroupOp, YieldOp>, ascir::EraseUnusedOperands<VectorGroupOp, YieldOp>,
        ascir::EraseUnusedResults<VectorGroupOp, YieldOp>>(context);
}

LogicalResult VectorGroupOp::verify() { return verifyCVGroupOp(*this); }

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify()
{
    if (getType().getLoc() == TensorLocation::UB)
        return verifyDataAlignment(getOperation(), getType());
    return success();
}

//===----------------------------------------------------------------------===//
// AscTileDialect
//===----------------------------------------------------------------------===//

Operation* AscTileDialect::materializeConstant(OpBuilder& builder, Attribute value, Type type, Location loc)
{
    return arith::ConstantOp::materialize(builder, value, type, loc);
}

void AscTileDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ascir/Dialect/AscTile/IR/AscTileOps.cpp.inc"
        >();
}

#define GET_OP_CLASSES
#include "ascir/Dialect/AscTile/IR/AscTileOps.cpp.inc"
