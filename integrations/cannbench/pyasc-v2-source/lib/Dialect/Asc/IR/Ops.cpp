/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Utils/CVGroupCanonicalization.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "ascir/Dialect/Asc/IR/AscendCOps.cpp.inc"

using namespace mlir;
using namespace mlir::ascendc;

namespace {

template <typename CVCondOp>
struct InlineNestedGroup : public OpRewritePattern<CVCondOp> {
    using OpRewritePattern<CVCondOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CVCondOp op, PatternRewriter& rewriter) const override
    {
        if (!op->template getParentOfType<CVCondOp>())
            return failure();
        Block* body = op.getBody();
        auto yieldOp = cast<YieldOp>(body->getTerminator());
        rewriter.inlineBlockBefore(body, op);
        rewriter.replaceOp(op, yieldOp.getOperands());
        rewriter.eraseOp(yieldOp);
        return success();
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// GlobalTensorOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalTensorOp::canonicalize(GlobalTensorOp op, PatternRewriter& rewriter)
{
    return ascir::eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// IfAICOp
//===----------------------------------------------------------------------===//

void IfAICOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context)
{
    results.add<
        ascir::EraseEmptyGroup<IfAICOp, YieldOp>, ascir::EraseUnusedOperands<IfAICOp, YieldOp>,
        ascir::EraseUnusedResults<IfAICOp, YieldOp>, InlineNestedGroup<IfAICOp>>(context);
}

//===----------------------------------------------------------------------===//
// IfAIVOp
//===----------------------------------------------------------------------===//

void IfAIVOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context)
{
    results.add<
        ascir::EraseEmptyGroup<IfAIVOp, YieldOp>, ascir::EraseUnusedOperands<IfAIVOp, YieldOp>,
        ascir::EraseUnusedResults<IfAIVOp, YieldOp>, InlineNestedGroup<IfAIVOp>>(context);
}

//===----------------------------------------------------------------------===//
// GlobalTensorSubIndexOp
//===----------------------------------------------------------------------===//

OpFoldResult GlobalTensorSubIndexOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    if (matchPattern(getIndex(), m_Zero()) && getType() == getTensor().getType())
        return getTensor();
    return {};
}

//===----------------------------------------------------------------------===//
// LocalTensorOp
//===----------------------------------------------------------------------===//

LogicalResult LocalTensorOp::canonicalize(LocalTensorOp op, PatternRewriter& rewriter)
{
    return ascir::eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// LocalTensorSubIndexOp
//===----------------------------------------------------------------------===//

OpFoldResult LocalTensorSubIndexOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    if (matchPattern(getIndex(), m_Zero()) && getType() == getTensor().getType())
        return getTensor();
    return {};
}

//===----------------------------------------------------------------------===//
// PipeBarrierOp
//===----------------------------------------------------------------------===//

LogicalResult PipeBarrierOp::canonicalize(PipeBarrierOp op, PatternRewriter& rewriter)
{
    Block* block = op->getBlock();
    auto nextIt = std::next(Block::iterator(op));
    if (nextIt == block->end())
        return failure();
    if (auto nextOp = dyn_cast<ascendc::PipeBarrierOp>(*nextIt)) {
        if (op.getPipe() == Pipe::PIPE_ALL) {
            rewriter.eraseOp(nextOp);
            return success();
        }
        if (op->getAttrs() == nextOp->getAttrs() || nextOp.getPipe() == Pipe::PIPE_ALL) {
            rewriter.eraseOp(op);
            return success();
        }
    }
    return failure();
}

//===----------------------------------------------------------------------===//
// LocalTensorReinterpretCastOp
//===----------------------------------------------------------------------===//

bool LocalTensorReinterpretCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    return inputs.size() == 1 && outputs.size() == 1 && isa<LocalTensorType>(inputs[0]) &&
           isa<LocalTensorType>(outputs[0]);
}

OpFoldResult LocalTensorReinterpretCastOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    Value in = getIn();
    Type resultType = getResult().getType();
    if (in.getType() == resultType)
        return in;
    if (auto defOp = in.getDefiningOp<LocalTensorReinterpretCastOp>()) {
        Value defIn = defOp.getIn();
        if (resultType == defIn.getType())
            return defIn;
        setOperand(defIn);
        return getResult();
    }
    return {};
}

//===----------------------------------------------------------------------===//
// LocalTensorAutoOp
//===----------------------------------------------------------------------===//

LogicalResult LocalTensorAutoOp::canonicalize(LocalTensorAutoOp op, PatternRewriter& rewriter)
{
    return ascir::eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// RegTensorOp
//===----------------------------------------------------------------------===//

LogicalResult RegTensorOp::canonicalize(RegTensorOp op, PatternRewriter& rewriter)
{
    return ascir::eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// AscendCDialect
//===----------------------------------------------------------------------===//

void AscendCDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ascir/Dialect/Asc/IR/AscendCOps.cpp.inc"
        >();
}
