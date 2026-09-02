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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_RESOLVEAUTOLOCATION
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

using TL = TensorLocation;
using LocRange = ArrayRef<TL>;

LocalTensorType withLoc(Type type, TL loc)
{
    using LTType = LocalTensorType;
    auto tensor = cast<LTType>(type);
    return cast<LTType>(LTType::get(tensor.getShape(), tensor.getElementType(), loc));
}

std::optional<TL> findRequiredLoc(Value value)
{
    DenseSet<TL> locs;
    for (auto* user : value.getUsers()) {
        auto castOp = dyn_cast<tensor::CastOp>(user);
        if (!castOp)
            return std::nullopt;
        locs.insert(cast<LocalTensorType>(castOp.getResult().getType()).getLoc());
    }
    if (locs.size() != 1)
        return std::nullopt;
    return *locs.begin();
}

LogicalResult acceptResultLoc(PatternRewriter& rewriter, OpResult result, LocRange allowedLocs = std::nullopt)
{
    auto oldType = dyn_cast<LocalTensorType>(result.getType());
    if (!oldType || oldType.getLoc() != TL::Auto)
        return failure();
    if (auto loc = findRequiredLoc(result)) {
        if (!allowedLocs.empty() && !llvm::is_contained(allowedLocs, *loc))
            return failure();
        rewriter.modifyOpInPlace(result.getOwner(), [&] { result.setType(withLoc(oldType, *loc)); });
        return success();
    }
    return failure();
}

LogicalResult acceptOperandLoc(PatternRewriter& rewriter, OpOperand* opnd, LocRange allowedLocs = std::nullopt)
{
    auto oldType = dyn_cast<LocalTensorType>(opnd->get().getType());
    if (!oldType || oldType.getLoc() != TL::Auto)
        return failure();
    auto castOp = opnd->get().getDefiningOp<tensor::CastOp>();
    if (!castOp)
        return failure();
    Value tensor = castOp.getOperand();
    if (!allowedLocs.empty() && !llvm::is_contained(allowedLocs, cast<LocalTensorType>(tensor.getType()).getLoc()))
        return failure();
    rewriter.modifyOpInPlace(opnd->getOwner(), [&] { opnd->assign(tensor); });
    return success();
}

template <typename... OpTypes>
struct RequireSameLoc : RewritePattern {
    SmallVector<TL, 4> allowedLocs;
    TL defaultLoc;

    RequireSameLoc(MLIRContext* context, LocRange allowedLocs, TL defaultLoc, PatternBenefit benefit = 1)
        : RewritePattern(MatchAnyOpTypeTag{}, benefit, context), allowedLocs(allowedLocs), defaultLoc(defaultLoc)
    {}

    static LocalTensorType getNewType(Value value, TL loc)
    {
        if (loc == TL::Auto)
            return {};
        auto oldType = dyn_cast<LocalTensorType>(value.getType());
        if (!oldType || oldType.getLoc() == loc)
            return {};
        return withLoc(oldType, loc);
    }

    LogicalResult matchAndRewrite(Operation* op, PatternRewriter& rewriter) const override
    {
        if (!isa<OpTypes...>(op))
            return failure();
        bool modified = false;
        rewriter.startOpModification(op);
        auto requiredLoc = defaultLoc;
        for (auto type : llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
            auto tensor = dyn_cast<LocalTensorType>(type);
            if (!tensor || tensor.getLoc() == TL::Auto)
                continue;
            if (allowedLocs.empty() || llvm::is_contained(allowedLocs, tensor.getLoc())) {
                requiredLoc = tensor.getLoc();
                break;
            }
        }
        for (auto& opnd : op->getOpOperands()) {
            auto newType = getNewType(opnd.get(), requiredLoc);
            if (!newType)
                continue;
            Value newOpnd = rewriter.create<tensor::CastOp>(op->getLoc(), newType, opnd.get());
            opnd.set(newOpnd);
            modified = true;
        }
        rewriter.setInsertionPointAfter(op);
        for (auto result : op->getResults()) {
            auto newType = getNewType(result, requiredLoc);
            if (!newType)
                continue;
            auto castOp = rewriter.create<tensor::CastOp>(op->getLoc(), result.getType(), result);
            result.setType(newType);
            rewriter.replaceAllUsesExcept(result, castOp.getResult(), castOp);
            modified = true;
        }
        if (modified)
            rewriter.finalizeOpModification(op);
        else
            rewriter.cancelOpModification(op);
        return success(modified);
    }
};

template <typename OpT, OpOperand& (OpT::*operandAccessor)(), TL... allowedLocs>
struct AcceptSameLoc : OpRewritePattern<OpT> {
    using OpRewritePattern<OpT>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpT op, PatternRewriter& rewriter) const override
    {
        SmallVector<TL> locs{allowedLocs...};
        OpOperand& opnd = (op.*operandAccessor)();
        auto opndLoc = [&opnd]() { return cast<LocalTensorType>(opnd.get().getType()).getLoc(); };
        auto oldType = op.getType();
        TL resultLoc = oldType.getLoc();
        if (TL curOpndLoc = opndLoc(); curOpndLoc == TL::Auto) {
            if (resultLoc != TL::Auto) {
                assert(curOpndLoc == TL::Auto && "operand tensor location must not be resolved yet");
                auto newType = withLoc(opnd.get().getType(), resultLoc);
                Value newOpnd = rewriter.create<tensor::CastOp>(op.getLoc(), newType, opnd.get());
                rewriter.modifyOpInPlace(op, [&] { opnd.set(newOpnd); });
                return success();
            }
            if (acceptOperandLoc(rewriter, &opnd, locs).failed())
                return failure();
        } else if (resultLoc != TL::Auto) {
            return failure();
        }
        rewriter.modifyOpInPlace(op, [&] {
            auto newType = withLoc(op.getType(), opndLoc());
            op.getResult().setType(newType);
        });
        rewriter.setInsertionPointAfter(op);
        auto castOp = rewriter.create<tensor::CastOp>(op.getLoc(), oldType, op.getResult());
        rewriter.replaceAllUsesExcept(op.getResult(), castOp.getResult(), castOp);
        return success();
    }
};

using AcceptCastLoc = AcceptSameLoc<CastOp, &CastOp::getInMutable, TL::UB, TL::L0C>;
using AcceptReluLoc = AcceptSameLoc<ReluOp, &ReluOp::getOperandMutable, TL::UB, TL::L0C>;
using AcceptReshapeLoc = AcceptSameLoc<ReshapeOp, &ReshapeOp::getInMutable>;
using AcceptTransposeLoc =
    AcceptSameLoc<TransposeOp, &TransposeOp::getOperandMutable, TL::UB, TL::L1, TL::L0A, TL::L0B>;

template <typename OpT, OpOperand& (OpT::*... operandAccessors)()>
struct Traits {
    using Op = OpT;

    static void populateOperands(OpT op, SmallVectorImpl<OpOperand*>& operands)
    {
        (operands.push_back(&(op.*operandAccessors)()), ...);
    }
};

template <typename Traits, unsigned... resultIndices>
struct AcceptAnyLoc : OpRewritePattern<typename Traits::Op> {
    using OpT = typename Traits::Op;
    using OpRewritePattern<OpT>::OpRewritePattern;

    void populateOperands(OpT op, SmallVectorImpl<OpOperand*>& operands) const
    {
        Traits::populateOperands(op, operands);
    }

    void populateResults(OpT op, SmallVectorImpl<OpResult>& results) const
    {
        (results.push_back(op->getOpResult(resultIndices)), ...);
    }

    LogicalResult matchAndRewrite(OpT op, PatternRewriter& rewriter) const override
    {
        auto matchResult = failure();
        SmallVector<OpOperand*, 2> operands;
        populateOperands(op, operands);
        for (auto* opnd : operands)
            if (acceptOperandLoc(rewriter, opnd).succeeded())
                matchResult = success();
        SmallVector<OpResult, 2> results;
        populateResults(op, results);
        for (auto result : results)
            if (acceptResultLoc(rewriter, result).succeeded())
                matchResult = success();
        return matchResult;
    }
};

using AcceptLoadLoc = AcceptAnyLoc<Traits<LoadOp>, 0>;
using AcceptCopyLoc = AcceptAnyLoc<Traits<CopyOp, &CopyOp::getBaseMutable>, 0>;
using AcceptStoreLoc = AcceptAnyLoc<Traits<StoreOp, &StoreOp::getValueMutable>>;
using AcceptSetValueLoc = AcceptAnyLoc<Traits<SetValueOp, &SetValueOp::getValueMutable>>;

struct AcceptForLoc : OpRewritePattern<scf::ForOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ForOp op, PatternRewriter& rewriter) const override
    {
        SmallVector<OpOperand*, 4> initArgs;
        for (auto& arg : op.getInitArgsMutable())
            if (auto tensor = dyn_cast<LocalTensorType>(arg.get().getType()))
                if (tensor.getLoc() == TL::Auto)
                    initArgs.push_back(&arg);
        if (initArgs.empty())
            return failure();
        for (auto* initArg : initArgs) {
            auto blockArg = op.getTiedLoopRegionIterArg(initArg);
            auto loc = findRequiredLoc(blockArg);
            if (!loc)
                return failure();
            auto* yielded = op.getTiedLoopYieldedValue(blockArg);
            auto newType = withLoc(blockArg.getType(), *loc);
            Value newInit = rewriter.create<tensor::CastOp>(initArg->get().getLoc(), newType, initArg->get());
            rewriter.setInsertionPoint(op.getBody()->getTerminator());
            Value newYield = rewriter.create<tensor::CastOp>(yielded->get().getLoc(), newType, yielded->get());
            rewriter.startOpModification(op);
            op.getTiedLoopResult(blockArg).setType(newType);
            initArg->set(newInit);
            blockArg.setType(newType);
            yielded->set(newYield);
            rewriter.finalizeOpModification(op);
        }
        return success();
    }
};

struct AcceptIfLoc : OpRewritePattern<scf::IfOp> {
    using OpRewritePattern::OpRewritePattern;

    static void handleOperands(PatternRewriter& rewriter, MutableArrayRef<OpOperand> operands, TypeRange resultTypes)
    {
        for (auto& opnd : operands) {
            (void)acceptOperandLoc(rewriter, &opnd);
            auto newType = resultTypes[opnd.getOperandNumber()];
            Value oldValue = opnd.get();
            if (oldValue.getType() == newType)
                continue;
            rewriter.setInsertionPoint(opnd.getOwner());
            Value newValue = rewriter.create<tensor::CastOp>(oldValue.getLoc(), newType, oldValue);
            rewriter.modifyOpInPlace(opnd.getOwner(), [&] { opnd.set(newValue); });
        }
    }

    LogicalResult matchAndRewrite(scf::IfOp op, PatternRewriter& rewriter) const override
    {
        if (op.getNumResults() == 0)
            return failure();
        SmallVector<Type> oldTypes(op->getResultTypes());
        for (auto result : op->getOpResults())
            (void)acceptResultLoc(rewriter, result);
        if (oldTypes == op->getResultTypes())
            return failure();
        handleOperands(rewriter, op.thenYield()->getOpOperands(), op->getResultTypes());
        if (op.elseBlock())
            handleOperands(rewriter, op.elseYield()->getOpOperands(), op->getResultTypes());
        return success();
    }
};

struct ReconcileTensorCast : OpRewritePattern<tensor::CastOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::CastOp op, PatternRewriter& rewriter) const override
    {
        Value origin = op.getOperand();
        while (auto castOp = origin.getDefiningOp<tensor::CastOp>())
            origin = castOp.getOperand();
        if (origin == op.getOperand())
            return failure();
        if (origin.getType() == op.getType())
            rewriter.replaceOp(op, origin);
        else
            rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), origin);
        return success();
    }
};

void populateFirstStage(RewritePatternSet& patterns)
{
    auto* context = patterns.getContext();
    patterns.add<
        AcceptLoadLoc, AcceptCopyLoc, AcceptStoreLoc, AcceptSetValueLoc, AcceptCastLoc, AcceptReluLoc, AcceptReshapeLoc,
        AcceptTransposeLoc, AcceptForLoc, AcceptIfLoc, ReconcileTensorCast>(context);
}

void populateSecondStage(RewritePatternSet& patterns)
{
    enum : unsigned { LowBenefit = 1, HighBenefit = 10 };
    auto* context = patterns.getContext();
    patterns.add<AcceptCopyLoc, AcceptSetValueLoc, AcceptForLoc, AcceptIfLoc, ReconcileTensorCast>(context, LowBenefit)
        .add<RequireSameLoc<LoadOp, StoreOp>>(context, std::nullopt, TL::UB, LowBenefit)
        .add<RequireSameLoc<CastOp, ReluOp>>(context, LocRange{TL::UB, TL::L0C}, TL::UB, HighBenefit)
        .add<RequireSameLoc<ReshapeOp>>(context, std::nullopt, TL::UB, HighBenefit)
        .add<RequireSameLoc<TransposeOp>>(context, LocRange{TL::UB, TL::L1, TL::L0A, TL::L0B}, TL::UB, HighBenefit);
}

struct ResolveAutoLocationPass : public asctile::impl::ResolveAutoLocationBase<ResolveAutoLocationPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = &getContext();
        for (auto populate : {populateFirstStage, populateSecondStage}) {
            RewritePatternSet patterns(context);
            populate(patterns);
            if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed()) {
                signalPassFailure();
                return;
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createResolveAutoLocationPass()
{
    return std::make_unique<ResolveAutoLocationPass>();
}
