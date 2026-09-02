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
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_MERGECVGROUPS
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

template <typename GroupOp>
struct MergeAdjacentGroups : OpRewritePattern<GroupOp> {
    using OpRewritePattern<GroupOp>::OpRewritePattern;

    static SmallVector<GroupOp> collectGroupsNearby(GroupOp group)
    {
        Block* parentBlock = group->getBlock();

        auto revIt = group->getReverseIterator();
        while (revIt != parentBlock->rend() && isa<GroupOp>(*revIt))
            ++revIt;
        auto firstGroupIt = (--revIt)->getIterator();

        auto notGroupIt = group->getIterator();
        do
            ++notGroupIt;
        while (notGroupIt != parentBlock->end() && isa<GroupOp>(*notGroupIt));

        return llvm::map_to_vector(
            llvm::make_range(firstGroupIt, notGroupIt), [](Operation& op) { return cast<GroupOp>(&op); });
    }

    static GroupOp mergeGroups(const SmallVector<GroupOp>& groups, PatternRewriter& rewriter)
    {
        Operation* firstGroup = groups.front();
        Location loc = firstGroup->getLoc();
        DenseSet<Value> intermediateValues;
        for (Operation* group : groups)
            for (Value result : group->getResults())
                intermediateValues.insert(result);
        SmallVector<Value> externalOperands;
        DenseSet<Value> seenOperands;
        for (Operation* group : groups)
            for (Value operand : group->getOperands()) {
                if (intermediateValues.contains(operand))
                    continue;
                if (seenOperands.insert(operand).second)
                    externalOperands.push_back(operand);
            }
        DenseSet<Operation*> groupSet(groups.begin(), groups.end());
        SmallVector<Value> externalResults;
        SmallVector<Type> resultTypes;
        for (Operation* group : groups)
            for (Value result : group->getResults())
                if (llvm::any_of(
                        result.getUses(), [&](OpOperand& use) { return !groupSet.contains(use.getOwner()); })) {
                    externalResults.push_back(result);
                    resultTypes.push_back(result.getType());
                }
        rewriter.setInsertionPoint(firstGroup);
        auto mergedGroup = rewriter.create<GroupOp>(loc, resultTypes, externalOperands);
        Block* mergedBlock = rewriter.createBlock(&mergedGroup->getRegion(0));
        IRMapping mapping;
        rewriter.setInsertionPointToEnd(mergedBlock);
        for (Operation* group : groups) {
            Block& groupBlock = group->getRegion(0).front();
            auto yieldOp = cast<asctile::YieldOp>(groupBlock.getTerminator());
            for (Operation& innerOp : groupBlock.without_terminator()) {
                Operation* cloned = rewriter.clone(innerOp, mapping);
                for (auto [i, result] : llvm::enumerate(innerOp.getResults()))
                    mapping.map(result, cloned->getResult(i));
            }
            for (auto [i, result] : llvm::enumerate(group->getResults())) {
                Value yieldOperand = yieldOp.getOperand(i);
                Value mappedYieldOperand = mapping.lookupOrDefault(yieldOperand);
                mapping.map(result, mappedYieldOperand);
            }
        }
        SmallVector<Value> yieldOperands;
        for (Value externalResult : externalResults)
            yieldOperands.push_back(mapping.lookup(externalResult));
        rewriter.create<asctile::YieldOp>(loc, yieldOperands);
        for (Value intermediateValue : intermediateValues)
            if (!llvm::is_contained(externalResults, intermediateValue) && mapping.contains(intermediateValue))
                rewriter.replaceAllUsesWith(intermediateValue, mapping.lookup(intermediateValue));
        for (auto [i, externalResult] : llvm::enumerate(externalResults))
            rewriter.replaceAllUsesWith(externalResult, mergedGroup->getResult(i));
        for (Operation* group : groups)
            rewriter.eraseOp(group);
        return mergedGroup;
    }

    LogicalResult matchAndRewrite(GroupOp op, PatternRewriter& rewriter) const override
    {
        auto nearGroups = collectGroupsNearby(op);
        if (nearGroups.size() <= 1)
            return failure();
        return success(mergeGroups(nearGroups, rewriter));
    }
};

template <typename GroupOp>
struct AbsorbOpsBeforeGroup : OpRewritePattern<GroupOp> {
    using OpRewritePattern<GroupOp>::OpRewritePattern;

    static bool isAbsorbableOpType(Operation* op)
    {
        // Constants are not absorbed: they are foldable, so the greedy driver would
        // fold/hoist them back out of the group, breaking the fixpoint.
        return !isa<asctile::CubeGroupOp, asctile::VectorGroupOp>(op) && !op->mightHaveTrait<OpTrait::ConstantLike>() &&
               (isa<scf::IfOp, scf::ForOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>() || isPure(op));
    }

    // Whether `op` (located outside the group body block) can be absorbed into
    // GroupOp. `inside` is the set of operations considered to live inside the
    // group body after absorption (the current body ops plus already chosen
    // candidates and their nested ops); all uses of `op`'s results must land there.
    static bool canBeAbsorbed(Operation* op, const DenseSet<Operation*>& inside)
    {
        if (!op || inside.contains(op))
            return false;
        for (Value result : op->getResults())
            if (llvm::any_of(result.getUsers(), [&inside](Operation* user) { return !inside.contains(user); }))
                return false;
        // Nested operations (e.g. body of an scf.if/scf.for candidate) must be of an
        // absorbable type too; the use-based criterion does not apply to them, since
        // their uses live inside the candidate rather than in the group body block.
        auto walkRes = op->walk([&](Operation* innerOp) {
            if (!isAbsorbableOpType(innerOp))
                return WalkResult::interrupt();
            return WalkResult::advance();
        });
        return !walkRes.wasInterrupted();
    }

    static SmallVector<Operation*> getAbsorbableOps(Operation* groupOp)
    {
        // `inside` tracks every operation considered to live in the group body after
        // absorption: the current body ops, plus chosen candidates and their nested
        // ops. Growing it as candidates are discovered makes the transitive closure
        // correct: an op becomes absorbable once its only remaining outside user
        // gets absorbed (and thus enters `inside`).
        DenseSet<Operation*> inside;
        SmallVector<Operation*, 16> worklist;
        auto markInside = [&](Operation* op) {
            op->walk([&](Operation* innerOp) {
                inside.insert(innerOp);
                worklist.push_back(innerOp);
            });
        };
        markInside(groupOp);
        SetVector<Operation*> candidates;
        while (!worklist.empty()) {
            Operation* cur = worklist.pop_back_val();
            for (Value operand : cur->getOperands()) {
                Operation* defOp = operand.getDefiningOp();
                if (canBeAbsorbed(defOp, inside)) {
                    Operation* parentOp = defOp->getParentOp();
                    if (parentOp && !candidates.contains(parentOp)) {
                        // Add only those operations are added as candidates for absorption
                        // which are dominate group block. If an analyzed operation is placed
                        // in a block of another operation, it is assumed that
                        // parent operation was traversed earlier and is already in the list
                        // of candidates.
                        candidates.insert(defOp);
                    }
                    markInside(defOp);
                }
            }
        }

        return llvm::to_vector(topologicalSort(candidates));
    }

    LogicalResult matchAndRewrite(GroupOp op, PatternRewriter& rewriter) const override
    {
        // Absorb every operation whose uses are all located inside it (transitively).
        SmallVector<Operation*> opsToAbsorb = getAbsorbableOps(op);
        if (opsToAbsorb.empty())
            return failure();
        Block* groupBlock = &op->getRegion(0).front();
        // Move candidates into the group body block in reverse post-order of the
        // candidate operand-DAG so that definitions precede uses (and precede the
        // existing body ops). `orderedCandidates` is post-order (defs first); moving
        // to the block begin in reverse order yields post-order in the block.
        for (Operation* c : llvm::reverse(opsToAbsorb))
            rewriter.moveOpBefore(c, groupBlock, groupBlock->begin());
        return success();
    }
};

template <typename GroupOp>
struct InterchangeWithParentOp : OpRewritePattern<GroupOp> {
    using OpRewritePattern<GroupOp>::OpRewritePattern;

    static SmallVector<GroupOp, 2> getPerfectlyNestedGroups(GroupOp group)
    {
        SmallVector<GroupOp, 2> nestedGroups;
        Operation* parentOp = group->getParentOp();
        for (Region& region : parentOp->getRegions())
            for (Block& block : region.getBlocks()) {
                auto op = dyn_cast<GroupOp>(&block.front());
                if (!op || !llvm::hasSingleElement(block.without_terminator()))
                    return {};
                nestedGroups.push_back(op);
            }
        return nestedGroups;
    }

    static bool interchangeWithParentOp(scf::IfOp ifOp, ArrayRef<GroupOp> groups, PatternRewriter& rewriter)
    {
        Location loc = ifOp->getLoc();

        ValueSet newGroupOperandSet;
        for (auto& g : groups)
            newGroupOperandSet.insert(g->getOperands().begin(), g->getOperands().end());
        ValueVector newGroupOperands(newGroupOperandSet.begin(), newGroupOperandSet.end());
        rewriter.setInsertionPoint(ifOp);
        Operation* outerGroup = rewriter.create<GroupOp>(loc, ifOp.getResults().getTypes(), newGroupOperands);

        Block* outerBlock = rewriter.createBlock(&outerGroup->getRegion(0));
        rewriter.setInsertionPointToEnd(outerBlock);
        auto newIfOp =
            rewriter.create<scf::IfOp>(loc, ifOp.getResults().getTypes(), ifOp.getCondition(), groups.size() > 1);
        rewriter.modifyOpInPlace(newIfOp, [&] { newIfOp->setAttrs(ifOp->getAttrDictionary()); });
        rewriter.create<asctile::YieldOp>(loc, newIfOp->getResults());

        for (auto [olfIfRegion, newIfRegion] : llvm::zip(ifOp->getRegions(), newIfOp->getRegions())) {
            if (olfIfRegion.empty())
                continue;
            Block& oldIfBlock = olfIfRegion.front();
            auto* oldGroup = &oldIfBlock.front();
            auto& oldGroupBlock = oldGroup->getRegion(0).front();
            auto termIt = std::prev(oldGroupBlock.end());
            ValueMap<Value> oldGroupYields;
            for (auto [result, yield] : llvm::zip(oldGroup->getResults(), termIt->getOperands()))
                oldGroupYields[result] = yield;
            SmallVector<Value> newIfBlockYields;
            for (Value oldThenYield : oldIfBlock.getTerminator()->getOperands())
                newIfBlockYields.push_back(oldGroupYields[oldThenYield]);
            Block& newIfBlock = newIfRegion.front();
            // The freshly-built scf.if may have added an implicit yield; drop any
            // pre-existing ops so the block is repopulated with the group body and a
            // new yield below. (Avoid getTerminator(): the block may be empty.)
            while (!newIfBlock.empty())
                rewriter.eraseOp(&newIfBlock.back());
            for (auto& op : llvm::make_early_inc_range(oldGroupBlock.without_terminator()))
                rewriter.moveOpBefore(&op, &newIfBlock, newIfBlock.end());
            rewriter.setInsertionPointToEnd(&newIfBlock);
            rewriter.create<scf::YieldOp>(loc, newIfBlockYields);
        }

        for (auto [newGroupRes, oldIfRes] : llvm::zip(outerGroup->getResults(), ifOp.getResults()))
            rewriter.replaceAllUsesWith(oldIfRes, newGroupRes);
        rewriter.eraseOp(ifOp);

        return true;
    }

    static bool interchangeWithParentOp(scf::ForOp forOp, GroupOp group, PatternRewriter& rewriter)
    {
        Location loc = forOp->getLoc();
        auto initArgs = forOp.getInitArgs();
        ValueMap<Value> iterArgInits;
        for (auto [iterArg, initVal] : llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs()))
            iterArgInits[iterArg] = initVal;
        SmallVector<Value> externalOperands;
        for (Value operand : group->getOperands()) {
            auto opndIter = iterArgInits.find(operand);
            if (opndIter != iterArgInits.end())
                externalOperands.push_back(opndIter->second);
            else
                externalOperands.push_back(operand);
        }
        auto* oldBody = forOp.getBody();
        auto& groupBlock = group->getRegion(0).front();
        auto termIt = std::prev(groupBlock.end());
        ValueMap<Value> groupYields;
        for (auto [result, yield] : llvm::zip(group->getResults(), termIt->getOperands()))
            groupYields[result] = yield;
        SmallVector<Value> newForYields;
        for (Value oldForYield : oldBody->getTerminator()->getOperands())
            newForYields.push_back(groupYields[oldForYield]);
        rewriter.setInsertionPoint(forOp);
        Operation* outerGroup = rewriter.create<GroupOp>(loc, forOp.getResults().getTypes(), externalOperands);
        Block* outerBlock = rewriter.createBlock(&outerGroup->getRegion(0));
        rewriter.setInsertionPointToEnd(outerBlock);
        auto newForOp = rewriter.create<scf::ForOp>(
            loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), forOp.getInitArgs(),
            [&](OpBuilder& b, Location loc, Value, ValueRange) { b.create<scf::YieldOp>(loc, newForYields); });
        rewriter.modifyOpInPlace(newForOp, [&] { newForOp->setAttrs(forOp->getAttrDictionary()); });
        rewriter.create<asctile::YieldOp>(loc, newForOp->getResults());
        auto* newBody = newForOp.getBody();
        for (auto& op : llvm::make_early_inc_range(groupBlock.without_terminator()))
            rewriter.moveOpBefore(&op, newBody, newBody->getTerminator()->getIterator());
        rewriter.replaceAllUsesWith(forOp.getInductionVar(), newForOp.getInductionVar());
        for (auto [newIterArg, oldIterArg] : llvm::zip(newForOp.getRegionIterArgs(), forOp.getRegionIterArgs()))
            rewriter.replaceAllUsesWith(oldIterArg, newIterArg);
        for (auto [newGroupRes, oldForRes] : llvm::zip(outerGroup->getResults(), forOp.getResults()))
            rewriter.replaceAllUsesWith(oldForRes, newGroupRes);
        rewriter.eraseOp(forOp);

        return true;
    }

    LogicalResult matchAndRewrite(GroupOp op, PatternRewriter& rewriter) const override
    {
        SmallVector<GroupOp, 2> nestedGroups = getPerfectlyNestedGroups(op);
        if (nestedGroups.empty())
            return failure();

        Operation* parentOp = op->getParentOp();
        bool modified = false;
        if (auto forOp = dyn_cast_if_present<scf::ForOp>(parentOp)) {
            assert(nestedGroups.size() == 1);
            modified = interchangeWithParentOp(forOp, op, rewriter);
        } else if (auto ifOp = dyn_cast_if_present<scf::IfOp>(parentOp)) {
            assert(nestedGroups.size() <= 2);
            modified = interchangeWithParentOp(ifOp, nestedGroups, rewriter);
        }
        return success(modified);
    }
};

struct MergeCVGroupsPass : public asctile::impl::MergeCVGroupsBase<MergeCVGroupsPass> {
    void runOnOperation() override
    {
        MLIRContext* ctx = &getContext();
        RewritePatternSet patterns(ctx);
        patterns.add<
            MergeAdjacentGroups<CubeGroupOp>, MergeAdjacentGroups<VectorGroupOp>, AbsorbOpsBeforeGroup<CubeGroupOp>,
            AbsorbOpsBeforeGroup<VectorGroupOp>, InterchangeWithParentOp<CubeGroupOp>,
            InterchangeWithParentOp<VectorGroupOp>>(ctx);
        if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createMergeCVGroupsPass() { return std::make_unique<MergeCVGroupsPass>(); }
