/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===----------------------------------------------------------------------===//
// ReuseTensorAllocation Pass
//===----------------------------------------------------------------------===//
//
// Reduces on-chip memory by reusing freed tensor allocations. Covers VECCALC (UB),
// A1 (L1), A2 (L0A), B2 (L0B), and CO1 (L0C) positions. Reuse is restricted to
// tensors of the same position, since each position maps to a distinct hardware buffer.
//
// Reuse decision:
// 1. Input/output tensor restriction: if both tensors are I/O and both have unroll_iter
//    attributes with different values, they cannot be reused (each iteration loads/stores
//    different data). If either tensor doesn't have unroll_iter, no restriction applies.
// 2. Lifetime overlap: top's endLife must precede bottom's beginLife in program order.
//    Uses opPrecedes with DominanceInfo to correctly handle operations in different blocks.
// 3. Same-op reuse: when top's endLife and bottom's beginLife are the same compute op,
//    hardware reads all inputs before writing the output — safe to share one allocation.
//
// User collection traverses CastOpInterface and LocalTensorSubIndexOp chains to find
// indirect users (ops that use a view of the allocation). View-producing ops themselves
// are skipped during read/write classification.
//
// Loop-carried values (scf.yield in ForOp/WhileOp, scf.condition in WhileOp) are excluded
// from the reuse list — their data persists across all iterations.
//
//===----------------------------------------------------------------------===//

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <unordered_map>
#include <unordered_set>

#define DEBUG_TYPE "ascendc-reuse-tensor-allocation"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_REUSETENSORALLOCATION
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

constexpr const char* const eraseMeAttr = "asc.erase_me";

using TensorOp = ascendc::LocalTensorAutoOp;
using Color = int64_t;
using ColorMap = std::unordered_map<TensorOp, Color, PointerLikeTypeHash<TensorOp>>;
using ColorSet = std::unordered_set<Color>;

struct LifetimeInfo {
    Operation* endLife = nullptr;
    Operation* beginLife = nullptr;
};

// Returns the actual allocation size matching AllocateTensor's sizing rules:
// cube-block-aligned for A1/A2/B2, raw byte size for VECCALC/CO1.
int64_t getAllocationSize(TensorOp op)
{
    auto shapedType = cast<ShapedType>(op.getType());
    auto pos = op.getPosition();
    if (pos == ascendc::TPosition::A1 || pos == ascendc::TPosition::B1 || pos == ascendc::TPosition::A2 ||
        pos == ascendc::TPosition::B2)
        return ascendc::getTypeSizeCubeBlockAlign(shapedType, pos);
    return ascendc::getTypeSize(op.getType());
}

bool isTensorGreaterOrEqual(TensorOp lhs, TensorOp rhs) { return getAllocationSize(lhs) >= getAllocationSize(rhs); }

bool isReusablePosition(ascendc::TPosition pos)
{
    // VECIN, VECOUT also reusable but for the same unroll factor
    return pos == ascendc::TPosition::VECCALC || pos == ascendc::TPosition::A1 || pos == ascendc::TPosition::B1 ||
           pos == ascendc::TPosition::C1 || pos == ascendc::TPosition::A2 || pos == ascendc::TPosition::B2 ||
           pos == ascendc::TPosition::CO1;
}

bool isReusable(TensorOp op) { return isReusablePosition(op.getPosition()) && op.getType().hasStaticShape(); }

std::optional<int64_t> getReuseGroup(Operation* op)
{
    if (auto iter = op->getAttrOfType<IntegerAttr>(ascendc::attr::reuseGroup))
        return iter.getValue().getSExtValue();
    return std::nullopt;
}

void markForErase(TensorOp op) { op->setAttr(eraseMeAttr, UnitAttr::get(op.getContext())); }

// Examples:
// simple lifetime:
// {
//  write(a) <- first use       ┐
// ...                          | lifetime
//  read(a) <- last use         ┘
// }
// nested recursive lifetime:
// write(a) <- first use      ┐
// for(...) {                 |
//   read(a) <- last use      | lifetime (because recursive effect memory)
//   ...                      |
//   yield                    ┘
// }
void adjustLifetime(Operation*& firstOp, Operation*& lastOp, DominanceInfo& di)
{
    auto* commonBlock = di.findNearestCommonDominator(firstOp->getBlock(), lastOp->getBlock());
    Block* stopBlock = commonBlock;
    Operation* op = lastOp;

    auto process = [&](Operation* op) {
        if (auto forOp = dyn_cast<scf::ForOp>(op))
            lastOp = forOp.getBody()->getTerminator();
        if (auto whileOp = dyn_cast<scf::WhileOp>(op))
            lastOp = whileOp.getBody()->getTerminator();
    };

    process(op);
    while (op->getBlock() != stopBlock) {
        op = op->getParentOp();
        process(op);
    }
}

LifetimeInfo computeLifetimeInfo(TensorOp tensorOp, DominanceInfo& di)
{
    LifetimeInfo info;
    SmallVector<Operation*> users = collectAllUsers(tensorOp);
    TensorOp root = ascendc::getAllocationRoot(tensorOp.getResult());
    llvm::stable_sort(users, [&](Operation* lhs, Operation* rhs) { return ascendc::opPrecedes(lhs, rhs, di); });
    info.beginLife = users.front();
    info.endLife = users.back();
    adjustLifetime(info.beginLife, info.endLife, di);
    return info;
}

bool intersectLifetime(const LifetimeInfo& lhs, const LifetimeInfo& rhs, DominanceInfo& di)
{
    bool f0 = ascendc::opPrecedes(rhs.beginLife, lhs.endLife, di);
    bool f1 = ascendc::opPrecedes(lhs.beginLife, rhs.endLife, di);
    return f0 && f1;
}

bool hasShareLoopAncestor(Operation* lhs, Operation* rhs)
{
    constexpr int nestedLevel = 2;
    std::unordered_set<Operation*> lhsLoops, rhsLoops;
    std::vector<Operation*> results;
    scf::ForOp forOp = lhs->getParentOfType<scf::ForOp>();
    for (int i = nestedLevel; i > 0 && forOp; --i) {
        lhsLoops.insert(forOp);
        forOp = forOp->getParentOfType<scf::ForOp>();
    }
    forOp = rhs->getParentOfType<scf::ForOp>();
    for (int i = nestedLevel; i > 0 && forOp; --i) {
        rhsLoops.insert(forOp);
        forOp = forOp->getParentOfType<scf::ForOp>();
    }
    std::set_intersection(
        lhsLoops.begin(), lhsLoops.end(), rhsLoops.begin(), rhsLoops.end(), std::back_inserter(results));
    return !results.empty();
}

bool isWriteToAllocation(Operation* op, TensorOp root)
{
    if (auto dstOp = dyn_cast<ascendc::OpWithDst>(op)) {
        for (Value dst : dstOp.getDstTensors()) {
            auto dstRoot = ascendc::getAllocationRoot(dst);
            if (dstRoot && dstRoot == root)
                return true;
        }
    }
    return false;
}

bool isReadToAllocation(Operation* op, TensorOp root)
{
    if (auto srcOp = dyn_cast<ascendc::OpWithSrc>(op)) {
        for (Value src : srcOp.getSrcTensors()) {
            auto srcRoot = ascendc::getAllocationRoot(src);
            if (srcRoot && srcRoot == root)
                return true;
        }
    }
    return false;
}

bool canReuse(TensorOp top, TensorOp bottom, DominanceInfo& di)
{
    // On-the-fly computation avoids stale data from prior reuses.
    LifetimeInfo topInfo = computeLifetimeInfo(top, di);
    LifetimeInfo bottomInfo = computeLifetimeInfo(bottom, di);
    if (ascendc::opPrecedes(bottomInfo.beginLife, topInfo.beginLife, di) ||
        (topInfo.beginLife == bottomInfo.beginLife) && ascendc::opPrecedes(bottomInfo.endLife, topInfo.endLife, di)) {
        std::swap(top, bottom);
        std::swap(topInfo, bottomInfo);
    }
    bool isTopInOut = top.getInput() || top.getOutput();
    auto iter1 = getReuseGroup(top);
    bool isBottomInOut = bottom.getInput() || bottom.getOutput();
    auto iter2 = getReuseGroup(bottom);
    if (hasShareLoopAncestor(topInfo.beginLife, bottomInfo.beginLife) &&
        (iter1 && iter2 && iter1 != iter2 && (isTopInOut || isBottomInOut))) {
        LLVM_DEBUG(llvm::dbgs() << "reject: in or out" << "\n");
        return false;
    }
    // Strict intersection without intersect borders
    bool flag = intersectLifetime(topInfo, bottomInfo, di);
    if (flag) {
        LLVM_DEBUG(llvm::dbgs() << "  reject: strict lifetime overlap\n");
        return false;
    }
    if (topInfo.endLife == bottomInfo.beginLife) {
        if (top.getType().getElementType() != bottom.getType().getElementType()) {
            LLVM_DEBUG(llvm::dbgs() << "  reject: same-op element type mismatch\n");
            return false;
        }
        if (!(isReadToAllocation(topInfo.endLife, top) && isWriteToAllocation(topInfo.endLife, bottom))) {
            LLVM_DEBUG(llvm::dbgs() << "  reject: invalid read-write dependency\n");
            return false;
        }
        // Only allow same-op reuse for operations marked with OpWithReusableSrc.
        // Operations like BroadcastOp that read sources multiple times are not marked
        // and will be rejected here.
        if (!isa<ascendc::OpWithReusableSrc>(topInfo.endLife)) {
            LLVM_DEBUG(llvm::dbgs() << "  reject: operation not marked with OpWithReusableSrcInterface\n");
            return false;
        }
    }
    return true;
}

auto getConflictedTensors(ArrayRef<TensorOp> allTensors, TensorOp op, DominanceInfo& di)
{
    SmallVector<TensorOp> conflicted;
    for (auto tensor : allTensors) {
        if (tensor == op)
            continue;
        if (!canReuse(op, tensor, di))
            conflicted.emplace_back(tensor);
    }
    return conflicted;
};

auto getColorsSet(const ColorMap& colors, ArrayRef<TensorOp> allTensors)
{
    ColorSet neighbors;
    for (TensorOp tensor : allTensors) {
        if (auto it = colors.find(tensor); it != colors.end())
            neighbors.insert(it->second);
    }
    return neighbors;
}

Color getFirstFree(const ColorSet& colors)
{
    int freeColor = 0;
    while (colors.count(freeColor)) {
        ++freeColor;
    }
    return freeColor;
}

void processTensorList(ArrayRef<TensorOp> allTensors, SmallVectorImpl<TensorOp>& tensorList, DominanceInfo& di)
{
    ColorMap colors;
    LLVM_DEBUG(llvm::dbgs() << "Processing tensor list with " << tensorList.size() << " tensors\n");
    while (!tensorList.empty()) {
        TensorOp topTensor = tensorList.pop_back_val();
        LLVM_DEBUG(llvm::dbgs() << "Trying coloring: " << topTensor << "\n");
        Color freeColor = getFirstFree(getColorsSet(colors, getConflictedTensors(allTensors, topTensor, di)));
        LLVM_DEBUG(llvm::dbgs() << "Set Color: " << freeColor << "\n");
        colors.insert({topTensor, freeColor});
    }

    SmallVector<std::pair<TensorOp, Color>> colorVec;
    int64_t colorSize = -1;
    for (auto pa : colors) {
        colorVec.push_back(pa);
        colorSize = std::max(colorSize, pa.second + 1);
    }

    llvm::sort(colorVec, [&](const std::pair<TensorOp, Color>& x, const std::pair<TensorOp, Color>& y) {
        return ascendc::opPrecedes(x.first, y.first, di);
    });

    SmallVector<SmallVector<TensorOp>> tensorGroups; // for stability
    tensorGroups.resize(colorSize);
    for (auto& [tensor, color] : colorVec) {
        tensorGroups[color].push_back(tensor);
    }

    for (int color = 0; color < colorSize; ++color) {
        auto& tensors = tensorGroups[color];
        TensorOp bigger = tensors[0];
        Block* block = bigger->getBlock();
        for (auto tensor : tensors) {
            block = di.findNearestCommonDominator(block, tensor->getBlock());
            if (isTensorGreaterOrEqual(tensor, bigger)) {
                bigger = tensor;
            }
        }
        bigger->moveBefore(block, block->begin());
        for (auto reused : tensors) {
            if (reused == bigger)
                continue;
            OpBuilder builder(reused);
            builder.setInsertionPointAfter(reused);
            auto castOp =
                builder.create<ascendc::LocalTensorReinterpretCastOp>(reused->getLoc(), reused.getType(), bigger);
            reused->replaceAllUsesWith(castOp);
            markForErase(reused);
        }
    }
}

struct ReuseTensorAllocationPass : public ascendc::impl::ReuseTensorAllocationBase<ReuseTensorAllocationPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();

        DominanceInfo di(funcOp);
        SmallVector<TensorOp> allTensors;
        std::map<TensorOp, LifetimeInfo> lifetimeInfoMap;
        funcOp.walk<WalkOrder::PreOrder>([&](TensorOp tensor) {
            if (tensor->getUsers().empty() || !isReusable(tensor))
                return;
            lifetimeInfoMap[tensor] = computeLifetimeInfo(tensor, di);
            allTensors.push_back(tensor);
        });

        llvm::stable_sort(allTensors, [&](TensorOp lhs, TensorOp rhs) {
            auto lhsSize = getAllocationSize(lhs);
            auto rhsSize = getAllocationSize(rhs);
            if (lhsSize != rhsSize)
                return lhsSize > rhsSize;
            Operation* lhsFreed = lifetimeInfoMap[lhs].endLife;
            Operation* rhsFreed = lifetimeInfoMap[rhs].endLife;
            return ascendc::opPrecedes(rhsFreed, lhsFreed, di);
        });

        std::map<ascendc::TPosition, SmallVector<TensorOp>> filtered;
        LLVM_DEBUG(llvm::dbgs() << "Lifetimes:\n");
        for (auto tensor : allTensors) {
            LLVM_DEBUG(llvm::dbgs() << "tensor: ");
            LLVM_DEBUG(tensor->dump());
            LLVM_DEBUG(llvm::dbgs() << "beginLife: ");
            LLVM_DEBUG(lifetimeInfoMap[tensor].beginLife->dump());
            LLVM_DEBUG(llvm::dbgs() << "endLife: ");
            LLVM_DEBUG(lifetimeInfoMap[tensor].endLife->dump());
            filtered[tensor.getPosition()].push_back(tensor);
        }
        for (auto& [position, tensors] : filtered) {
            std::reverse(tensors.begin(), tensors.end());
            auto order = tensors;
            processTensorList(tensors, order, di);
        }
        funcOp.walk([](TensorOp op) {
            if (op->hasAttr(eraseMeAttr))
                op.erase();
        });
        funcOp.walk([](Operation* op) { op->removeAttr(ascendc::attr::reuseGroup); });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createReuseTensorAllocationPass()
{
    return std::make_unique<ReuseTensorAllocationPass>();
}
