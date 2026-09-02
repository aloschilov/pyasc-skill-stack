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
#include "ascir/Dialect/AscTile/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_PROMOTEPUREOPS
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

class Promoter {
    DominanceInfo dom;

    static bool canPromote(Operation* op) { return isPure(op) && !op->mightHaveTrait<OpTrait::IsTerminator>(); }

    void promote(Operation* op, Block& block)
    {
        for (auto& prevOp : llvm::make_range(block.begin(), Block::iterator(op))) {
            if (llvm::all_of(op->getOperands(), [this, &prevOp](Value opnd) {
                    if (auto* defOp = opnd.getDefiningOp())
                        return ascendc::opPrecedes(defOp, &prevOp, dom);
                    return dom.dominates(opnd, &prevOp);
                })) {
                op->moveBefore(&prevOp);
                return;
            }
        }
    }

public:
    Promoter() = default;
    ~Promoter() = default;

    void runWithinBlock(Block& block)
    {
        SmallVector<Operation*> ops;
        for (auto& op : block) {
            if (canPromote(&op))
                ops.push_back(&op);
        }
        for (auto* op : ops)
            promote(op, block);
    }
};

struct PromotePureOpsPass : public asctile::impl::PromotePureOpsBase<PromotePureOpsPass> {
    void runOnOperation() override
    {
        auto op = getOperation();
        Promoter promoter;
        op.walk([&](Block* block) { promoter.runWithinBlock(*block); });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createPromotePureOpsPass() { return std::make_unique<PromotePureOpsPass>(); }
