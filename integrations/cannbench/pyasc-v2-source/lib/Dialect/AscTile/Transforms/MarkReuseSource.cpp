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
#include "ascir/Dialect/AscTile/Transforms/Passes.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_MARKREUSESOURCE
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

struct MarkReuseSourcePass : public asctile::impl::MarkReuseSourceBase<MarkReuseSourcePass> {
    template <typename OpT>
    void mark()
    {
        getOperation().walk([](OpT op) -> void {
            bool mark = true;
            auto value = op.getOperand();
            for (auto* user : value.getUsers()) {
                if (user == op)
                    continue;
                if (!ascendc::opPrecedes(user, op)) {
                    mark = false;
                    break;
                }
            }
            if (mark) {
                op->setAttr(asctile::attr::reuseSource, UnitAttr::get(op->getContext()));
            }
        });
    }

    void runOnOperation() override { mark<asctile::ReduceOp>(); }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createMarkReuseSourcePass() { return std::make_unique<MarkReuseSourcePass>(); }
