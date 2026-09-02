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
#include "ascir/Dialect/AscTile/Utils/Attributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_MARKMATMULACCWITHBIAS
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;

namespace {

struct MarkMatmulAccWithBiasPass : public asctile::impl::MarkMatmulAccWithBiasBase<MarkMatmulAccWithBiasPass> {
    void runOnOperation() override
    {
        getOperation().walk([](asctile::MatmulAccOp mmadOp) {
            if (auto acc = dyn_cast_if_present<asctile::AccumulatorOp>(mmadOp.getAcc().getDefiningOp())) {
                if (acc.getBias())
                    mmadOp->setAttr(asctile::attr::hasBias, UnitAttr::get(mmadOp.getContext()));
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createMarkMatmulAccWithBiasPass()
{
    return std::make_unique<MarkMatmulAccWithBiasPass>();
}
