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

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_LEGALIZEMATMUL
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

bool checkCorrectMatmulAcc(func::FuncOp funcOp)
{
    bool correct = true;
    funcOp.walk([&correct](asctile::MatmulAccOp matmulAccOp) {
        correct &= isa_and_present<asctile::AccumulatorOp>(matmulAccOp.getAcc().getDefiningOp());
        if (!correct) {
            matmulAccOp.emitError() << "Incorrect use of accumulator in matmul operation.";
        }
    });
    return correct;
}

struct LegalizeMatmulPass : public asctile::impl::LegalizeMatmulBase<LegalizeMatmulPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (!checkCorrectMatmulAcc(funcOp))
            signalPassFailure();
    }
};
} // namespace

std::unique_ptr<Pass> mlir::asctile::createLegalizeMatmulPass() { return std::make_unique<LegalizeMatmulPass>(); }
