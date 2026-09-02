/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <unordered_map>

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_ALLOCATETENSOR
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

TPosition normalizePosition(TPosition position)
{
    switch (position) {
        case TPosition::A1:
        case TPosition::A2:
        case TPosition::B2:
        case TPosition::CO1:
        case TPosition::C2:
        case TPosition::VECCALC:
            return position;
        case TPosition::B1:
        case TPosition::C1:
            return TPosition::A1;
        case TPosition::VECIN:
        case TPosition::VECOUT:
            return TPosition::VECCALC;
        default:
            llvm_unreachable("unexpected TPosition value");
    }
}

struct AllocateTensorPass : public ascendc::impl::AllocateTensorBase<AllocateTensorPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        std::unordered_map<ascendc::TPosition, uint32_t> offsets;
        funcOp.walk<WalkOrder::PreOrder>([this, &offsets](LocalTensorAutoOp op) {
            auto type = op.getType();
            if (!type.hasStaticShape()) {
                op.emitOpError() << "must have static shape";
                signalPassFailure();
            }
            OpBuilder builder(op);
            auto position = normalizePosition(op.getPosition());
            uint32_t& addr = offsets[position];
            uint32_t byteSize;
            if (position == TPosition::A1 || position == ascendc::TPosition::A2 || position == ascendc::TPosition::B2)
                byteSize = getTypeSizeCubeBlockAlign(type, op.getPosition());
            else
                byteSize = llvm::alignTo<ubBlockSize>(getTypeSize(type));
            uint32_t tileSize = byteSize / static_cast<uint32_t>(getElementTypeSize(type));
            auto tensor = builder.create<LocalTensorV3Op>(op.getLoc(), type, position, addr, tileSize);
            op->replaceAllUsesWith(tensor);
            op.erase();
            addr += byteSize;
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createAllocateTensorPass() { return std::make_unique<AllocateTensorPass>(); }
