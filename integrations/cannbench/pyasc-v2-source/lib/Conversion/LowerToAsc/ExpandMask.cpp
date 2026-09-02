/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Conversion/LowerToAsc/Passes.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_EXPANDMASK
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace asclower;

namespace {

template <typename OpT>
void processMask(func::FuncOp funcOp)
{
    SmallVector<OpT> maskVector;
    funcOp.walk<WalkOrder::PreOrder>([&](OpT op) { maskVector.push_back(op); });
    for (OpT maskOp : maskVector) {
        RewritePatternSet patterns(funcOp.getContext());
        ascendc::populateLowerToL0Patterns(patterns);
        walkAndApplyPatterns(maskOp, std::move(patterns));
        auto updateMask = [&](auto l0Op) {
            OpBuilder builder(l0Op);
            auto loc = l0Op.getLoc();
            if (auto other = maskOp.getOther()) {
                ascir::ConstantOpBuilder consts(builder);
                Value scalar = convertScalarToDtype(builder, loc, other, getElementTypeOrSelf(l0Op.getDst()), false);
                builder.create<ascendc::DuplicateL2Op>(loc, l0Op.getDst(), scalar, consts.i32(0));
            }
            if constexpr (std::is_same_v<OpT, asctile::CountMaskOp>) {
                l0Op.getMaskMutable().assign(maskOp.getCount());
            } else if constexpr (std::is_same_v<OpT, asctile::BitwiseMaskOp>) {
                auto mask = builder.create<emitasc::MaskOp>(loc, maskOp.getHighBits(), maskOp.getLowBits());
                l0Op.getMaskMutable().assign(mask);
            } else {
                llvm_unreachable("not implemented");
            }
            l0Op->setAttr(ascendc::attr::maskSet, UnitAttr::get(l0Op.getContext()));
        };
        maskOp.walk([&](ascendc::UnaryL0Op uOp) { updateMask(uOp); });
        maskOp.walk([&](ascendc::BinaryL0Op bOp) { updateMask(bOp); });
        maskOp.walk([&](ascendc::VecScalarL0Op bOp) { updateMask(bOp); });
        for (auto& innerOp : llvm::make_early_inc_range(maskOp.getRegion().front().without_terminator())) {
            innerOp.moveBefore(maskOp);
        }
        maskOp.erase();
    }
}

struct ExpandMaskPass : public asclower::impl::ExpandMaskBase<ExpandMaskPass> {
    void runOnOperation() override
    {
        auto funcOp = getOperation();
        processMask<asctile::CountMaskOp>(funcOp);
        processMask<asctile::BitwiseMaskOp>(funcOp);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createExpandMaskPass() { return std::make_unique<ExpandMaskPass>(); }
