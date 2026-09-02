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
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_WRAPCVGROUPS
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

enum class ComputeUnit {
    Cube,
    Vector,
    Neither,
};

ComputeUnit classifyByTileType(Type type)
{
    if (auto tileType = dyn_cast<LocalTensorType>(type))
        return tileType.getLoc() != TensorLocation::UB ? ComputeUnit::Cube : ComputeUnit::Vector;
    return ComputeUnit::Neither;
}

ComputeUnit classifyOperation(Operation* op)
{
    if (isa<CubeGroupOp, VectorGroupOp>(op))
        return ComputeUnit::Neither;
    // SyncAll can be moved inside IS_AIV block only if has IsAIVOnly == true.
    if (isa<ascendc::SyncAllHardOp>(op))
        return ComputeUnit::Vector;
    if (isa<LoadOp>(op))
        return classifyByTileType(op->getResult(0).getType());
    if (isa<StoreOp, CopyOp, StoreFixpipeOp, CopyFixpipeOp, AtomicRMWOp>(op))
        return classifyByTileType(op->getOperand(0).getType());
    if (!isa<arith::ArithDialect, asctile::AscTileDialect, math::MathDialect, tensor::TensorDialect>(op->getDialect()))
        return ComputeUnit::Neither;
    for (auto type : llvm::concat<Type>(op->getResultTypes(), op->getOperandTypes()))
        if (auto unit = classifyByTileType(type); unit != ComputeUnit::Neither)
            return unit;
    return ComputeUnit::Neither;
}

void wrapSingleOp(Operation* op, ComputeUnit unit, OpBuilder& builder)
{
    SmallVector<Value> inputValues;
    for (auto opnd : op->getOperands())
        if (isa<LocalTensorType>(opnd.getType()))
            inputValues.push_back(opnd);
    SmallVector<Type> resultTypes(op->getResultTypes());
    builder.setInsertionPoint(op);
    Location loc = op->getLoc();
    Operation* blockOp = (unit == ComputeUnit::Cube) ? builder.create<CubeGroupOp>(loc, resultTypes, inputValues) :
                                                       builder.create<VectorGroupOp>(loc, resultTypes, inputValues);
    Block* newBlock = &blockOp->getRegion(0).emplaceBlock();
    builder.setInsertionPointToEnd(newBlock);
    Operation* clonedOp = builder.clone(*op);
    builder.create<YieldOp>(loc, clonedOp->getResults());
    for (auto [i, result] : llvm::enumerate(op->getResults()))
        result.replaceAllUsesWith(blockOp->getResult(i));
    op->erase();
}

struct WrapCVGroupsPass : public asctile::impl::WrapCVGroupsBase<WrapCVGroupsPass> {
    void runOnOperation() override
    {
        SmallVector<Operation*> ops;
        getOperation().walk([&](Operation* op) {
            if (op->hasTrait<OpTrait::IsTerminator>())
                return;
            ops.push_back(op);
        });
        OpBuilder builder(getOperation().getContext());
        for (auto* op : ops) {
            ComputeUnit unit = classifyOperation(op);
            if (unit != ComputeUnit::Neither) {
                wrapSingleOp(op, unit, builder);
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createWrapCVGroupsPass() { return std::make_unique<WrapCVGroupsPass>(); }
