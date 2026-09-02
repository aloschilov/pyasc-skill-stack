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
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/AscVF/Transforms/Passes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_REORDEROPSINVECSCOPE
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

enum class Order { ConstantOp, RegTensorOp, CreateMaskOp, DuplicateOp, Expr, Any };

Order getOrder(Operation* op)
{
    return llvm::TypeSwitch<Operation*, Order>(op)
        .Case([](ascendc::DuplicateScalarRegOp) { return Order::DuplicateOp; })
        .Case([](arith::ConstantOp) { return Order::ConstantOp; })
        .Case([](ascendc::CreateMaskOp) { return Order::CreateMaskOp; })
        .Case<
            arith::AddIOp, arith::SubIOp, arith::DivSIOp, arith::CeilDivSIOp, arith::MulIOp, arith::RemSIOp,
            arith::CmpIOp, emitasc::VariableOp, ascendc::GetVecLenOp, arith::IndexCastOp>(
            [](auto) { return Order::Expr; })
        .Case([](ascendc::RegTensorOp) { return Order::RegTensorOp; })
        .Default([](auto) { return Order::Any; });
}

void hoistOperations(ascvf::VecScopeOp vecScopeOp)
{
    vecScopeOp.walk([&](Block* block) {
        SmallVector<std::pair<Order, Operation*>> pairs;
        for (auto& op : *block) {
            auto order = getOrder(&op);
            pairs.emplace_back(order, &op);
        }
        llvm::stable_sort(pairs, [](const std::pair<Order, Operation*>& f, const std::pair<Order, Operation*>& s) {
            return f.first < s.first;
        });
        auto builder = OpBuilder::atBlockBegin(block);
        for (auto& pair : llvm::make_range(pairs.rbegin(), pairs.rend())) {
            pair.second->moveBefore(block, block->getOperations().begin());
        }
    });
}

struct ReorderOpsInVecScopePass : public ascvf::impl::ReorderOpsInVecScopeBase<ReorderOpsInVecScopePass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk([](ascvf::VecScopeOp vecScope) { hoistOperations(vecScope); });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createReorderOpsInVecScopePass()
{
    return std::make_unique<ReorderOpsInVecScopePass>();
}
