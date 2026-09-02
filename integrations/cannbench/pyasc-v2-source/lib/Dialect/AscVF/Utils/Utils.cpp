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
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/AscVF/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"
namespace mlir {
namespace ascvf {
ValueVector deduplicate(ArrayRef<Value> values)
{
    ValueVector result;
    ValueSet unique(values.begin(), values.end());
    for (auto value : values) {
        auto it = unique.find(value);
        if (it == unique.end())
            continue;
        result.push_back(value);
        unique.erase(it);
    }
    return result;
}

ValueVector getDst(Operation* op)
{
    return llvm::TypeSwitch<Operation*, ValueVector>(op)
        .Case<
            ascvf::LoadOp, ascendc::BinaryRegOp, ascendc::UnaryRegOp, ascendc::VecScalarRegOp, ascendc::ReduceMaxRegOp,
            ascendc::ReduceMinRegOp, ascendc::ReduceSumRegOp, ascendc::DuplicateRegOp>(
            [](auto op) { return ValueVector{op.getDstReg()}; })
        .Default([](Operation* op) { return ValueVector{op->getResults()}; });
}

bool belong(Block* block, Block* parentBlock, DominanceInfo& di)
{
    assert(block && parentBlock);
    auto* commonBlock = di.findNearestCommonDominator(block, parentBlock);
    return commonBlock == parentBlock;
}

} // namespace ascvf
} // namespace mlir
