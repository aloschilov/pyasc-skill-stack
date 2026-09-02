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
#include "ascir/Dialect/AscVF/Transforms/Passes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscVF/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_MATERIALIZELOADSTORE
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

// Find local tensors that are used as dst or src
ValueVector getUsedLocalTensors(ascvf::VecScopeOp vecScope)
{
    ValueVector usedLocalTensors;
    vecScope.walk([&](Operation* op) {
        if (auto load = dyn_cast<ascvf::LoadOp>(op)) {
            usedLocalTensors.emplace_back(load.getSrcTensor());
        } else if (auto store = dyn_cast<ascvf::StoreOp>(op)) {
            usedLocalTensors.emplace_back(store.getDstTensor());
        }
    });
    return ascvf::deduplicate(usedLocalTensors);
}

ValueMap<Value> setAddress(ascvf::VecScopeOp vecScopeOp, ArrayRef<Value> usedTensors, Type groupType)
{
    ValueMap<Value> addrTensors;
    OpBuilder builder(vecScopeOp);
    for (auto value : usedTensors) {
        auto shape = cast<ascendc::LocalTensorType>(value.getType()).getShape();
        auto type = MemRefType::get(shape, groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
        auto getPhyAddrOp = builder.create<ascendc::LocalTensorGetPhyAddrV2Op>(builder.getUnknownLoc(), type, value);
        addrTensors[value] = getPhyAddrOp.getResult();
    }
    return addrTensors;
}

void materialize(ascvf::VecScopeOp vecScopeOp, Type groupType)
{
    ValueMap<Value> addrTensors;
    // materialize getPhyAddr
    SmallVector<Value> usedTensors = getUsedLocalTensors(vecScopeOp);
    addrTensors = setAddress(vecScopeOp, usedTensors, groupType);

    // materialize DataCopy from LoadMicro, StoreMicro
    auto builder = OpBuilder::atBlockBegin(vecScopeOp.getBody());
    vecScopeOp->walk([&](Operation* op) {
        OpBuilder builder(op);
        ascir::ConstantOpBuilder consts(builder);
        if (auto load = dyn_cast<ascvf::LoadOp>(op)) {
            Value tensor = load.getSrcTensor();
            auto shape = cast<ascendc::LocalTensorType>(tensor.getType()).getShape();
            auto resultType = MemRefType::get(shape, groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
            auto srcAddr = builder.create<emitasc::PtrOffsetOp>(
                builder.getUnknownLoc(), resultType, addrTensors[tensor], IntegerAttr{}, load.getOffset());
            builder.create<ascendc::DataCopyLoadOp>(builder.getUnknownLoc(), load.getDstReg(), srcAddr);
            load.erase();
        } else if (auto store = dyn_cast<ascvf::StoreOp>(op)) {
            Value tensor = store.getDstTensor();
            auto shape = cast<ascendc::LocalTensorType>(tensor.getType()).getShape();
            auto resultType = MemRefType::get(shape, groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
            auto dstAddr = builder.create<emitasc::PtrOffsetOp>(
                builder.getUnknownLoc(), resultType, addrTensors[tensor], IntegerAttr{}, store.getOffset());
            builder.create<ascendc::DataCopyStoreOp>(
                builder.getUnknownLoc(), dstAddr, store.getSrcReg(), store.getMask());
            store.erase();
        }
    });
}

struct MaterializeLoadStorePass : public ascvf::impl::MaterializeLoadStoreBase<MaterializeLoadStorePass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk([](ascvf::VFGroupOp fusedOp) {
            fusedOp.walk([&](ascvf::VecScopeOp vecScope) { materialize(vecScope, fusedOp.getGroupType()); });
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createMaterializeLoadStorePass()
{
    return std::make_unique<MaterializeLoadStorePass>();
}
