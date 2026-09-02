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
#include "ascir/Dialect/Asc/Utils/Attributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_FUSEBUFIDSYNC
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

void eraseGetBuf(Operation* op)
{
    auto* prevNode = op->getPrevNode();
    if (!prevNode)
        return;
    if (auto getBuf = dyn_cast<ascendc::GetBufOp>(prevNode)) {
        prevNode->erase();
    } else {
        eraseGetBuf(prevNode);
    }
}

void eraseRlsBuf(Operation* op)
{
    auto* nextNode = op->getNextNode();
    if (!nextNode)
        return;
    if (auto rlsBuf = dyn_cast<ascendc::RlsBufOp>(nextNode)) {
        nextNode->erase();
    } else {
        eraseRlsBuf(nextNode);
    }
}

void eraseSync(SmallVectorImpl<Operation*>& fuseGroup)
{
    if (fuseGroup.size() <= 1)
        return;
    for (size_t i = 0; i < fuseGroup.size(); i++) {
        auto bufIdVec = dyn_cast<ArrayAttr>(fuseGroup[i]->getAttr(ascendc::attr::bufIds));
        if (!bufIdVec)
            break;
        auto numElements = bufIdVec.size();
        if (i != 0) {
            for (auto j = 0; j < numElements; j++) {
                eraseGetBuf(fuseGroup[i]);
            }
        }
        if (i != fuseGroup.size() - 1) {
            for (auto j = 0; j < numElements; j++) {
                eraseRlsBuf(fuseGroup[i]);
            }
        }
    }
}

void processOp(Operation* op, SmallVectorImpl<Operation*>& fuseGroup, ascendc::Pipe& state, ascendc::Pipe currentPipe)
{
    if (state == currentPipe) {
        if (fuseGroup.empty())
            return;
        auto bufIdAttr = op->getAttr(ascendc::attr::bufIds);
        auto fuseGroupAttr = fuseGroup.back()->getAttr(ascendc::attr::bufIds);
        if (!bufIdAttr || !fuseGroupAttr)
            return;
        if (bufIdAttr == fuseGroupAttr) {
            fuseGroup.push_back(op);
        } else {
            eraseSync(fuseGroup);
            fuseGroup.clear();
            fuseGroup.push_back(op);
        }
    } else {
        state = currentPipe;
        eraseSync(fuseGroup);
        fuseGroup.clear();
        fuseGroup.push_back(op);
    }
}

void fuseBufIdSync(func::FuncOp funcOp)
{
    SmallVector<Operation*> fuseGroup;
    ascendc::Pipe state = ascendc::Pipe::PIPE_ALL;
    funcOp.walk<WalkOrder::PreOrder>([&state, &fuseGroup](Operation* op) {
        if (auto vecOp = dyn_cast<ascendc::VectorOp>(op)) {
            processOp(vecOp, fuseGroup, state, ascendc::Pipe::PIPE_V);
        } else if (auto copyOp = dyn_cast<ascendc::DataCopyOp>(op)) {
            ascendc::Pipe pipe = ascendc::Pipe::PIPE_V;
            if (copyOp.isGlobalToLocal()) {
                pipe = ascendc::Pipe::PIPE_MTE2;
            } else if (copyOp.isLocalToGlobal()) {
                pipe = ascendc::Pipe::PIPE_MTE3;
            }
            processOp(copyOp, fuseGroup, state, pipe);
        } else if (isa<LoopLikeOpInterface, scf::IfOp, scf::YieldOp, func::ReturnOp>(op)) {
            state = ascendc::Pipe::PIPE_ALL;
            eraseSync(fuseGroup);
            fuseGroup.clear();
        }
    });
}

void removeBufIdAttr(func::FuncOp funcOp)
{
    funcOp.walk([](Operation* op) {
        op->removeAttr(ascendc::attr::bufIds);
        op->removeAttr(ascendc::attr::bufId);
    });
}

class FuseBufIdSyncPass : public ascendc::impl::FuseBufIdSyncBase<FuseBufIdSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        fuseBufIdSync(funcOp);
        removeBufIdAttr(funcOp);
    }
};
} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createFuseBufIdSyncPass() { return std::make_unique<FuseBufIdSyncPass>(); }
} // namespace ascendc
} // namespace mlir
