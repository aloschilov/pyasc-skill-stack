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
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTBUFIDSYNC
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

class InsertBufIdSync {
    using VisitedBufIdMap = DenseMap<Operation*, std::set<int32_t>>;

    VisitedBufIdMap visitedOps;
    VisitedBufIdMap bufIdMap;
    int32_t bufId = 0;

    void updateBufId()
    {
        constexpr int32_t bufIdMax = 32;
        bufId = (bufId + 1) % bufIdMax;
    }

    void insertGetRlsBuf(Operation* op, int32_t bufId)
    {
        OpBuilder builder(op);
        auto pipe = ascendc::getOpPipe(op);
        builder.create<ascendc::GetBufOp>(op->getLoc(), pipe, bufId, false);
        builder.setInsertionPointAfter(op);
        builder.create<ascendc::RlsBufOp>(op->getLoc(), pipe, bufId, false);
    }

    void collectBufIds(Value value, std::set<int32_t>& bufIds)
    {
        if (auto blockArg = dyn_cast<BlockArgument>(value)) {
            auto argNumber = blockArg.getArgNumber();
            auto* block = blockArg.getOwner();
            auto* parentOp = block->getParentOp();
            if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
                if (argNumber == 0)
                    return;
                auto* init = forOp.getTiedLoopInit(blockArg);
                collectBufIds(init->get(), bufIds);
                return;
            }
            if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
                auto* region = block->getParent();
                if (region == &whileOp.getBefore()) {
                    collectBufIds(whileOp.getOperands()[argNumber], bufIds);
                } else if (region == &whileOp.getAfter()) {
                    auto cond = whileOp.getConditionOp();
                    auto condArg = cond.getArgs()[argNumber];
                    collectBufIds(condArg, bufIds);
                }
                return;
            }
            return;
        }
        auto* defOp = value.getDefiningOp();
        if (isa<ascendc::LocalTensorV3Op, ascendc::TBufGetTensorOp>(defOp)) {
            bufIds.insert(visitedOps[defOp].begin(), visitedOps[defOp].end());
            if (bufIds.empty()) {
                visitedOps[defOp].insert(bufId);
                Builder builder(defOp->getContext());
                defOp->setAttr(ascendc::attr::bufId, builder.getI32IntegerAttr(bufId));
                updateBufId();
                bufIds = visitedOps[defOp];
            }
            return;
        }
        if (auto rCastOp = dyn_cast<ascendc::LocalTensorReinterpretCastOp>(defOp)) {
            collectBufIds(rCastOp.getIn(), bufIds);
            return;
        }
        if (auto subTensorOp = dyn_cast<ascendc::LocalTensorSubIndexOp>(defOp)) {
            collectBufIds(subTensorOp.getTensor(), bufIds);
            return;
        }
        if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
            collectBufIds(selectOp.getTrueValue(), bufIds);
            collectBufIds(selectOp.getFalseValue(), bufIds);
            return;
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
            auto result = dyn_cast<OpResult>(value);
            if (!result)
                return;
            auto resultNumber = result.getResultNumber();
            auto thenYield = ifOp.thenYield();
            auto thenOperand = thenYield.getOperands()[resultNumber];
            collectBufIds(thenOperand, bufIds);
            auto* elseBlock = ifOp.elseBlock();
            if (elseBlock == nullptr)
                return;
            auto elseYield = ifOp.elseYield();
            auto elseOperand = elseYield.getOperands()[resultNumber];
            collectBufIds(elseOperand, bufIds);
            return;
        }
        if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
            auto result = dyn_cast<OpResult>(value);
            if (!result)
                return;
            auto iterArg = forOp.getTiedLoopRegionIterArg(result);
            auto* loopInit = forOp.getTiedLoopInit(iterArg);
            auto* loopYield = forOp.getTiedLoopYieldedValue(iterArg);
            collectBufIds(loopInit->get(), bufIds);
            collectBufIds(loopYield->get(), bufIds);
            return;
        }
        if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
            auto result = dyn_cast<OpResult>(value);
            if (!result)
                return;
            auto resultNumber = result.getResultNumber();
            auto cond = whileOp.getConditionOp();
            auto condArg = cond.getArgs()[resultNumber];
            collectBufIds(condArg, bufIds);
            auto yieldOp = whileOp.getYieldOp();
            // TODO: Add support for case when yieldOp.getNumOperands() != whileOp.getNumResults()
            if (yieldOp.getNumOperands() != whileOp.getNumResults())
                return;
            collectBufIds(yieldOp.getOperands()[resultNumber], bufIds);
        }
    }

    std::set<int32_t> getBufIds(Value value)
    {
        std::set<int32_t> bufIds;
        collectBufIds(value, bufIds);
        return bufIds;
    }

    void insertSync(Operation* op, Value tensor)
    {
        auto bufIds = getBufIds(tensor);
        for (const auto& bufId : bufIds) {
            if (!bufIdMap[op].count(bufId)) {
                bufIdMap[op].insert(bufId);
                insertGetRlsBuf(op, bufId);
            }
        }
    }

public:
    InsertBufIdSync() = default;
    ~InsertBufIdSync() = default;

    void process(Operation* op)
    {
        auto copyOp = dyn_cast<ascendc::DataCopyOp>(op);
        if (copyOp && copyOp.isLocalToGlobal()) {
            insertSync(op, copyOp.getSrc());
            return;
        }
        if (auto opWithDst = dyn_cast<ascendc::OpWithDst>(op)) {
            for (auto dst : opWithDst.getDstTensors()) {
                insertSync(op, dst);
            }
        }
        if (auto opWithSrc = dyn_cast<ascendc::OpWithSrc>(op)) {
            for (auto src : opWithSrc.getSrcTensors()) {
                insertSync(op, src);
            }
        }
        if (auto fusedOp = dyn_cast<ascvf::VFGroupOp>(op)) {
            for (auto dstTensor : fusedOp.getDstList()) {
                insertSync(op, dstTensor);
            }
            for (auto srcTensor : fusedOp.getSrcList()) {
                insertSync(op, srcTensor);
            }
        }
    }

    void setBufIdAttr(MLIRContext* context)
    {
        Builder builder(context);
        for (const auto& [op, bufIdSet] : bufIdMap) {
            SmallVector<int32_t> bufIdVec(bufIdSet.begin(), bufIdSet.end());
            op->setAttr(ascendc::attr::bufIds, builder.getI32ArrayAttr(bufIdVec));
        }
    }
};

class InsertBufIdSyncPass : public ascendc::impl::InsertBufIdSyncBase<InsertBufIdSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        MLIRContext* context = &getContext();
        InsertBufIdSync insertBufIdSync;
        funcOp.walk([&insertBufIdSync](Operation* op) { insertBufIdSync.process(op); });
        insertBufIdSync.setBufIdAttr(context);
        funcOp.walk([](scf::ForOp forOp) {
            if (forOp->hasAttrOfType<UnitAttr>(asctile::attr::gmBarrier))
                forOp->removeAttr(asctile::attr::gmBarrier);
            else
                return;
            auto* terminator = forOp.getBody()->getTerminator();
            OpBuilder builder(terminator);
            ascir::ConstantOpBuilder consts(builder);
            auto const0 = consts.i32(0);
            builder.create<ascendc::SetFlagOp>(terminator->getLoc(), ascendc::HardEvent::MTE3_MTE2, const0);
            builder.create<ascendc::SetFlagOp>(terminator->getLoc(), ascendc::HardEvent::MTE3_MTE1, const0);
            builder.create<ascendc::WaitFlagOp>(terminator->getLoc(), ascendc::HardEvent::MTE3_MTE2, const0);
            builder.create<ascendc::WaitFlagOp>(terminator->getLoc(), ascendc::HardEvent::MTE3_MTE1, const0);
        });
    }
};
} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createInsertBufIdSyncPass() { return std::make_unique<InsertBufIdSyncPass>(); }
} // namespace ascendc
} // namespace mlir
