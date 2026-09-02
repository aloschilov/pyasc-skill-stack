/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/Transforms/Passes.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_UNROLLLOOP
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

class Annotator {
    bool enable;
    int64_t loopId = 0;

    void setI64Attr(StringRef name, int64_t value, Operation* op, Builder builder) const
    {
        if (enable)
            op->setAttr(name, builder.getI64IntegerAttr(value));
    }

public:
    explicit Annotator(bool enable) : enable(enable) {}
    ~Annotator() = default;

    void applyLoopId(Operation* op) { setI64Attr(attr::unrolledLoop, loopId++, op, Builder(op)); }

    void operator()(unsigned iter, Operation* op, Builder builder) const
    {
        setI64Attr(attr::unrollIter, static_cast<int64_t>(iter), op, builder);
    }
};

int64_t getUnrollFactor(scf::ForOp loop)
{
    if (auto a = loop->getAttrOfType<IntegerAttr>(attr::unrollFactor))
        return std::max(1L, a.getValue().getSExtValue());
    return 1L;
}

void wrapLoopBeforeUnroll(scf::ForOp loop)
{
    int64_t unrollFactor = getUnrollFactor(loop);
    if (unrollFactor <= 1)
        return;
    OpBuilder builder(loop);
    auto exec = builder.create<scf::ExecuteRegionOp>(loop.getLoc(), loop.getResultTypes());
    exec->setAttr(attr::unrollFactor, builder.getI64IntegerAttr(unrollFactor));
    auto* body = &exec.getRegion().emplaceBlock();
    loop->moveBefore(body, body->end());
    loop->replaceAllUsesWith(exec.getResults());
    builder.setInsertionPointToEnd(body);
    builder.create<scf::YieldOp>(loop.getLoc(), loop.getResults());
}

struct UnrollLoopPass : public asctile::impl::UnrollLoopBase<UnrollLoopPass> {
    UnrollLoopPass(const UnrollLoopOptions& options) : UnrollLoopBase(options) {}

    void runOnOperation() override
    {
        Annotator annotator(annotate);
        auto op = getOperation();
        op.walk(wrapLoopBeforeUnroll);
        op.walk([&](scf::ForOp loop) {
            int64_t unrollFactor = getUnrollFactor(loop);
            loop->removeAttr(attr::unrollFactor);
            if (unrollFactor <= 1)
                return;
            Builder builder(loop);
            for (auto& op : loop.getBody()->without_terminator())
                annotator(unrollFactor, &op, builder);
            auto result = loopUnrollByFactor(loop, unrollFactor, annotator);
            if (failed(result))
                signalPassFailure();
        });
        op.walk([&](scf::ExecuteRegionOp exec) { annotator.applyLoopId(exec); });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createUnrollLoopPass(bool annotate)
{
    UnrollLoopOptions options;
    options.annotate = annotate;
    return std::make_unique<UnrollLoopPass>(options);
}
