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
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERTENSOR
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

struct ConvertSplat : public ConvertOp<tensor::SplatOp> {
    using ConvertOp::ConvertOp;

    LogicalResult matchAndRewrite(tensor::SplatOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        rewriter.create<ascendc::DuplicateL2Op>(loc, dst, op.getInput(), consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct LowerTensorPass : public asclower::impl::LowerTensorBase<LowerTensorPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addIllegalOp<tensor::SplatOp>();
        target.addLegalDialect<ascendc::AscendCDialect, arith::ArithDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        RewritePatternSet patterns(context);
        patterns.insert<ConvertSplat>(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerTensorPass() { return std::make_unique<LowerTensorPass>(); }
