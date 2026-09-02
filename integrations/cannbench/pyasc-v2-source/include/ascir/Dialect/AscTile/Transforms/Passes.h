/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASCTILE_TRANSFORMS_PASSES_H
#define ASCIR_DIALECT_ASCTILE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace asctile {

#define GEN_PASS_DECL
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createCubeTransposeToLoadPass();
std::unique_ptr<Pass> createDetectBiasLoadPass();
std::unique_ptr<Pass> createFoldCastPass();
std::unique_ptr<Pass> createLegalizeMatmulPass();
std::unique_ptr<Pass> createLocationCastToCopyPass();
std::unique_ptr<Pass> createMarkMatmulAccWithBiasPass();
std::unique_ptr<Pass> createMarkReuseSourcePass();
std::unique_ptr<Pass> createMergeCVGroupsPass();
std::unique_ptr<Pass> createPromotePureOpsPass();
std::unique_ptr<Pass> createResolveAutoLocationPass();
std::unique_ptr<Pass> createSplitCubeLoadPass();
std::unique_ptr<Pass> createTransformMathOpsPass();
std::unique_ptr<Pass> createTransformStoreFixpipePass();
std::unique_ptr<Pass> createUnrollLoopPass(bool annotate = false);
std::unique_ptr<Pass> createUnscalarizeReductionPass();
std::unique_ptr<Pass> createVectorTransposeToLoadStorePass();
std::unique_ptr<Pass> createVerifyTensorLocationPass();
std::unique_ptr<Pass> createWrapCVGroupsPass();

} // namespace asctile

#define GEN_PASS_REGISTRATION
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // ASCIR_DIALECT_ASCTILE_TRANSFORMS_PASSES_H
