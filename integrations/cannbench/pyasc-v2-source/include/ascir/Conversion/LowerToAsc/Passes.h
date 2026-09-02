/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_CONVERSION_LOWERTOASC_PASSES_H
#define ASCIR_CONVERSION_LOWERTOASC_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace asclower {

#define GEN_PASS_DECL
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"

std::unique_ptr<Pass> createDisplaceConcatPass();
std::unique_ptr<Pass> createExpandMaskPass();
std::unique_ptr<Pass> createExpandMathPass();
std::unique_ptr<Pass> createLowerArithPass();
std::unique_ptr<Pass> createLowerArithBinaryPass();
std::unique_ptr<Pass> createLowerAscTilePass();
std::unique_ptr<Pass> createLowerAscTileDataTransferPass();
std::unique_ptr<Pass> createLowerAscTileToBasicPass();
std::unique_ptr<Pass> createLowerAtomicPass();
std::unique_ptr<Pass> createLowerMathPass();
std::unique_ptr<Pass> createLowerSCFPass();
std::unique_ptr<Pass> createLowerTensorPass();
std::unique_ptr<Pass> createRealizeConversionCastPass();
std::unique_ptr<Pass> createRedressI1TensorPass();

} // namespace asclower

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"

} // namespace mlir

#endif // ASCIR_CONVERSION_LOWERTOASC_PASSES_H
