/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASCVF_UTILS_UTILS_H
#define ASCIR_DIALECT_ASCVF_UTILS_UTILS_H

#include "ascir/Dialect/Utils/Utils.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace ascvf {

ValueVector deduplicate(ArrayRef<Value> values);

ValueVector getDst(Operation* op);

bool belong(Block* block, Block* parentBlock, DominanceInfo& di);

} // namespace ascvf
} // namespace mlir

#endif // ASCIR_DIALECT_ASCVF_UTILS_UTILS_H
