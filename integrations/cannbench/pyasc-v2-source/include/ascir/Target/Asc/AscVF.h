/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_ASCVF_H
#define ASCIR_TARGET_ASC_ASCVF_H

#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascvf {

LogicalResult printOperation(CodeEmitter& emitter, ascvf::VFForOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascvf::VecScopeOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascvf::VFGroupOp op);

} // namespace ascvf
} // namespace mlir

#endif // ASCIR_TARGET_ASC_ASCVF_H
