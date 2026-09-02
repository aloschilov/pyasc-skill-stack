# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_mlir_conversion_library(MLIRLowerToAsc
  DisplaceConcat.cpp
  ExpandMask.cpp
  ExpandMath.cpp
  LowerArith.cpp
  LowerArithBinary.cpp
  LowerAscTile.cpp
  LowerAscTileDataTransfer.cpp
  LowerAscTileToBasic.cpp
  LowerAtomic.cpp
  LowerMath.cpp
  LowerSCF.cpp
  LowerTensor.cpp
  RealizeConversionCast.cpp
  RedressI1Tensor.cpp

ADDITIONAL_HEADER_DIRS
  ${ASCIR_INCLUDE_DIR}/ascir/Conversion/LowerToAsc

DEPENDS
  MLIRLowerToAscPassIncGen

LINK_COMPONENTS
  Core

LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAsc
  MLIRAscVF
  MLIRAscTile
  MLIRMathDialect
  MLIRMathTransforms
  MLIRSCFDialect
)
