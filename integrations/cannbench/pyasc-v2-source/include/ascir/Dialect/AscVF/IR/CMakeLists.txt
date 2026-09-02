# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(AscVFDialect.h.inc -gen-dialect-decls -dialect=ascvf)
mlir_tablegen(AscVFDialect.cpp.inc -gen-dialect-defs -dialect=ascvf)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(AscVFOps.h.inc -gen-op-decls)
mlir_tablegen(AscVFOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(AscVFTypes.h.inc -gen-typedef-decls -dialect=ascvf -typedefs-dialect=ascvf)
mlir_tablegen(AscVFTypes.cpp.inc -gen-typedef-defs -dialect=ascvf -typedefs-dialect=ascvf)

set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(AscVFAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=ascvf)
mlir_tablegen(AscVFAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ascvf)
mlir_tablegen(AscVFEnums.h.inc -gen-enum-decls)
mlir_tablegen(AscVFEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(MLIRAscVFIncGen)

add_mlir_doc(Doc AscVF Dialects/ -gen-dialect-doc -dialect ascvf)
