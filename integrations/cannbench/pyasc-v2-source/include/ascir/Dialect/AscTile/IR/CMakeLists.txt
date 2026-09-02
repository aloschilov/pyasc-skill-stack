# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(AscTileDialect.h.inc -gen-dialect-decls -dialect=asctile)
mlir_tablegen(AscTileDialect.cpp.inc -gen-dialect-defs -dialect=asctile)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(AscTileOps.h.inc -gen-op-decls)
mlir_tablegen(AscTileOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(AscTileTypes.h.inc -gen-typedef-decls -dialect=asctile -typedefs-dialect=asctile)
mlir_tablegen(AscTileTypes.cpp.inc -gen-typedef-defs -dialect=asctile -typedefs-dialect=asctile)

set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(AscTileAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=asctile)
mlir_tablegen(AscTileAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=asctile)
mlir_tablegen(AscTileEnums.h.inc -gen-enum-decls)
mlir_tablegen(AscTileEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(MLIRAscTileIncGen)

add_mlir_doc(Doc AscTile Dialects/ -gen-dialect-doc -dialect asctile)

set(LLVM_TARGET_DEFINITIONS Ops.td)
tablegen(AscIR AscTileOpBindings.h.inc -gen-pybind-defs)
add_public_tablegen_target(AscTilePybindGen)
