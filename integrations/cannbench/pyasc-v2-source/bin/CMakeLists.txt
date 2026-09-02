# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# ascir-lsp

set(ASCIR_LSP_NAME ascir-lsp)
add_mlir_tool(${ASCIR_LSP_NAME} ascir-lsp.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(${ASCIR_LSP_NAME})
target_link_libraries(${ASCIR_LSP_NAME} PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRIR
  MLIRLspServerLib
  MLIRAsc
  MLIRAscTile
  MLIRAscVF
  MLIREmitAsc
)

mlir_check_all_link_libraries(${ASCIR_LSP_NAME})
set_target_properties(${ASCIR_LSP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ascir-opt

set(ASCIR_OPT_NAME ascir-opt)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

  MLIRAsc
  MLIRAscTransforms
  MLIRAscUtils
  MLIRAscTile
  MLIRAscTileTransforms
  MLIRAscVF
  MLIRAscVFTransforms
  MLIREmitAsc
  MLIRLowerToAsc
)

add_mlir_tool(${ASCIR_OPT_NAME}
  ascir-opt.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS
  ${LIBS}
  SUPPORT_PLUGINS
)

target_link_libraries(${ASCIR_OPT_NAME} PRIVATE ${LIBS} MLIRMlirOptMain)
llvm_update_compile_flags(${ASCIR_OPT_NAME})

mlir_check_all_link_libraries(${ASCIR_OPT_NAME})

set_target_properties(${ASCIR_OPT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ascir-translate

set(ASCIR_TRANSLATE_NAME ascir-translate)

set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_mlir_tool(${ASCIR_TRANSLATE_NAME} ascir-translate.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(${ASCIR_TRANSLATE_NAME})
target_link_libraries(${ASCIR_TRANSLATE_NAME}
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport

  MLIRTargetAsc
)

mlir_check_link_libraries(${ASCIR_TRANSLATE_NAME})

set_target_properties(${ASCIR_TRANSLATE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
