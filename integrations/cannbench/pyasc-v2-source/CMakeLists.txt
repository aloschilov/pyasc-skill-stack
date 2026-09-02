# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.20.0)

if(POLICY CMP0116)
# Introduced in cmake 3.20
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
  cmake_policy(SET CMP0116 OLD)
endif()

# Debian's packaged LLVM/MLIR config probes libffi and terminfo with C
# snippets, so C must be enabled even though AscIR itself is C++.
project(AscIR LANGUAGES C CXX)

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Linker: ${CMAKE_LINKER}")
message(STATUS "Selected build configuration: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Ninja compile timing report (for build benchmarking)
if(CMAKE_GENERATOR MATCHES "Ninja")
  if(DEFINED ENV{PYASC_BUILD_TIMING} AND "$ENV{PYASC_BUILD_TIMING}" STREQUAL "1")
    message(STATUS "Build timing report enabled (PYASC_BUILD_TIMING=1)")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-report")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-trace")
    else()
      message(WARNING "Build timing not supported for ${CMAKE_CXX_COMPILER_ID}")
    endif()
  endif()
endif()

if(DEFINED LLVM_PREFIX_PATH)
  list(APPEND CMAKE_PREFIX_PATH "${LLVM_PREFIX_PATH}")
endif()

if(NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

option(ASCIR_CCACHE "Use ccache as a compiler launcher" OFF)
if (ASCIR_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "ccache is enabled")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
  else()
    message(STATUS "ccache is not found")
  endif()
endif()

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(ASCIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ASCIR_BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
set(ASCIR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(AscIR_TABLEGEN_EXE ${ASCIR_BINARY_DIR}/TableGen)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

option(ASCIR_WERROR "Treat all warnings as errors" ON)
if (ASCIR_WERROR)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

option(ASCIR_COVERAGE "Enable code coverage collection" OFF)
if (ASCIR_COVERAGE)
  message(STATUS "Code coverage collection is enabled")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options("-g" "-O0" "-fprofile-instr-generate" "-fcoverage-mapping")
    add_link_options("-fprofile-instr-generate")
    if (NOT CMAKE_LINKER MATCHES "lld")
      add_link_options("-Wl,-z,nostart-stop-gc")
    endif()
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options("-g" "-O0" "-fprofile-arcs" "-ftest-coverage" "-fprofile-update=atomic" "--coverage")
    add_link_options("-lgcov" "--coverage")
  endif()
endif()

option(ASCIR_ASAN "Compile with AddressSanitizer" OFF)
if (ASCIR_ASAN)
  message(STATUS "AddressSanitizer is enabled")
  add_compile_options("-fsanitize=address" "-fno-omit-frame-pointer" "-g")
  add_link_options("-fsanitize=address")
endif()

# Declare TableGen toolchain early
add_subdirectory(lib/TableGen)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)

add_subdirectory(python/src)
if(ASCIR_LLT_TEST)
  add_subdirectory(test)
endif()
