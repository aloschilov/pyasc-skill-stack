/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint> // NOLINT(llvm-include-order)

#include "kernel_tiling/kernel_tiling.h"
#include "tiling/tiling_api.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pybind11 {
namespace asc {
void pyasc_init_enums(py::module& m)
{
    using namespace matmul_tiling;

    py::enum_<TPosition>(m, "TPosition", py::module_local())
        .value("GM", TPosition::GM)
        .value("A1", TPosition::A1)
        .value("A2", TPosition::A2)
        .value("B1", TPosition::B1)
        .value("B2", TPosition::B2)
        .value("C1", TPosition::C1)
        .value("C2", TPosition::C2)
        .value("CO1", TPosition::CO1)
        .value("CO2", TPosition::CO2)
        .value("VECIN", TPosition::VECIN)
        .value("VECOUT", TPosition::VECOUT)
        .value("VECCALC", TPosition::VECCALC)
        .value("LCM", TPosition::LCM)
        .value("SPM", TPosition::SPM)
        .value("SHM", TPosition::SHM)
        .value("TSCM", TPosition::TSCM)
        .value("MAX", TPosition::MAX);

    py::enum_<CubeFormat>(m, "CubeFormat", py::module_local())
        .value("ND", CubeFormat::ND)
        .value("NZ", CubeFormat::NZ)
        .value("ZN", CubeFormat::ZN)
        .value("ZZ", CubeFormat::ZZ)
        .value("NN", CubeFormat::NN)
        .value("ND_ALIGN", CubeFormat::ND_ALIGN)
        .value("SCALAR", CubeFormat::SCALAR)
        .value("VECTOR", CubeFormat::VECTOR)
        .value("ROW_MAJOR", CubeFormat::ROW_MAJOR)
        .value("COLUMN_MAJOR", CubeFormat::COLUMN_MAJOR);

    py::enum_<DataType>(m, "DataType", py::module_local())
        .value("DT_FLOAT", DataType::DT_FLOAT)
        .value("DT_FLOAT16", DataType::DT_FLOAT16)
        .value("DT_INT8", DataType::DT_INT8)
        .value("DT_INT16", DataType::DT_INT16)
        .value("DT_UINT8", DataType::DT_UINT8)
        .value("DT_UINT16", DataType::DT_UINT16)
        .value("DT_INT64", DataType::DT_INT64)
        .value("DT_INT32", DataType::DT_INT32)
        .value("DT_UINT64", DataType::DT_UINT64)
        .value("DT_UINT32", DataType::DT_UINT32)
        .value("DT_BOOL", DataType::DT_BOOL)
        .value("DT_STRING", DataType::DT_STRING)
        .value("DT_DOUBLE", DataType::DT_DOUBLE)
        .value("DT_COMPLEX64", DataType::DT_COMPLEX64)
        .value("DT_COMPLEX128", DataType::DT_COMPLEX128)
        .value("DT_QINT16", DataType::DT_QINT16)
        .value("DT_QINT32", DataType::DT_QINT32)
        .value("DT_QINT8", DataType::DT_QINT8)
        .value("DT_QUINT8", DataType::DT_QUINT8)
        .value("DT_QUINT16", DataType::DT_QUINT16)
        .value("DT_STRING_REF", DataType::DT_STRING_REF)
        .value("DT_DUAL", DataType::DT_DUAL)
        .value("DT_BF16", DataType::DT_BF16)
        .value("DT_VARIANT", DataType::DT_VARIANT)
        .value("DT_INT4", DataType::DT_INT4)
        .value("DT_UNDEFINED", DataType::DT_UNDEFINED)
        .value("DT_UINT1", DataType::DT_UINT1)
        .value("DT_INT2", DataType::DT_INT2)
        .value("DT_BFLOAT16", DataType::DT_BFLOAT16)
        .value("DT_MAX", DataType::DT_MAX);

    py::enum_<MatrixTraverse>(m, "MatrixTraverse", py::module_local())
        .value("NOSET", MatrixTraverse::NOSET)
        .value("FIRSTM", MatrixTraverse::FIRSTM)
        .value("FIRSTN", MatrixTraverse::FIRSTN);

    py::enum_<MatrixMadType>(m, "MatrixMadType", py::module_local())
        .value("NORMAL", MatrixMadType::NORMAL)
        .value("HF32", MatrixMadType::HF32)
        .value("MXMODE", MatrixMadType::MXMODE);

    py::enum_<DequantType>(m, "DequantType", py::module_local())
        .value("SCALAR", DequantType::SCALAR)
        .value("TENSOR", DequantType::TENSOR);

    py::enum_<ScheduleType>(m, "ScheduleType", py::module_local())
        .value("INNER_PRODUCT", ScheduleType::INNER_PRODUCT)
        .value("OUTER_PRODUCT", ScheduleType::OUTER_PRODUCT);
}
} // namespace asc
} // namespace pybind11
