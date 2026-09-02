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
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pybind11 {
namespace asc {
void pyasc_init_matmul_api_tiling(py::module& m)
{
    using namespace matmul_tiling;

    // MatmulConfigParams struct
    py::class_<MatmulConfigParams>(m, "MatmulConfigParams", py::module_local())
        .def(
            py::init<int32_t, bool, ScheduleType, MatrixTraverse, bool>(), "mm_config_type"_a = 1,
            "enable_l1_cache_ub"_a = false, "schedule_type"_a = ScheduleType::INNER_PRODUCT,
            "traverse"_a = MatrixTraverse::NOSET, "en_vec_nd2nz"_a = false)
        .def_readwrite("mm_config_type", &MatmulConfigParams::mmConfigType)
        .def_readwrite("enable_l1_cache_ub", &MatmulConfigParams::enableL1CacheUB)
        .def_readwrite("schedule_type", &MatmulConfigParams::scheduleType)
        .def_readwrite("traverse", &MatmulConfigParams::traverse)
        .def_readwrite("en_vec_nd2nz", &MatmulConfigParams::enVecND2NZ);

    // MatmulApiTilingBase class
    py::class_<MatmulApiTilingBase>(m, "MatmulApiTilingBase", py::module_local())
        // Enable methods
        .def(
            "enable_bias", [](MatmulApiTilingBase& self, bool isBiasIn) { return self.EnableBias(isBiasIn); },
            "is_bias_in"_a = false,
            R"doc(
          设置Bias是否参与运算，设置的信息必须与Kernel侧保持一致。

          **对应的Ascend C函数原型**

          .. code-block:: c++

              int32_t EnableBias(bool isBiasIn = false)

          **参数说明**
          
          - is_bias_in：设置是否有Bias参与运算。

          **返回值说明**
          
          -1表示设置失败；0表示设置成功。

          **调用示例**
          
          .. code-block:: python

              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.enable_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        // Get methods
        .def(
            "get_base_k", [](MatmulApiTilingBase& self) { return self.GetBaseK(); },
            R"doc(
          获取Tiling计算得到的baseK值。

          **对应的Ascend C函数原型**

          .. code-block:: c++

              int32_t GetBaseK()

          **返回值说明**
          
          返回值为Tiling计算得到的baseK值。

          **约束说明**

          使用创建的Tiling对象调用该接口，且需在完成Tiling计算（GetTiling）后调用。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
              bask_k = tiling.get_base_k()
          )doc")
        .def(
            "get_base_m", [](MatmulApiTilingBase& self) { return self.GetBaseM(); },
            R"doc(
          获取Tiling计算得到的baseM值。

          **对应的Ascend C函数原型**

          .. code-block:: c++

              int32_t GetBaseM()

          **返回值说明**
          
          返回值为Tiling计算得到的baseM值。

          **约束说明**

          使用创建的Tiling对象调用该接口，且需在完成Tiling计算（GetTiling）后调用。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
              bask_m = tiling.get_base_m()
          )doc")
        .def(
            "get_base_n", [](MatmulApiTilingBase& self) { return self.GetBaseN(); },
            R"doc(
          获取Tiling计算得到的baseN值。

          **对应的Ascend C函数原型**

          .. code-block:: c++

              int32_t GetBaseN()

          **返回值说明**
          
          返回值为Tiling计算得到的baseN值。

          **约束说明**

          使用创建的Tiling对象调用该接口，且需在完成Tiling计算（GetTiling）后调用。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
              bask_n = tiling.get_bask_n()
          )doc")
        .def(
            "get_tiling",
            [](MatmulApiTilingBase& self, py::object& tiling) {
                py::object method = tiling.attr("addressof");
                py::object result = method();
                auto cpp_int = py::cast<size_t>(result);
                auto* tiling_new = reinterpret_cast<TCubeTiling*>(cpp_int);
                return self.GetTiling(*tiling_new);
            },
            "tiling"_a,
            R"doc(
          获取Tiling参数。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int64_t GetTiling(optiling::TCubeTiling &tiling)
              int64_t GetTiling(TCubeTiling &tiling)

          **参数说明**
          
          - tiling：Tiling结构体存储最终的tiling结果。

          **约束说明**

          在Tiling计算失败的场景，若需查看Tiling计算失败的原因，请将日志级别设置为WARNING级别，并在日志中搜索关键字“MatmulApi Tiling”。
          
          **返回值说明**
          
          如果返回值不为-1，则代表Tiling计算成功，用户可以使用该Tiling结构的值。如果返回值为-1，则代表Tiling计算失败，该Tiling结果无法使用。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        // Set methods
        .def(
            "set_a_layout",
            [](MatmulApiTilingBase& self, int32_t b, int32_t s, int32_t n, int32_t g, int32_t d) {
                return self.SetALayout(b, s, n, g, d);
            },
            "b"_a, "s"_a, "n"_a, "g"_a, "d"_a,
            R"doc(
          设置A矩阵的Layout轴信息，包括B、S、N、G、D轴。对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用IterateBatch接口之前，
          需要在Host侧Tiling实现中通过本接口设置A矩阵的Layout轴信息。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetALayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d)

          **参数说明**

          - b：A矩阵Layout的B轴信息。
          - s：A矩阵Layout的S轴信息。
          - n：A矩阵Layout的N轴信息。
          - g：A矩阵Layout的G轴信息。
          - d：A矩阵Layout的D轴信息。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **约束说明**

          对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用iterate_batch接口之前，需要在Host侧Tiling实现中通过本接口设置A矩阵的Layout轴信息。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              m = 32
              n = 256
              k = 64
              tiling.set_dim(1)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(m, n, k)
              tiling.set_org_shape(m, n, k)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              a_bnum = 2
              a_snum = 32
              a_gnum = 3
              a_dnum = 64
              b_bnum = 2
              b_snum = 256
              b_gnum = 3
              b_dnum = 64
              c_bnum = 2
              c_snum = 32
              c_gnum = 3
              c_dnum = 256
              batch_num = 3
              tiling.set_a_layout(a_bnum, a_snum, 1, a_gnum, a_dnum) # 设置 A 矩阵排布
              tiling.set_b_layout(b_bnum, b_snum, 1, b_gnum, b_dnum)
              tiling.set_c_layout(c_bnum, c_snum, 1, c_gnum, c_dnum)
              tiling.set_batch_num(batch_num)
              tiling.set_buffer_space(-1, -1, -1);
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_a_type",
            [](MatmulApiTilingBase& self, TPosition pos, CubeFormat type, DataType dataType, bool isTrans) {
                return self.SetAType(pos, type, dataType, isTrans);
            },
            "pos"_a, "type"_a, "data_type"_a, "is_trans"_a,
            R"doc(
          设置A矩阵的位置，数据格式，数据类型，是否转置等信息，这些信息需要和kernel侧的设置保持一致。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetAType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false)

          **参数说明**
          
          - pos：A矩阵所在的buffer位置。
          - type：A矩阵的数据格式。
          - data_type：A矩阵的数据类型。
          - is_trans：A矩阵是否转置。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              # 设置A矩阵，buffer位置为GM，数据格式为ND，数据类型为bfloat16，默认不转置
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_b_layout",
            [](MatmulApiTilingBase& self, int32_t b, int32_t s, int32_t n, int32_t g, int32_t d) {
                return self.SetBLayout(b, s, n, g, d);
            },
            "b"_a, "s"_a, "n"_a, "g"_a, "d"_a,
            R"doc(
          设置B矩阵的Layout轴信息，包括B、S、N、G、D轴。对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用IterateBatch接口之前，
          需要在Host侧Tiling实现中通过本接口设置B矩阵的Layout轴信息。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetBLayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d)

          **参数说明**

          - b：B矩阵Layout的B轴信息。
          - s：B矩阵Layout的S轴信息。
          - n：B矩阵Layout的N轴信息。
          - g：B矩阵Layout的G轴信息。
          - d：B矩阵Layout的D轴信息。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **约束说明**

          对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用IterateBatch接口之前，需要在Host侧Tiling实现中通过本接口设置A矩阵的Layout轴信息。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              m = 32
              n = 256
              k = 64
              tiling.set_dim(1)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(m, n, k)
              tiling.set_org_shape(m, n, k)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              a_bnum = 2
              a_snum = 32
              a_gnum = 3
              a_dnum = 64
              b_bnum = 2
              b_snum = 256
              b_gnum = 3
              b_dnum = 64
              c_bnum = 2
              c_snum = 32
              c_gnum = 3
              c_dnum = 256
              batch_num = 3
              tiling.set_a_layout(a_bnum, a_snum, 1, a_gnum, a_dnum) 
              tiling.set_b_layout(b_bnum, b_snum, 1, b_gnum, b_dnum)    # 设置B矩阵排布
              tiling.set_c_layout(c_bnum, c_snum, 1, c_gnum, c_dnum)
              tiling.set_batch_num(batch_num)
              tiling.set_buffer_space(-1, -1, -1);
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_b_type",
            [](MatmulApiTilingBase& self, TPosition pos, CubeFormat type, DataType dataType, bool isTrans) {
                return self.SetBType(pos, type, dataType, isTrans);
            },
            "pos"_a, "type"_a, "data_type"_a, "is_trans"_a,
            R"doc(
          设置B矩阵的位置，数据格式，数据类型，是否转置等信息，这些信息需要和kernel侧的设置保持一致。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetBType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false)

          **参数说明**
          
          - pos：B矩阵所在的buffer位置。
          - type：B矩阵的数据格式。
          - data_type：B矩阵的数据类型。
          - is_trans：B矩阵是否转置。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              # 设置B矩阵，buffer位置为GM，数据格式为ND，数据类型为bfloat16，默认不转置
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_batch_info_for_normal",
            [](MatmulApiTilingBase& self, int32_t batchA, int32_t batchB, int32_t m, int32_t n, int32_t k) {
                return self.SetBatchInfoForNormal(batchA, batchB, m, n, k);
            },
            "batch_a"_a, "batch_b"_a, "m"_a, "n"_a, "k"_a,
            R"doc(
          设置A/B矩阵的M/N/K轴信息，以及A/B矩阵的Batch数。Layout类型为NORMAL的场景，
          调用IterateBatch或者IterateNBatch接口之前，需要在Host侧Tiling实现中通过本接口设置A/B矩阵的M/N/K轴等信息。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetBatchInfoForNormal(int32_t batchA, int32_t batchB, int32_t m, int32_t n, int32_t k)

          **参数说明**
          
          - batch_a：A矩阵的batch数。
          - batch_b：B矩阵的batch数。
          - m：A矩阵的M轴信息
          - n：B矩阵的N轴信息
          - k：A/B矩阵的K轴信息
          
          **返回值说明**

          -1表示设置失败； 0表示设置成功。

          **约束说明**

          Layout类型为NORMAL的场景，调用iterate_batch或者iterate_n_batch接口之前，需要在Host侧Tiling实现中通过本接口设置A/B矩阵的M/N/K轴等信息。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              m = 32
              n = 256
              k = 64
              tiling.set_dim(1)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(m, n, k)
              tiling.set_org_shape(m, n, k)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              batch_num = 3
              tiling.set_batch_info_for_normal(batch_num, batch_num, m, n, k)
              tiling.set_buffer_space(-1, -1, -1);
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_batch_num", [](MatmulApiTilingBase& self, int32_t batch) { return self.SetBatchNum(batch); },
            "batch"_a,
            R"doc(
          设置多Batch计算的最大Batch数，最大Batch数为A矩阵batchA和B矩阵batchB中的最大值。
          调用IterateBatch接口之前，需要在Host侧Tiling实现中通过本接口设置多Batch计算的Batch数。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetBatchNum(int32_t batch)

          **参数说明**
        
          - batch：多Batch计算的Batch数，Batch数为A矩阵batchA和B矩阵batchB中的最大值。
          
          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **约束说明**

          调用iterate_batch接口之前，需要在Host侧Tiling实现中通过本接口设置多Batch计算的Batch数。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              m = 32
              n = 256
              k = 64
              tiling.set_dim(1)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(m, n, k)
              tiling.set_org_shape(m, n, k)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              a_bnum = 2
              a_snum = 32
              a_gnum = 3
              a_dnum = 64
              b_bnum = 2
              b_snum = 256
              b_gnum = 3
              b_dnum = 64
              c_bnum = 2
              c_snum = 32
              c_gnum = 3
              c_dnum = 256
              batch_num = 3
              tiling.set_a_layout(a_bnum, a_snum, 1, a_gnum, a_dnum) # 设置 A 矩阵排布
              tiling.set_b_layout(b_bnum, b_snum, 1, b_gnum, b_dnum)
              tiling.set_c_layout(c_bnum, c_snum, 1, c_gnum, c_dnum)
              tiling.set_batch_num(batch_num)   # 设置Batch数
              tiling.set_buffer_space(-1, -1, -1);
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_bias_type",
            [](MatmulApiTilingBase& self, TPosition pos, CubeFormat type, DataType dataType) {
                return self.SetBiasType(pos, type, dataType);
            },
            "pos"_a, "type"_a, "data_type"_a,
            R"doc(
          设置Bias的位置，数据格式，数据类型，是否转置等信息，这些信息需要和kernel侧的设置保持一致。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetBiasType(TPosition pos, CubeFormat type, DataType dataType)

          **参数说明**
          
          - pos：Bias矩阵所在的buffer位置。
          - type：Bias矩阵的数据格式。
          - data_type：Bias矩阵的数据类型。
          
          **返回值说明**

          -1表示设置失败； 0表示设置成功。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_buffer_space",
            [](MatmulApiTilingBase& self, int32_t l1Size, int32_t l0CSize, int32_t ubSize, int32_t btSize) {
                return self.SetBufferSpace(l1Size, l0CSize, ubSize, btSize);
            },
            "l1_size"_a = -1, "l0_c_size"_a = -1, "ub_size"_a = -1, "bt_size"_a = -1,
            R"doc(
          设置Matmul计算时可用的L1 Buffer/L0C Buffer/Unified Buffer/BiasTable Buffer空间大小，单位为字节。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetBufferSpace(int32_t l1Size = -1, int32_t l0CSize = -1, int32_t ubSize = -1, int32_t btSize = -1)

          **参数说明**
          
          - l1_size：设置Matmul计算时，能够使用的L1 Buffer大小，单位为字节。默认值-1，表示使用AI处理器L1 Buffer大小。
          - l0_c_size：设置Matmul计算时，能够使用的L0C Buffer大小，单位为字节。默认值-1，表示使用AI处理器L0C Buffer大小。
          - ub_size：设置Matmul计算时，能够使用的UB Buffer大小，单位为字节。默认值-1，表示使用AI处理器UB Buffer大小。
          - bt_size：设置Matmul计算时，能够使用的BiasTable Buffer大小，单位为字节。默认值-1，表示使用AI处理器BiasTable Buffer大小。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              # 设置B矩阵，buffer位置为GM，数据格式为ND，数据类型为bfloat16，默认不转置
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1, -1)   # 设置计算时可用的L1/L0C/UB/BT空间大小
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_c_layout",
            [](MatmulApiTilingBase& self, int32_t b, int32_t s, int32_t n, int32_t g, int32_t d) {
                return self.SetCLayout(b, s, n, g, d);
            },
            "b"_a, "s"_a, "n"_a, "g"_a, "d"_a,
            R"doc(
          设置C矩阵的Layout轴信息，包括B、S、N、G、D轴。对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用IterateBatch接口之前，
          需要在Host侧Tiling实现中通过本接口设置C矩阵的Layout轴信息。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetCLayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d)

          **参数说明**

          - b：C矩阵Layout的B轴信息。
          - s：C矩阵Layout的S轴信息。
          - n：C矩阵Layout的N轴信息。
          - g：C矩阵Layout的G轴信息。
          - d：C矩阵Layout的D轴信息。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **约束说明**

          对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用iterate_batch接口之前，需要在Host侧Tiling实现中通过本接口设置C矩阵的Layout轴信息。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              m = 32
              n = 256
              k = 64
              tiling.set_dim(1)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(m, n, k)
              tiling.set_org_shape(m, n, k)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              a_bnum = 2
              a_snum = 32
              a_gnum = 3
              a_dnum = 64
              b_bnum = 2
              b_snum = 256
              b_gnum = 3
              b_dnum = 64
              c_bnum = 2
              c_snum = 32
              c_gnum = 3
              c_dnum = 256
              batch_num = 3
              tiling.set_a_layout(a_bnum, a_snum, 1, a_gnum, a_dnum) 
              tiling.set_b_layout(b_bnum, b_snum, 1, b_gnum, b_dnum)
              tiling.set_c_layout(c_bnum, c_snum, 1, c_gnum, c_dnum) # 设置C矩阵排布
              tiling.set_batch_num(batch_num)
              tiling.set_buffer_space(-1, -1, -1);
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_c_type",
            [](MatmulApiTilingBase& self, TPosition pos, CubeFormat type, DataType dataType) {
                return self.SetCType(pos, type, dataType);
            },
            "pos"_a, "type"_a, "data_type"_a,
            R"doc(
          设置C矩阵的位置，数据格式，数据类型，是否转置等信息，这些信息需要和kernel侧的设置保持一致。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetCType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false)

          **参数说明**
          
          - pos：C矩阵所在的buffer位置。
          - type：C矩阵的数据格式。
          - data_type：C矩阵的数据类型。
          - is_trans：C矩阵是否转置。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              # 设置C矩阵，buffer位置为GM，数据格式为ND，数据类型为float，默认不转置
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_dequant_type",
            [](MatmulApiTilingBase& self, DequantType dequantType) { return self.SetDequantType(dequantType); },
            "dequant_type"_a,
            R"doc(
          该接口用于设置量化或反量化的模式。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetDequantType(DequantType dequantType)

          **参数说明**
          
          - dequant_type：设置量化或反量化时的模式。
          
          **返回值说明**
          
          1表示设置失败； 0表示设置成功。

          **约束说明**

          本接口支持的同一系数的量化/反量化模式、向量的量化/反量化模式分别与Kernel侧接口set_quant_scalar和set_quant_vector对应，本接口设置的量化/反量化模式必须与Kernel侧使用的接口保持一致。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.enable_bias(True)
              tiling.set_dequant_type(host.DequantType.SCALAR)  # 设置同一系数的量化/反量化模式
              # tiling.set_dequant_type(host.DequantType.TENSOR)  # 设置向量的量化/反量化模式
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_double_buffer",
            [](MatmulApiTilingBase& self, bool a, bool b, bool c, bool bias, bool transND2NZ, bool transNZ2ND) {
                return self.SetDoubleBuffer(a, b, c, bias, transND2NZ, transNZ2ND);
            },
            "a"_a, "b"_a, "c"_a, "bias"_a, "trans_nd2nz"_a = true, "trans_nz2nd"_a = true,
            R"doc(
          设置A/B/C/Bias是否使能double buffer功能，以及是否需要做ND2NZ或者NZ2ND的转换，主要用于Tiling函数内部调优。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetDoubleBuffer(bool a, bool b, bool c, bool bias, bool transND2NZ = true, bool transNZ2ND = true)

          **参数说明**
          
          - dequant_type：设置量化或反量化时的模式。
          
          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。
          )doc")
        .def(
            "set_fix_split",
            [](MatmulApiTilingBase& self, int32_t baseMIn, int32_t baseNIn, int32_t baseKIn) {
                return self.SetFixSplit(baseMIn, baseNIn, baseKIn);
            },
            "base_m_in"_a = -1, "base_n_in"_a = -1, "base_k_in"_a = -1,
            R"doc(
          设置A/B/C/Bias是否使能double buffer功能，以及是否需要做ND2NZ或者NZ2ND的转换，主要用于Tiling函数内部调优。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetFixSplit(int32_t baseMIn = -1, int32_t baseNIn = -1, int32_t baseKIn = -1)

          **参数说明**

          - dequant_type：设置量化或反量化时的模式。
          
          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **约束说明**

          - base_m*base_n个输出元素所占的存储空间大小不能超过L0C Buffer大小，即base_m * base_n * sizeof(C_TYPE) <= L0CSize。
          - base_m需要小于等于single_m按16个元素向上对齐后的值（如ceil(single_m/16)*16），base_n需要小于等于single_n以C0_size个元素向上对齐的值，其中single_m为单核内M轴长度，singleN为单核内N轴长度，half/bfloat16_t数据类型的C0_size为16，float数据类型的C0_size为8，int8_t数据类型的C0_size为32，int4b_t数据类型的C0_size为64。例如single_m=12，则base_m需要小于等于16，同时base_m需要满足分形对齐的要求，所以base_m只能取16；如果base_m取其他超过16的值，获取Tiling将失败。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_fix_split(16, 16, -1)  # 设置固定的base_m, bakse_n
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_mad_type", [](MatmulApiTilingBase& self, MatrixMadType madType) { return self.SetMadType(madType); },
            "mad_type"_a,
            R"doc(
          设置是否使能HF32模式。当前版本暂不支持。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetMadType(MatrixMadType madType)

          **参数说明**

          - mad_type：设置Matmul模式。
          
          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。
          )doc")
        .def(
            "set_matmul_config_params",
            [](MatmulApiTilingBase& self, int32_t mmConfigType, bool enableL1CacheUB, ScheduleType scheduleType,
               MatrixTraverse traverse, bool enVecND2NZ) {
                return self.SetMatmulConfigParams(mmConfigType, enableL1CacheUB, scheduleType, traverse, enVecND2NZ);
            },
            "mm_config_type"_a = 1, "enable_l1_cache_ub"_a = false, "schedule_type"_a = ScheduleType::INNER_PRODUCT,
            "traverse"_a = MatrixTraverse::NOSET, "en_vec_nd2nz"_a = false,
            R"doc(
          在计算Tiling时，用于自定义设置MatmulConfig参数。本接口中配置的参数对应的功能在Tiling与Kernel中需要保持一致，
          所以本接口中的参数取值，需要与Kernel侧对应的MatmulConfig参数值保持一致。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              void SetMatmulConfigParams(int32_t mmConfigTypeIn = 1, bool enableL1CacheUBIn = false, 
                                        ScheduleType scheduleTypeIn = ScheduleType::INNER_PRODUCT, 
                                        MatrixTraverse traverseIn = MatrixTraverse::NOSET, bool enVecND2NZIn = false)
              void SetMatmulConfigParams(const MatmulConfigParams& configParams)

          **参数说明**

          - mm_config_type_in：设置Matmul的模板类型，需要与Matmul对象创建的模板一致，当前只支持配置为0或1。
          - enable_l1_cache_ub_in：配置是否使能L1缓存UB计算块；参考使能场景：MTE3和MTE2流水串行较多的场景。
          - schedule_type_in：配置Matmul数据搬运模式。
          - traverse_in：Matmul做矩阵运算的循环迭代顺序，即一次迭代计算出[baseM, baseN]大小的C矩阵分片后，自动偏移到下一次迭代输出的C矩阵位置的偏移顺序。
          - en_vec_nd2nz_in：是否使能ND2NZ。
          - config_params：config相关参数，类型为MatmulConfigParams。
          
          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **约束说明**

          - 本接口必须在GetTiling接口前调用。
          - 若Matmul对象使用NBuffer33模板策略，即MatmulPolicy为NBuffer33MatmulPolicy，则在调用GetTiling接口生成Tiling参数前，必须通过本接口将scheduleTypeIn参数设置为ScheduleType::N_BUFFER_33，以启用NBuffer33模板策略的Tiling生成逻辑。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling.set_matmul_config_params(0)    # 额外设置
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_matmul_config_params",
            [](MatmulApiTilingBase& self, const MatmulConfigParams& configParams) {
                return self.SetMatmulConfigParams(configParams);
            },
            "config_params"_a)
        .def(
            "set_org_shape",
            [](MatmulApiTilingBase& self, int32_t orgMIn, int32_t orgNIn, int32_t orgKIn) {
                return self.SetOrgShape(orgMIn, orgNIn, orgKIn);
            },
            "org_m_in"_a, "org_n_in"_a, "org_k_in"_a,
            R"doc(
          设置Matmul计算时的原始完整的形状M、N、K或Ka/Kb，单位均为元素个数。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKIn)
              int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKaIn, int32_t orgKbIn)

          **参数说明**

          - org_m_in：设置原始完整的形状M大小，单位为元素。
          - org_n_in：设置原始完整的形状N大小，单位为元素。
          - org_k_in：设置原始完整的形状K大小，单位为元素。原始完整形状Ka=Kb时可设置。
          - org_ka_in：设置矩阵A原始完整的形状Ka大小，单位为元素。
          - org_kb_in：	设置矩阵B原始完整的形状Kb大小，单位为元素。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **约束说明**

          参数org_ka_in和org_kb_in可以不相等，即原始矩阵形状Ka和Kb不相等，并不是实际Matmul计算时的K，此参数只用于辅助Matmul API搬运时的偏移计算。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_org_shape",
            [](MatmulApiTilingBase& self, int32_t orgMIn, int32_t orgNIn, int32_t orgKaIn, int32_t orgKbIn) {
                return self.SetOrgShape(orgMIn, orgNIn, orgKaIn, orgKbIn);
            },
            "org_m_in"_a, "org_n_in"_a, "org_ka_in"_a, "org_kb_in"_a)
        .def(
            "set_shape",
            [](MatmulApiTilingBase& self, int32_t m, int32_t n, int32_t k) { return self.SetShape(m, n, k); }, "m"_a,
            "n"_a, "k"_a,
            R"doc(
          设置Matmul计算的形状m、n、k，该形状可以为原始完整矩阵或其局部矩阵，单位为元素。该形状的矩阵乘可以由单核或多核计算完成。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetShape(int32_t m, int32_t n, int32_t k)

          **参数说明**

          - m：设置Matmul计算的M方向大小，单位为元素。
          - n：设置Matmul计算的N方向大小，单位为元素。
          - k：设置Matmul计算的K方向大小，单位为元素。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_sparse", [](MatmulApiTilingBase& self, bool isSparceIn) { return self.SetSparse(isSparceIn); },
            "is_sparce_in"_a = false,
            R"doc(
          设置Matmul的使用场景是否为Sparse Matmul场景。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetSparse(bool isSparseIn = false)

          **参数说明**

          - is_sparse_in：设置是否为Sparse Matmul稀疏场景。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **约束说明**

          本接口必须在get_tiling接口前调用。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_sparse(True)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_traverse",
            [](MatmulApiTilingBase& self, MatrixTraverse traverse) { return self.SetTraverse(traverse); }, "traverse"_a,
            R"doc(
          设置固定的Matmul计算方向，M轴优先还是N轴优先。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetTraverse(MatrixTraverse traverse)

          **参数说明**

          - traverse：设置固定的Matmul计算方向。可选值：MatrixTraverse::FIRSTM/MatrixTraverse::FIRSTN。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MatmulApiTiling(ascendc_platform)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_traverse(host.MatrixTraverse.FIRSTM)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_split_range",
            [](MatmulApiTilingBase& self, int32_t maxBaseM, int32_t maxBaseN, int32_t maxBaseK, int32_t minBaseM,
               int32_t minBaseN, int32_t minBaseK) {
                return self.SetSplitRange(maxBaseM, maxBaseN, maxBaseK, minBaseM, minBaseN, minBaseK);
            },
            "max_base_m"_a = -1, "max_base_n"_a = -1, "max_base_k"_a = -1, "min_base_m"_a = -1, "min_base_n"_a = -1,
            "min_base_k"_a = -1,
            R"doc(
          设置baseM/baseN/baseK的最大值和最小值。 目前Tiling暂时不支持该功能。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetSplitRange(int32_t maxBaseM = -1, int32_t maxBaseN = -1, int32_t maxBaseK = -1, 
                                    int32_t minBaseM = -1, int32_t minBaseN = -1, int32_t minBaseK = -1)

          **参数说明**

          - max_base_m：设置最大的baseM值，默认值为-1。-1表示不设置指定的baseM最大值，该值由Tiling函数自行计算。
          - max_base_n：设置最大的baseN值，默认值为-1。-1表示不设置指定的baseN最大值，该值由Tiling函数自行计算。
          - max_base_k：设置最大的baseK值，默认值为-1。-1表示不设置指定的baseK最大值，该值由Tiling函数自行计算。
          - min_base_m：设置最小的baseM值，默认值为-1。-1表示不设置指定的baseM最小值，该值由Tiling函数自行计算。
          - min_base_n：设置最小的baseN值，默认值为-1。-1表示不设置指定的baseN最小值，该值由Tiling函数自行计算。
          - min_base_k：设置最小的baseK值，默认值为-1。-1表示不设置指定的baseK最小值，该值由Tiling函数自行计算。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **约束说明**

          若base_m/base_n/base_k不满足C0_size对齐，计算Tiling时会将该值对齐到C0_size。提示，half/bfloat16_t数据类型的C0_size为16，float数据类型的C0_size为8，int8_t数据类型的C0_size为32，int4b_t数据类型的C0_size为64。
          )doc");

    // MatmulApiTiling class
    py::class_<MatmulApiTiling, MatmulApiTilingBase>(m, "MatmulApiTiling", py::module_local())
        .def(
            py::init<const platform_ascendc::PlatformAscendC&>(),
            R"doc(
          创建MatmulApiTiling对象。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              explicit MatmulApiTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
              MatmulApiTiling()

          **参数说明**

          - ascendc_platform：传入硬件平台的信息。

          **调用示例**
          
          - 无参构造函数

            .. code-block:: python
            
                import asc.lib.host as host
                # 单核Tiling
                tiling = host.MatmulApiTiling()
                tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
                ...
                tiling_data = host.TCubeTiling()
                ret = tiling.get_tiling(tiling_data)

          - 带参构造函数

            .. code-block:: python

                import asc.lib.host as host
                # 单核Tiling
                ascendc_platform = host.get_ascendc_platform()
                tiling = host.MatmulApiTiling(ascendc_platform)
                tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
                ...
                tiling_data = host.TCubeTiling()
                ret = tiling.get_tiling(tiling_data)
          )doc");

    // MultiCoreMatmulTiling class
    py::class_<MultiCoreMatmulTiling, MatmulApiTilingBase>(m, "MultiCoreMatmulTiling", py::module_local())
        .def(
            py::init<const platform_ascendc::PlatformAscendC&>(),
            R"doc(
          创建MultiCoreMatmulTiling对象。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              explicit MultiCoreMatmulTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
              MultiCoreMatmulTiling()

          **参数说明**

          - ascendc_platform：传入硬件平台的信息。

          **调用示例**
          
          - 无参构造函数

            .. code-block:: python
            
                import asc.lib.host as host
                # 多核Tiling
                tiling = host.MultiCoreMatmulTiling()
                tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
                ...
                tiling_data = host.TCubeTiling()
                ret = tiling.get_tiling(tiling_data)

          - 带参构造函数

            .. code-block:: python

                import asc.lib.host as host
                # 多核Tiling
                ascendc_platform = host.get_ascendc_platform()
                tiling = host.MultiCoreMatmulTiling(ascendc_platform)
                tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
                ...
                tiling_data = host.TCubeTiling()
                ret = tiling.get_tiling(tiling_data)
          )doc")
        // Enable methods
        .def(
            "enable_multi_core_split_k",
            [](MultiCoreMatmulTiling& self, bool flag) { return self.EnableMultiCoreSplitK(flag); }, "flag"_a,
            R"doc(
          多核场景，通过该接口使能切K轴。不调用该接口的情况下，默认不切K轴。在GetTiling接口调用前使用。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              void EnableMultiCoreSplitK(bool flag)

          **参数说明**

          - flag：是否使能切K轴。

          **约束说明**

          - 如果在算子中使用该接口，获取C矩阵结果时仅支持输出到Global Memory。
          - 如果在算子中使用该接口，需在Kernel侧代码中首次将C矩阵分片的结果写入Global Memory之前，先清零Global Memory，随后在获取C矩阵分片的结果时，再开启AtomicAdd累加。如果不预先清零Global Memory，可能会因为累加Global Memory中原始的无效数据而产生精度问题。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              tiling.set_dim(use_core_nums)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling.enable_multi_core_split_k(true);  // 使能切K轴
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)

          )doc")
        // Get methods
        .def(
            "get_core_num",
            [](MultiCoreMatmulTiling& self) -> py::object {
                int32_t dim, mDim, nDim;
                auto ret = self.GetCoreNum(dim, mDim, nDim);
                if (ret != 0) {
                    return py::none();
                } else {
                    return py::make_tuple(dim, mDim, nDim);
                }
            },
            R"doc(
          获得多核切分所使用的BlockNum参数。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t GetCoreNum(int32_t &dim, int32_t &mDim, int32_t &nDim)

          **返回值说明**
          
          以元组方式返回(dim, m_dim, n_dim)。

          **约束说明**

          使用创建的Tiling对象调用该接口，且需在完成Tiling计算（get_tiling）后调用。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              tiling.set_dim(use_core_nums)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_single_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
              # 获得多核切分后，使用的BlockNum
              dim, m_dim, n_dim = 0
              ret1 = tiling.get_core_num(dim, m_dim, n_dim)
          )doc")
        .def(
            "get_single_shape",
            [](MultiCoreMatmulTiling& self) -> py::object {
                int32_t shapeM, shapeN, shapeK;
                auto ret = self.GetSingleShape(shapeM, shapeN, shapeK);
                if (ret != 0) {
                    return py::none();
                } else {
                    return py::make_tuple(shapeM, shapeN, shapeK);
                }
            },
            R"doc(
          获取计算后的single_core_m/single_core_n/single_core_k。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t GetSingleShape(int32_t &shapeM, int32_t &shapeN, int32_t &shapeK)

          **参数说明**

          - flag：是否使能切K轴。

          **返回值说明**
          
          以元组方式返回(single_core_m, single_core_m, single_core_k)。

          **约束说明**

          使用创建的Tiling对象调用该接口，且需在完成Tiling计算（get_tiling）后调用。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              tiling.set_dim(use_core_nums)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_single_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
              # 获取计算后的singleCoreM/singleCoreN/singleCoreK
              single_m, single_n, single_k = 0
              ret = tiling.get_single_shape(single_m, single_n, single_k)
          )doc")

        // Set methods
        .def(
            "set_align_split",
            [](MultiCoreMatmulTiling& self, int32_t alignM, int32_t alignN, int32_t alignK) {
                return self.SetAlignSplit(alignM, alignN, alignK);
            },
            "align_m"_a, "align_n"_a, "align_k"_a,
            R"doc(
          多核切分时， 设置single_core_m/single_core_n/single_core_k的对齐值。比如设置single_core_m的对齐值为64（单位为元素），切分出的singleCoreM为64的倍数。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetAlignSplit(int32_t alignM, int32_t alignN, int32_t alignK)

          **参数说明**

          - align_m：single_core_m的对齐值。若传入-1或0，表示不设置指定的single_core_m的对齐值，该值由Tiling函数自行计算。
          - align_n：single_core_n的对齐值。若传入-1或0，表示不设置指定的single_core_n的对齐值，该值由Tiling函数自行计算。
          - align_k：single_core_k的对齐值。若传入-1或0，表示不设置指定的single_core_k的对齐值，该值由Tiling函数自行计算。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              tiling.set_dim(use_core_nums)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_align_split(-1, 64, -1);  # 设置single_core_m/single_core_n/single_core_k的对齐值
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret1 = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_dim", [](MultiCoreMatmulTiling& self, int32_t dim) { return self.SetDim(dim); }, "dim"_a,
            R"doc(
          设置多核Matmul时，参与运算的核数。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetDim(int32_t dim)

          **参数说明**

          - dim：多核Matmul tiling计算时，可以使用的核数。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              tiling.set_dim(use_core_nums) # 设置参与运算的核数
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_single_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_single_shape",
            [](MultiCoreMatmulTiling& self, int32_t singleMIn, int32_t singleNIn, int32_t singleKIn) {
                return self.SetSingleShape(singleMIn, singleNIn, singleKIn);
            },
            "single_m_in"_a = -1, "single_n_in"_a = -1, "single_k_in"_a = -1,
            R"doc(
          设置Matmul单核计算的形状single_m_in，single_n_in，single_k_in，单位为元素。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetSingleShape(int32_t singleMIn = -1, int32_t singleNIn = -1, int32_t singleKIn = -1)

          **参数说明**

          - single_m_in：设置的single_m_in大小，单位为元素，默认值为-1。-1表示不设置指定的single_m_in，该值由tiling函数自行计算。
          - single_n_in：设置的single_n_in大小，单位为元素，默认值为-1。-1表示不设置指定的single_n_in，该值由tiling函数自行计算。
          - single_k_in：设置的single_k_in大小，单位为元素，默认值为-1。-1表示不设置指定的single_k_in，该值由tiling函数自行计算。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              tiling.set_dim(use_core_nums)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)    # 设置Matmul单次计算的形状
              tiling.set_single_shape(1024, 1024, 1024) # 设置单核计算的形状
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc")
        .def(
            "set_single_range",
            [](MultiCoreMatmulTiling& self, int32_t maxM, int32_t maxN, int32_t maxK, int32_t minM, int32_t minN,
               int32_t minK) { return self.SetSingleRange(maxM, maxN, maxK, minM, minN, minK); },
            "max_m"_a = -1, "max_n"_a = -1, "max_k"_a = -1, "min_m"_a = -1, "min_n"_a = -1, "min_k"_a = -1,
            R"doc(
          设置single_core_m/single_core_n/single_core_k的最大值与最小值。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t SetSingleRange(int32_t maxM = -1, int32_t maxN = -1, int32_t maxK = -1, int32_t minM = -1, int32_t minN = -1, int32_t minK = -1)

          **参数说明**

          - max_m：设置最大的single_core_m值，默认值为-1，表示不设置指定的single_core_m最大值，该值由Tiling函数自行计算。
          - max_n：设置最大的single_core_n值，默认值为-1，表示不设置指定的single_core_n最大值，该值由Tiling函数自行计算。
          - max_k：设置最大的single_core_k值，默认值为-1，表示不设置指定的single_core_k最大值，该值由Tiling函数自行计算。
          - min_m：设置最小的single_core_m值，默认值为-1，表示不设置指定的single_core_m最小值，该值由Tiling函数自行计算。
          - min_n：设置最小的single_core_n值，默认值为-1，表示不设置指定的single_core_n最小值，该值由Tiling函数自行计算。
          - min_k：设置最小的single_core_k值，默认值为-1，表示不设置指定的single_core_k最小值，该值由Tiling函数自行计算。

          **返回值说明**
          
          -1表示设置失败； 0表示设置成功。
          
          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              tiling.set_dim(use_core_nums)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_single_range(1024, 1024, 1024, 1024, 1024, 1024) # 设置single_core_m/single_core_n/single_core_k的最大值与最小值
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
          )doc");

    // BatchMatmulTiling class
    py::class_<BatchMatmulTiling, MatmulApiTilingBase>(m, "BatchMatmulTiling", py::module_local())
        .def(
            py::init<const platform_ascendc::PlatformAscendC&>(),
            R"doc(
          创建BatchMatmulTiling对象。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              explicit BatchMatmulTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
              BatchMatmulTiling()

          **参数说明**

          - ascendc_platform：传入硬件平台的信息。

          **调用示例**
          
          - 无参构造函数

            .. code-block:: python
            
                import asc.lib.host as host
                # BatchMatmul Tiling
                tiling = host.BatchMatmulTiling()
                tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
                ...
                tiling_data = host.TCubeTiling()
                ret = tiling.get_tiling(tiling_data)

          - 带参构造函数

            .. code-block:: python

                import asc.lib.host as host
                # BatchMatmul Tiling
                ascendc_platform = host.get_ascendc_platform()
                tiling = host.BatchMatmulTiling(ascendc_platform)
                tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
                ...
                tiling_data = host.TCubeTiling()
                ret = tiling.get_tiling(tiling_data)
          )doc")
        // Get methods
        .def(
            "get_core_num",
            [](BatchMatmulTiling& self) -> py::object {
                int32_t dim, mDim, nDim, batchCoreM, batchCoreN;
                auto ret = self.GetCoreNum(dim, mDim, nDim, batchCoreM, batchCoreN);
                if (ret != 0) {
                    return py::none();
                } else {
                    return py::make_tuple(dim, mDim, nDim, batchCoreM, batchCoreN);
                }
            },
            R"doc(
          获得多核切分所使用的BlockNum参数。

          **对应的Ascend C函数原型**
          
          .. code-block:: c++

              int32_t GetCoreNum(int32_t &dim, int32_t &mDim, int32_t &nDim)

          **返回值说明**
          
          以元组方式返回(dim, m_dim, n_dim, batch_core_m,  batch_core_n)

          **约束说明**

          使用创建的Tiling对象调用该接口，且需在完成Tiling计算（get_tiling）后调用。

          **调用示例**
          
          .. code-block:: python
          
              import asc.lib.host as host
              ascendc_platform = host.get_ascendc_platform()
              tiling = host.MultiCoreMatmulTiling(ascendc_platform)
              tiling.set_dim(use_core_nums)
              tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
              tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
              tiling.set_shape(1024, 1024, 1024)
              tiling.set_single_shape(1024, 1024, 1024)
              tiling.set_org_shape(1024, 1024, 1024)
              tiling.set_bias(True)
              tiling.set_buffer_space(-1, -1, -1)
              tiling_data = host.TCubeTiling()
              ret = tiling.get_tiling(tiling_data)
              # 获得多核切分后，使用的BlockNum
              dim, m_dim, n_dim = 0
              ret1 = tiling.get_core_num(dim, m_dim, n_dim)
          )doc");
}
} // namespace asc
} // namespace pybind11
