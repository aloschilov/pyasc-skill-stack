# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Callable, Optional, TypeVar

T = TypeVar("T", bound=Callable)


def set_quant_scalar_docstring():
    func_introduction = """
    本接口提供对输出矩阵的所有值采用同一系数进行量化或反量化的功能，即整个C矩阵对应一个量化参数，量化参数的shape为[1]。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetQuantScalar(const uint64_t quantScalar)

    """

    param_list = """
    **参数说明**

    - quant_scalar：量化或反量化系数。
    """

    constraint_list = """
    **约束说明**

    - 需与set_dequant_type保持一致。
    - 本接口必须在iterate或者iterate_all前调用。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        tmp = 0.1
        ans = int.from_bytes(struct.pack('<f', tmp), 'little', signed=True) & 0xFFFFFFFF
        mm.set_quant_scalar(ans)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)
        mm.set_bias(gm_bias)
        mm.iterate_all(gm_c)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_quant_vector_docstring():
    func_introduction = """
    本接口提供对输出矩阵采用向量进行量化或反量化的功能，即对于输入shape为[1, N]的参数向量，
    N值为Matmul矩阵计算时M/N/K中的N值，对输出矩阵的每一列都采用该向量中对应列的系数进行量化或反量化。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetQuantVector(const GlobalTensor<uint64_t>& quantTensor)

    """

    param_list = """
    **参数说明**

    - quant_vector：量化或反量化运算时的参数向量。
    """

    constraint_list = """
    **约束说明**

    - 需与set_dequant_type保持一致。
    - 本接口必须在iterate或者iterate_all前调用。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        gm_quant = asc.GlobalTensor()
        ...
        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_quant_vector(gm_quant)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)
        mm.set_bias(gm_bias)
        mm.iterate_all(gm_c)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_org_shape_docstring():
    func_introduction = """
    设置Matmul计算原始完整的形状M、N、K，单位为元素个数。用于运行时修改shape，比如复用同一个Matmul对象，从不同的矩阵块取数据计算。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK)
        
    .. code-block:: c++

        __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc = 0)

    """

    param_list = """
    **参数说明**

    - org_m：设置原始完整的形状M大小，单位为元素。
    - org_n：设置原始完整的形状N大小，单位为元素。
    - org_ka：设置矩阵A原始完整的形状Ka大小，单位为元素。
    - org_kb：设置矩阵B原始完整的形状Kb大小，单位为元素。
    - org_kc：设置输出C矩阵的N，单位为元素。需要输入B矩阵的N和输出C矩阵的N不一样时可设置，默认为0（即使用B矩阵的N，不进行修改）。
    备注：Ascend C第一个函数原型对应的python参数：org_m，org_n，org_ka；Ascend C第二个函数原型对应的python参数：org_m，org_n，org_ka，org_kb，org_kc。
    """

    constraint_list = """
    **约束说明**

    - 本接口需要在set_tensor_a接口、set_tensor_b接口及set_single_shape接口前调用。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)
        mm.set_bias(gm_bias)
        mm.iterate_all(gm_c)
        # 复用mm对象
        mm.set_org_shape(org_m, org_n, org_k)
        mm.set_tensor_a(gm_a1)
        mm.set_tensor_b(gm_b1)
        mm.set_bias(gm_bias1)
        mm.iterate_all(gm_c1)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_single_shape_docstring():
    func_introduction = """
    设置Matmul单核计算的形状singleCoreM、singleCoreN、singleCoreK，单位为元素。
    用于运行时修改shape，比如复用Matmul对象来处理尾块。与SetTail接口功能一致，建议使用本接口。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK)

    """

    param_list = """
    **参数说明**

    - single_m：设置的singleCoreM大小，单位为元素。
    - single_n：设置的singleCoreN大小，单位为元素。
    - single_k：设置的singleCoreK大小，单位为元素。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)
        mm.set_bias(gm_bias)
        mm.set_single_shape(tail_m, tail_n ,tail_k)     # 如果是尾核，需要调整single_core_m/single_core_n/single_core_k
        mm.iterate_all(gm_c)
    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def set_self_define_data_docstring():
    func_introduction = """
    使能模板参数MatmulCallBackFunc（自定义回调函数）时，设置需要的计算数据或在GM上存储的数据地址等信息，用于回调函数使用。复用同一个Matmul对象时，可以多次调用本接口重新设置对应数据信息。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetSelfDefineData(const uint64_t dataPtr)

    .. code-block:: c++

        __aicore__ inline void SetSelfDefineData(T dataPtr)

    Ascend 910C 不支持SetSelfDefineData(T dataPtr)接口原型。
    Ascend 910B 不支持SetSelfDefineData(T dataPtr)接口原型。
    """

    param_list = """
    **参数说明**

    - data_ptr：设置的算子回调函数需要的计算数据或在GM上存储的数据地址等信息。其中，类型T支持用户自定义基础结构体。
    """

    constraint_list = """
    **约束说明**

    - 若回调函数中需要使用data_ptr参数时，必须调用此接口；若回调函数不使用data_ptr参数，无需调用此接口。
    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    - 本接口必须在set_tensor_a接口、set_tensor_b接口之前调用。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        data_gm_ptr = asc.GlobalTensor()    # 保存有回调函数需使用的计算数据的GM
        mm.set_self_define_data(data_gm_ptr)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)
        mm.iterate_all()
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_sparse_index_docstring():
    func_introduction = """
    设置稀疏矩阵稠密化过程生成的索引矩阵。
    索引矩阵的Format格式要求为NZ格式。
    本接口仅支持在纯Cube模式（只有矩阵计算）且MDL模板的场景使用。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetSparseIndex(const GlobalTensor<uint8_t>& indexGlobal)
    """

    param_list = """
    **参数说明**

    - index_global：索引矩阵在Global Memory上的首地址，类型为GlobalTensor。
    """

    constraint_list = """
    **约束说明**

    - 索引矩阵的Format格式要求为NZ格式。
    - 本接口仅支持在纯Cube模式（只有矩阵计算）且MDL模板的场景使用。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        @asc.jit(matmul_cube_only=True) # 使能纯Cube模式（只有矩阵计算）
        def matmul_kernel(...):
            ...
            asc.adv.register_matmul(pipe, workspace, mm, tiling)
            mm.set_tensor_a(gm_a)
            mm.set_tensor_b(gm_b)
            mm.set_sparse_index(gm_index) # 设置索引矩阵
            mm.set_bias(gm_bias)
            mm.iterate_all(gm_c)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_user_def_info_docstring():
    func_introduction = """
    使能模板参数MatmulCallBackFunc（自定义回调函数）时，设置算子tiling地址，用于回调函数使用，该接口仅需调用一次。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetUserDefInfo(const uint64_t tilingPtr)
    """

    param_list = """
    **参数说明**

    - tiling_ptr：设置的算子tiling地址。
    """

    constraint_list = """
    **约束说明**

    - 若回调函数中需要使用tiling_ptr参数时，必须调用此接口；若回调函数不使用tilingPtr参数，无需调用此接口。
    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        tiling_ptr = tiling    
        mm.set_user_def_info(tiling_ptr)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)
        mm.iterate_all()
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_matmul_api_tiling_docstring():
    func_introduction = """
    本接口用于在编译期间获取常量化的Matmul Tiling参数。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template<class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
        __aicore__ constexpr MatmulApiStaticTiling GetMatmulApiTiling(const MatmulConfig& mmCFG, int32_t l1Size = Impl::L1_SIZE)
    """

    param_list = """
    **参数说明**

    - mm_cfg：获取的MatmulConfig模板。
    - l1_size：可用的L1大小，默认值L1_SIZE。
    - a_type：A矩阵类型信息，通过MatmulType来定义。
    - b_type：B矩阵类型信息，通过MatmulType来定义。
    - c_type：C矩阵类型信息，通过MatmulType来定义。
    - bias_type：BIAS矩阵类型信息，通过MatmulType来定义。

    """

    return_list = """
    **返回值说明**

    MatmulApiStaticTiling，常量化Tiling参数。
    """

    constraint_list = """
    **约束说明**

    - 入参mm_cfg，在调用获取MatmulConfig模板的接口获取时，需要使用常数值指定(base_m, base_n, base_k)或者指定(base_m, base_n, base_k, single_core_m, single_core_n, single_core_k)，并且指定的参数值需要和tiling计算的值保持一致。
    - Batch Matmul场景支持全量常量化，但不支持使用空指针替代REGIST_MATMUL_OBJ的入参tiling。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # 定义Matmul对象
        a_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half)
        b_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half)
        c_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float)
        bias_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float)
        # 这里CFG使用get_normal_config接口获取，并指定已知的singleshape信息和base_m, base_n, base_k,指定的数值跟运行时tiling保持一致
        static_tiling = asc.adv.get_api_tiling(mm_cfg, 524288, a_type, b_type, c_type, bias_type)
        mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, static_tiling)
    """

    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def iterate_n_batch_docstring():
    func_introduction = """
    调用一次IterateNBatch，会进行N次IterateBatch计算，计算出N个多Batch的singleCoreM * singleCoreN大小的C矩阵。
    在调用该接口前，需将MatmulConfig中的isNBatch参数设为true，使能多Batch输入多Batch输出功能，并调用SetWorkspace接口申请临时空间，
    用于缓存计算结果，即IterateNBatch的结果输出至SetWorkspace指定的Global Memory内存中。
    对于BSNGD、SBNGD、BNGS1S2的Layout格式，
    调用该接口之前需要在tiling中使用SetALayout/SetBLayout/SetCLayout/SetBatchNum设置A/B/C的Layout轴信息和最大BatchNum数；
    对于Normal数据格式则需使用SetBatchInfoForNormal设置A/B/C的M/N/K轴信息和A/B矩阵的BatchNum数。
    实例化Matmul时，通过MatmulType设置Layout类型，当前支持3种Layout类型：BSNGD、SBNGD、BNGS1S2。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <bool sync = true, bool waitIterateBatch = false>
        __aicore__ inline void IterateNBatch(const uint32_t batchLoop, uint32_t batchA, uint32_t batchB, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0)

    """

    param_list = """
    **参数说明**

    - sync：设置同步或者异步模式。
    - wait_iterate_batch：是否需要通过WaitIterateBatch接口等待IterateNBatch执行结束，仅在异步场景下使用。
    - batch_loop：当前计算的BMM个数。
    - batch_a：当前单次BMM调用计算左矩阵的batch数。
    - batch_b：当前单次BMM调用计算右矩阵的batch数，brc场景batchA/B不相同。
    - en_sequential_write：输出是否连续存放数据。
    - matrix_stride_a：A矩阵源操作数相邻nd矩阵起始地址间的偏移，默认值是0。
    - matrix_stride_b：B矩阵源操作数相邻nd矩阵起始地址间的偏移，默认值是0。
    - matrix_stride_c：该参数预留，开发者无需关注。
    """

    constraint_list = """
    **约束说明**

    - 单BMM内计算遵循之前的约束条件。
    - 对于BSNGD、SBNGD、BNGS1S2 Layout格式，输入A、B矩阵多Batch数据总和应小于L1 Buffer的大小。
    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        @asc.jit
        def kernel_matmul_rpc_batch(a_gm: asc.GlobalAddress, b_gm: asc.GlobalAddress, c_gm: asc.GlobalAddress, bias_gm: asc.GlobalAddress, tiling: asc.adv.TCubeTiling, workspace_gm: asc.GlobalAddress, is_transpose_a_in: int, is_transpose_b_in: int, batch_a: int, batch_b: int):
            # 定义matmul type
            a_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half, False, asc.LayoutMode.BSNGD)
            b_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half, True, asc.LayoutMode.BSNGD)
            c_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float, False, asc.LayoutMode.BNGS1S2)
            bias_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float)
            a_global = asc.GlobalTensor()
            size_a = tiling.a_layout_info_b * tiling.a_layout_info_s * tiling.a_layout_info_n * tiling.a_layout_info_g * tiling.a_layout_info_d * 4
            size_a = tiling.b_layout_info_b * tiling.b_layout_info_s * tiling.b_layout_info_n * tiling.b_layout_info_g * tiling.b_layout_info_d * 4
            size_bias = tiling.c_layout_info_b * tiling.c_layout_info_n * tiling.c_layout_info_g * tiling.c_layout_info_s2 * 8
            a_global = set_global_buffer(a_gm, size_a)
            b_global = set_global_buffer(b_gm, size_b)
            bias_global = set_global_buffer(bias_gm, size_bias)
            tiling.share_mode = 0
            tiling.share_l1_size = 512 * 1024
            tiling.share_l0c_size = 128 * 1024
            tiling.share_ub_size = 0
            offset_a = 0
            offset_b = 0
            offset_c = 0
            offset_bias = 0
            a_global = a_global[offset_a]
            b_global = b_global[offset_b]
            bias_global = bias_global[offset_bias]
            # 创建Matmul实例
            mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type)
            pipe = asc.Tpipe()
            asc.adv.register_matmul(pipe, mm)
            mm.init(tiling)
            mm.set_tensor_a(a_global, is_transpose_a_in)
            mm.set_tensor_b(b_global, is_transpose_b_in)
            g_lay = tiling.a_layout_info_g
            if tiling.b_layout_info > g_lay:
                g_lay = tiling.b_layout_info_g
            for_extent = tiling.a_layout_info_b * tiling.a_layout_info_n * g_lay / tiling.batch_num
            mm.set_workspace(c_global)
            mm.iterate_n_batch(for_extent, batch_a, batch_b, False)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def end_docstring():
    func_introduction = """
    多个Matmul对象之间切换计算时，必须调用一次End函数，用于释放Matmul计算资源，防止多个Matmul对象的计算资源冲突。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void End()

    """

    param_list = """
    **参数说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        mm.iterate_all(gm_c)
        mm.end()
                
    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def set_hf32_docstring():
    func_introduction = """
    在纯Cube模式（只有矩阵计算）下，设置是否使能HF32（矩阵乘计算时可采用的数据类型）模式。使能后，在矩阵乘计算时，
    float32数据类型会转换为hf32数据类型，可提升计算性能，但同时也会带来精度损失。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetHF32(bool enableHF32 = false, int32_t transMode = 0)

    """

    param_list = """
    **参数说明**

    - enable_hf32：配置是否开启HF32模式，默认值false(不开启)。
    - trans_mode：配置在开启HF32模式时，float转换为hf32时所采用的ROUND模式。默认值0。
    """

    constraint_list = """
    **约束说明**

    本接口仅支持在纯Cube模式下调用
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.register_matmul(pip, mm, tiling)  # A/B/C/BIAS类型是float
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)
        mm.set_bias(gm_bias)
        mm.set_hf32(True)
        mm.iterate_all(gm_c)
        mm.set_hf32(False)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_tail_docstring():
    func_introduction = """
    在不改变Tiling的情况下，重新设置本次计算的singleCoreM/singleCoreN/singleCoreK，以元素为单位。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetTail(int tailM = -1, int tailN = -1, int tailK = -1)

    """

    param_list = """
    **参数说明**

    - tail_m：重新设置的singleCoreM值。
    - tail_n：重新设置的singleCoreN值。
    - tail_k：重新设置的singleCoreK值。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.register_matmul(pip, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)
        mm.set_bias(gm_bias)
        mm.set_tail(tail_m, tail_n, tail_k) # 如果是尾核，需要调整single_core_m/single_core_n/single_core_k
        mm.iterate_all(gm_c)
                
    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def set_batch_num_docstring():
    func_introduction = """
    在不改变Tiling的情况下，重新设置多Batch计算的Batch数。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetTail(int tailM = -1, int tailN = -1, int tailK = -1)

    """

    param_list = """
    **参数说明**

    - tail_m：重新设置的singleCoreM值。
    - tail_n：重新设置的singleCoreN值。
    - tail_k：重新设置的singleCoreK值。
    """

    constraint_list = """
    **约束说明**

    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    - 本接口仅支持在纯Cube模式（只有矩阵计算）下调用。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # 纯Cube模式
        mm.set_tensor_a(gm_a, is_transpose_a_in)
        mm.set_tensor_b(gm_b, is_transpose_b_in)
        if tiling.is_bias:
            mm.set_bias(gm_bias)
        mm.set_batch_num(batch_a, batch_b)
        # 多batch Matmul计算
        mm.iterate_batch(tensor=gm_c, en_partial_sum=False, en_atomic=0, en_sequential_write=False)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_workspace_docstring():
    func_introduction = """
    Iterate计算的异步场景，调用本接口申请一块临时空间来缓存计算结果，然后调用GetTensorC时会在该临时空间中获取C的矩阵分片。
    IterateNBatch计算时，调用本接口申请一块临时空间来缓存计算结果，然后根据同步或异步场景进行其它接口的调用。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <class T> __aicore__ inline void SetWorkspace(GlobalTensor<T>& addr)

    .. code-block:: c++

        template <class T> __aicore__ inline void SetWorkspace(__gm__ const T* addr, int size)

    """

    param_list = """
    **参数说明**

    - addr：用户传入的GM上的workspace空间，GlobalTensor类型。
    - addr：用户传入的GM上的workspace空间，GM地址类型。
    - size：传入GM地址时，需要配合传入元素个数。
    """

    constraint_list = """
    **约束说明**

    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_workspace(workspace_gm)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)
        mm.set_bias(gm_bias)
        mm.iterate(sync=True)
        for i in range(single_corem // base_m * single_core_n // base_n):
            mm.get_tensor_c(tensor=gm_c, sync=False)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def wait_get_tensor_c_docstring():
    func_introduction = """
    当使用GetTensorC异步接口将结果矩阵从GM拷贝到UB，且UB后续需要进行Vector计算时，需要调用WaitGetTensorC进行同步。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void WaitGetTensorC()

    """

    param_list = """
    **参数说明**

    无。
    """

    constraint_list = """
    **约束说明**

    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # 异步模式样例
        mm.iterate(sync=False)
        # 其他操作
        for i in range(single_corem // base_m * single_core_n // base_n):
            mm.get_tensor_c(tensor=ub_cmatrix, sync=False)
            mm.wait_get_tensor_c()
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_offset_c_docstring():
    func_introduction = """
    预留接口，为后续功能做预留。
    获取本次计算时当前分片在整个C矩阵中的位置。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline MatrixOffset GetOffsetC()

    """

    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **MatrixOffset结构体如下：**

    .. code-block:: c++

        struct MatrixOffset {   
            int32_t offset;   
            int32_t row, col;   
            int32_t height, width; 
        };
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", ""]


def async_get_tensor_c_docstring():
    func_introduction = """
    获取Iterate接口异步计算的结果矩阵。该接口功能已被GetTensorC覆盖，建议直接使用GetTensorC异步接口。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void AsyncGetTensorC(const LocalTensor<DstT>& c)

    """

    param_list = """
    **参数说明**

    - c：结果矩阵

    """

    constraint_list = """
    **约束说明**

    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, ""]


def get_tensor_c_docstring():
    func_introduction = """
    本接口和iterate接口配合使用，用于在调用iterate完成迭代计算后，
    根据MatmulConfig参数中的ScheduleType取值获取一块或两块baseM * baseN大小的矩阵分片。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <bool sync = true>
        __aicore__ inline void GetTensorC(const LocalTensor<DstT>& co2Local, uint8_t enAtomic = 0, bool enSequentialWrite = false)

    .. code-block:: c++

        template <bool sync = true>
        __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0, bool enSequentialWrite = false)

    .. code-block:: c++

        template <bool sync = true>
        __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& co2Local, uint8_t enAtomic = 0, bool enSequentialWrite = false)

    .. code-block:: c++

        template <bool sync = true>
        __aicore__ inline GlobalTensor<DstT> GetTensorC(uint8_t enAtomic = 0, bool enSequentialWrite = false)

    """

    param_list = """
    **参数说明**

    - tensor: 取出C矩阵到VECIN/GM。
    - en_atomic: 是否开启Atomic操作，默认值为0。
    - en_sequential_write: 是否开启连续写模式，默认值false。
    - sync: 设置同步或者异步模式。
    - optional_tensor: 取出C矩阵到VECIN，此参数使能时，tensor类型必须为GlobalTensor。
    """

    constraint_list = """
    **约束说明**

    - 传入的C矩阵地址空间大小需要保证不小于base_m * base_n。
    - 异步场景时，需要使用一块临时空间来缓存iterate计算结果，调用get_tensor_c时会在该临时空间中获取C的矩阵分片。临时空间通过set_workspace接口进行设置。set_workspace接口需要在iterate接口之前调用。
    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    """

    py_example = """
    **调用示例**

    - 获取C矩阵，输出至VECIN

      .. code-block:: python

          # 同步模式样例
          while mm.iterate() as count:
          mm.get_tensor_c(tensor=ub_cmatrix)
          # 异步模式样例
          mm.iterate(sync=False)
          # 其他操作
          for i in range(single_m // base_m * single_n // base_n):
              mm.get_tensor_c(tensor=ub_cmatrix, sync=False)

    - 获取C矩阵，输出至GM，同步模式样例

      .. code-block:: python

          while mm.iterate() as count:
              mm.get_tensor_c(tensor=gm)
    
    - 获取C矩阵，同时输出至GM和VECIN，同步模式样例

      .. code-block:: python

          while mm.iterate() as count:
              mm.get_tensor_c(tensor=gm, optional_tensor=ub_cmatrix)
    
    - 获取API接口返回的GM上的C矩阵，手动拷贝至UB，异步模式样例

      .. code-block:: python

          # base_m * base_n = 128 * 256
          mm.set_tensor_a(gm_a)
          mm.set_tensor_b(gm_b)
          mm.set_tail(single_m, single_n, single_k)
          mm.iterate(sync=False)
          for i in range(single_m // base_m * single_n // base_n):
              global = mm.get_tensor_c(sync=False)
              for j in range(4):
                  local = que.alloc_tensor(dtype=asc.half)
                  asc.data_copy(local, global[64* 128 * i:], count=64 * 128)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def iterate_docstring():
    func_introduction = """
    每调用一次Iterate，会计算出一块baseM * baseN的C矩阵。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <bool sync = true>
        __aicore__ inline bool Iterate(bool enPartialSum = false)

    .. code-block:: c++

        template <bool sync = true, typename T>
        __aicore__ inline bool Iterate(bool enPartialSum, const LocalTensor<T>& localCmatrix)

    """

    param_list = """
    **参数说明**

    - en_partial_sum: 是否将矩阵乘的结果累加于现有的CO1数据，默认值为false。
    - sync: 设置同步或者异步模式。
    - local_c_matrix: 由用户申请的CO1上的LocalTensor内存，用于存放矩阵乘的计算结果。
    """

    constraint_list = """
    **约束说明**

    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    - 对于用户自主管理CO1的iterate函数，创建Matmul对象时，必须定义C矩阵的内存逻辑位置为TPosition::CO1、数据排布格式为CubeFormat::NZ、数据类型为float或int32_t。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # 同步模式样例
        while mm.iterate() as count:
            mm.get_tensor_c(tensor=ub_cmatrix)
        # 异步模式样例
        mm.iterate(sync=False)
        # 其他操作
        for i in range(single_m // base_m * single_n // base_n):
            mm.get_tensor_c(tensor=ub_cmatrix, sync=False)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def iterate_all_docstring():
    func_introduction = """
    调用一次iterate_all，会计算出singleCoreM * singleCoreN大小的C矩阵。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <bool sync = true>
        __aicore__ inline void IterateAll(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0, bool enSequentialWrite = false, bool waitIterateAll = false, bool fakeMsg = false)
    
    .. code-block:: c++

        template <bool sync = true>
        __aicore__ inline void IterateAll(const LocalTensor<DstT>& ubCmatrix, uint8_t enAtomic = 0)

    """

    param_list = """
    **参数说明**

    - tensor: C矩阵，类型为GlobalTensor或LocalTensor。
    - en_atomic: 是否开启Atomic操作，默认值为0。
    - sync: 设置同步或者异步模式。
    - en_sequential_write: 是否开启连续写模式，仅支持输出到Global Memory场景。
    - wait_iterate_all: 是否需要通过wait_iterate_all接口等待iterate_all执行结束，仅支持异步输出到Global Memory场景。
    - fake_msg: 仅在IBShare场景和IntraBlockPartSum场景使用，仅在支持输出到Global Memory场景。
    """

    constraint_list = """
    **约束说明**

    - 传入的C矩阵地址空间大小需要保证不小于single_core_m * single_core_n个元素。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)
        mm.set_bias(gm_bias)
        mm.iterate_all(gm_c)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def wait_iterate_all_docstring():
    func_introduction = """
    等待iterate_all异步接口返回，支持连续输出到Global Memory。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK)

    """

    param_list = """
    **参数说明**

    无。
    """

    constraint_list = """
    **约束说明**

    - 配套iterate_all异步接口使用。
    - 仅支持连续输出至Global Memory。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        mm = asc.adv.Matmul(a_type, b_type, c_type, bais_type)
        mm.set_tensor_a(gm_a[offset_a:])
        mm.set_tensor_b(gm_b[offset_b:])
        if tiling.is_bias:
            mm.set_bias(gm_bias[offset_bias])
        mm.iterate_all(tensor=gm_c[offset_c], en_atomic=0, sync=False, en_sequential_write=False, wait_iterate_all=True)
        mm.wait_iterate_all()
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def iterate_batch_docstring():
    func_introduction = """
    该接口提供批量处理Matmul的功能，调用一次iterate_batch，可以计算出多个singleCoreM * singleCoreN大小的C矩阵。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <bool sync = true, bool waitIterateBatch = false>
        __aicore__ inline void IterateBatch(const GlobalTensor<DstT>& gm, uint32_t batchA, uint32_t batchB, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0, const bool enPartialSum = false, const uint8_t enAtomic = 0)

    .. code-block:: c++

        template <bool sync = true>
        __aicore__ inline void IterateBatch(const LocalTensor<DstT>& ubCmatrix, uint32_t batchA, uint32_t batchB, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0, const bool enPartialSum = false, const uint8_t enAtomic = 0)

    .. code-block:: c++

        __aicore__ inline void IterateBatch(const GlobalTensor<DstT>& gm, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0)

    .. code-block:: c++

        __aicore__ inline void IterateBatch(const LocalTensor<DstT>& ubCmatrix, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0)

    """

    param_list = """
    **参数说明**

    - tensor: C矩阵。类型为GlobalTensor或LocalTensor。
    - batch_a: 左矩阵的batch数。
    - batch_b: 右矩阵的batch数。
    - en_sequential_write: 是否开启连续写模式。
    - matrix_stride_a: A矩阵源操作数相邻nd矩阵起始地址间的偏移，单位是元素，默认值是0。
    - matrix_stride_b: B矩阵源操作数相邻nd矩阵起始地址间的偏移，单位是元素，默认值是0。
    - matrix_stride_c: 该参数预留，开发者无需关注。
    - en_partial_sum: 是否将矩阵乘的结果累加于现有的CO1数据，默认值为false。
    - en_atomic: 是否开启Atomic操作，默认值为0。
    - sync: 设置同步或者异步模式。
    - wait_iterate_batch: 是否需要通过wait_iterate_batch接口等待iterate_batch执行结束，仅在异步场景下使用。
    """

    constraint_list = """
    **约束说明**

    - 该接口只支持Norm模板，即BatchMatmul只支持Norm模板。
    - 对于BSNGD、SBNGD、BNGS1S2 Layout格式，输入A、B矩阵按分形对齐后的多Batch数据总和应小于L1 Buffer的大小；对于NORMAL Layout格式没有这种限制，但需通过MatmulConfig配置输入A、B矩阵多Batch数据大小与L1 Buffer的大小关系；
    - 对于BSNGD、SBNGD、BNGS1S2 Layout格式，称左矩阵、右矩阵的G轴分别为a_layout_info_g、b_layout_info_g，则a_layout_info_g / batch_a = b_layout_info_g / batch_b；对于NORMAL Layout格式，batch_a、batch_b必须满足倍数关系。
    - 如果接口输出到Unified Buffer上，输出C矩阵大小Base_m*Base_n应小于分配的Unified Buffer内存大小。
    - 如果接口输出到Unified Buffer上，且单核计算的N方向大小single_core_n非32字节对齐，C矩阵的CubeFormat仅支持ND_ALIGN格式，输出C矩阵片时，自动将single_core_n方向上的数据补齐至32字节。
    - 对于BSNGD、SBNGD Layout格式，输入输出只支持ND格式数据。对于BNGS1S2、NORMAL Layout格式， 输入支持ND/NZ格式数据。
    - 对于BSNGD、SBNGD Layout格式，不支持连续写模式。
    - 该接口不支持量化模式，即不支持set_quant_scalar、set_quant_vector接口。
    - BSNGD场景，不支持一次计算多行SD，需要算子程序中循环计算，即(a_layout_info_n * a_layout_info_g) / batch_a、(b_layout_info_n * b_layout_info_g) / batch_b均为整数。
    - 异步模式不支持iterate_batch搬运到UB上。
    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    - 使用该接口时，A矩阵、B矩阵不支持int4b_t类型的输入，即BatchMatmul不支持int4b_t类型的矩阵输入。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # 定义matmul type
        a_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half, False, asc.LayoutMode.BSNGD)
        b_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half, True, asc.LayoutMode.BSNGD)
        c_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float, False, asc.LayoutMode.BNGS1S2)
        bias_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float)
        mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type)
        asc.adv.register_matmul(pipe, mm)
        mm.init(tiling)
        batch_c = batch_a
        if batch_b > batch_c:
            batch_c = batch_b
        g_lay = tiling.a_layout_info_g
        if tiling.b_layout_info > g_lay:
            g_lay = tiling.b_layout_info_g
        for_extent = tiling.a_layout_info_b * tiling.a_layout_info_n * g_lay / tiling.batch_num
        for i in range(for_extent):
            batch_offset_a = i * tiling.a_layout_info_d * batch_a
            batch_offset_b = i * tiling.b_layout_info_d * batch_b
            mm.set_tensor_a(gm_a[batch_offset_a], is_transpose_a_in)
            mm.set_tensor_b(gm_b[batch_offset_b], is_transpose_b_in)
            idx_c = i * batch_c
            if tiling.c_layout_info_g == 1 and (tiling.b_layout_info_g != 1 or tiling.a_layout_info_g != 1):
                d = tiling.b_layout_info_g
                if tiling.a_layout_info_g > d:
                    d = tiling.a_layout_info_g
                idx_c = idx_c // d
            if tiling.is_bias:
                batch_offset_bias = idx_c * tiling.c_layout_info_s2
                mm.ste_bias(gm_bias[batch_offset_bias])
            batch_offset_c = idx_c * tiling.c_layout_info_s2
            if c_type.layout == asc.LayoutMode.BNGS1S2:
                batch_offset_c = idx_c * tiling.c_layout_infos2 * tiling.c_layout_info_s1
            mm.iterate_batch(tensor=gm_c[offsetc], batch_a=batch_a, batch_b=batch_b, en_sequential_write=False)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def wait_iterate_batch_docstring():
    func_introduction = """
    等待iterate_batch异步接口或iterate_nbatch异步接口返回，支持连续输出到Global Memory。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void WaitIterateBatch()

    """

    param_list = """
    **参数说明**

    无。
    """

    constraint_list = """
    **约束说明**

    - 配套iterate_batchiIterate_n_batch异步接口使用。
    - 仅支持连续输出至Global Memory。
    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type)
        mm.set_tensor_a(gm_a[offset_a])
        mm.set_tensor_b(gm_b[offset_b])
        if tiling.is_bias:
            mm.set_bias(gm_bias[offset_bias])
        mm.iterate_batch(tensor=gm_c[offsetc], batch_a=batch_a, batch_b=batch_b, en_sequential_write=False)
        mm.wait_iterate_batch()
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_batch_tensor_c_docstring():
    func_introduction = """
    调用一次get_batch_tensor_c，会获取C矩阵片，该接口可以与iterate_n_batch异步接口配合使用。
    用于在调用iterate_n_batch迭代计算后，获取一片std::max(batch_a, batch_b) * singleCoreM * singleCoreN大小的矩阵分片。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <bool sync = true>
        __aicore__ inline GlobalTensor<DstT> GetBatchTensorC(uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)

    .. code-block:: c++

        template <bool sync = true>
        __aicore__ inline void GetBatchTensorC(const LocalTensor<DstT>& c, uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)

    """

    param_list = """
    **参数说明**

    - batch_a: 左矩阵的batch数。
    - batch_b: 右矩阵的batch数。
    - en_sequential_write: 该参数预留，开发者无需关注。
    - tensor: C矩阵放置于Local Memory的地址，用于保存矩阵分片。
    """

    constraint_list = """
    **约束说明**

    - 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
    - C矩阵片输出到Local Memory，且单核计算的N方向大小single_core_n非32字节对齐的场景，C矩阵的CubeFormat仅支持ND_ALIGN格式，输出C矩阵片时，自动将single_core_n方向上的数据补齐至32字节。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        for_extent = tiling.a_layout_info_b * tiling.a_layout_info_n * g_lay // tiling.batch_num
        mm.set_tensor_a(gm_a, is_transpose_a_in)
        mm.set_tensor_b(gm_b, is_transpose_b_in)
        if tiling.is_bias:
            mm.set_bias(gm_bias)
        mm.iterate_n_batch(for_extent, batch_a, batch_b, False, sync=False)
        # ...其他计算
        for i in range(for_extent):
            mm.get_batch_tensor_c(tensor=ub_cmatrix, sync=False)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_tensor_a_docstring():
    func_introduction = """
    设置矩阵乘的左矩阵A。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetTensorA(const GlobalTensor<SrcAT>& gm, bool isTransposeA = false)

    .. code-block:: c++

        __aicore__ inline void SetTensorA(const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA = false)

    .. code-block:: c++

        __aicore__ inline void SetTensorA(SrcAT aScalar)

    """

    param_list = """
    **参数说明**

    - scalar: A矩阵中设置的值，为标量。
    - tensor: A矩阵。类型为GlobalTensor或LocalTensor。
    - transpose: A矩阵是否需要转置。
    """

    constraint_list = """
    **约束说明**

    - 传入的TensorA地址空间大小需要保证不小于single_m * single_k。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        # 示例一：左矩阵在Global Memory
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)
        mm.set_bias(gm_bias)
        mm.iterate_all(gm_c)
        # 示例二：左矩阵在Local Memory
        mm.set_tensor_a(local_a)
        # 示例三：设置标量数据
        mm.set_tensor_a(scalar_a)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_tensor_b_docstring():
    func_introduction = """
    设置矩阵乘的右矩阵B。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetTensorB(const GlobalTensor<SrcBT>& gm, bool isTransposeB = false)

    .. code-block:: c++

        __aicore__ inline void SetTensorB(const LocalTensor<SrcBT>& leftMatrix, bool isTransposeB = false)

    .. code-block:: c++

        __aicore__ inline void SetTensorB(SrcBT bScalar)

    """

    param_list = """
    **参数说明**

    - scalar: B矩阵中设置的值，为标量。
    - tensor: B矩阵。类型为GlobalTensor或LocalTensor。
    - transpose: B矩阵是否需要转置。
    """

    constraint_list = """
    **约束说明**

    - 传入的TensorB地址空间大小需要保证不小于single_k * single_n。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)   # 设置右矩阵B
        mm.set_bias(gm_bias)
        mm.iterate_all(gm_c)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_bias_docstring():
    func_introduction = """
    设置矩阵乘的Bias。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

         __aicore__ inline void SetBias(const GlobalTensor<BiasT>& biasGlobal)

    .. code-block:: c++
    
        __aicore__ inline void SetBias(const LocalTensor<BiasT>& inputBias)

    """

    param_list = """
    **参数说明**

    - tensor: Bias矩阵。类型为GlobalTensor或LocalTensor。
    """

    constraint_list = """
    **约束说明**

    - 在Matmul Tiling计算中，必须配置TCubeTiling结构中的is_bias参数为1，即使能Bias后，才能调用本接口设置Bias矩阵。
    - 传入的Bias地址空间大小需要保证不小于single_n。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)   
        mm.set_bias(gm_bias)    # 设置Bias
        mm.iterate_all(gm_c)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def disable_bias_docstring():
    func_introduction = """
    清除Bias标志位，表示Matmul计算时没有Bias参与。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void DisableBias()

    """

    param_list = """
    **参数说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)   
        mm.disable_bias()   # 清除tiling中的Bias标志位
        mm.iterate_all(gm_c)
                
    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def get_basic_config_docstring():
    func_introduction = """
    用于配置BasicBlock模板的参数，获取自定义BasicBlock模板。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ constexpr MatmulConfig GetBasicConfig(const uint32_t basicM, const uint32_t basicN, 
        const uint32_t basicK, const bool intrinsicsLimit = false, const bool batchLoop = false, 
        const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1)
    """

    param_list = """
    **参数说明**

    - basic_m: 用于设置参数basicM。与TCubeTiling结构体中的baseM参数含义相同，Matmul计算时base块M轴长度，以元素为单位。
    - basic_n: 用于设置参数basicN。与TCubeTiling结构体中的baseN参数含义相同，Matmul计算时base块N轴长度，以元素为单位。
    - basic_k: 用于设置参数basicK。与TCubeTiling结构体中的baseK参数含义相同，Matmul计算时base块K轴长度，以元素为单位。
    - intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
      - false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
      - true：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
    - batch_loop: 用于设置参数isNBatch。参数取值如下：
      - false：不使能多Batch（默认值）。
      - true：使能多Batch。
    - bmm_mode: 用于设置参数batchMode。参数取值如下：
      - batchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size。
      - batchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size。
      - batchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。
    """

    return_list = """
    **返回值说明**

    MatmulConfig结构体。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        mm_cfg = asc.adv.get_basic_config(128, 256, 64)
        mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, mm_cfg)
        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)   
        mm.set_bias(gm_bias)
        mm.iterate_all(gm_c)
                
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def get_special_basic_config_docstring():
    func_introduction = """
    用于配置SpecialBasicBlock模板的参数，获取自定义SpecialBasicBlock模板。当前为预留接口。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ constexpr MatmulConfig GetSpecialBasicConfig(const uint32_t basicM, const uint32_t basicN, 
        const uint32_t basicK, const uint32_t singleCoreM, const uint32_t singleCoreN, const uint32_t singleCoreK, 
        const uint32_t stepM, const uint32_t stepN, const bool intrinsicsLimit = false, const bool batchLoop = false, 
        const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1)
    """

    param_list = """
    **参数说明**
    
    - basic_m: 用于设置参数basicM。与TCubeTiling结构体中的baseM参数含义相同，Matmul计算时base块M轴长度，以元素为单位。
    - basic_n: 用于设置参数basicN。与TCubeTiling结构体中的baseN参数含义相同，Matmul计算时base块N轴长度，以元素为单位。
    - basic_k: 用于设置参数basicK。与TCubeTiling结构体中的baseK参数含义相同，Matmul计算时base块K轴长度，以元素为单位。
    - single_core_m: 用于设置参数singleCoreM。单核内M轴shape大小，以元素为单位。
    - single_core_n: 用于设置参数singleCoreN。单核内N轴shape大小，以元素为单位。
    - single_core_k: 用于设置参数singleCoreK。单核内K轴shape大小，以元素为单位。
    - step_m: 用于设置参数stepM。左矩阵在A1中缓存的bufferM方向上baseM的倍数。
    - step_n: 用于设置参数stepN。右矩阵在B1中缓存的bufferN方向上baseN的倍数。
    - intrinsics_limit: 用于设置参数intrinsicsCheck。
      当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否使能循环执行数据从Global Memory到
      L1 Buffer的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API
      内部通过循环执行数据的搬入。参数取值如下：
      - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
      - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
    - batch_loop: 用于设置参数isNBatch。
      是否多Batch输入多Batch输出。仅对BatchMatmul有效，使能该参数后，仅支持Norm模板，且需调用IterateNBatch实现多
      Batch输入多Batch输出。参数取值如下：
      - False：不使能多Batch（默认值）。
      - True：使能多Batch。
    - bmm_mode: 用于设置参数batchMode。
      BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与L1 Buffer的大小关系。
      参数取值如下：
      - batchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size。
      - batchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size。
      - batchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。
    """

    return_list = """
    **返回值说明**

    MatmulConfig结构体。
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", ""]


def get_normal_config_docstring():
    func_introduction = """
    用于配置Norm模板的参数，获取自定义Norm模板。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ constexpr MatmulConfig GetNormalConfig(const bool intrinsicsLimit = false, const bool batchLoop = false, 
        const bool isVecND2NZ = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1, const bool isMsgReuse = true, 
        const IterateOrder iterateOrder = IterateOrder::UNDEF, const ScheduleType scheduleType = ScheduleType::INNER_PRODUCT, 
        const bool enUnitFlag = true, const bool enableMixDualMaster = false)
    """

    param_list = """
    **参数说明**
    
    - intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
      - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
      - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
    - batch_loop: 用于设置参数isNBatch。参数取值如下：
      - False：不使能多Batch（默认值）。
      - True：使能多Batch。
    - is_vec_nd2_nz: 用于设置参数enVecND2NZ。参数取值如下：
      - False：不使能通过vector指令进行ND2NZ（默认值）。
      - True：使能通过vector指令进行ND2NZ。
    - bmm_mode: 用于设置参数batchMode。参数取值如下：
      - batchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size。
      - batchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size。
      - batchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。
    - is_msg_reuse: 用于设置参数enableReuse。参数取值如下：
      - True：直接传递计算数据，仅限单个值。
      - False：传递GM上存储的数据地址信息。
    - iterate_order: 用于设置参数iterateOrder。
    - schedule_type: 用于设置参数scheduleType。配置Matmul数据搬运模式。参数取值如下：
      - ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运。
      - ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运。
    - en_unit_flag: 用于设置参数enUnitFlag。参数取值如下：
      - False：不使能UnitFlag功能。
      - True：使能UnitFlag功能。
    """

    return_list = """
    **返回值说明**

    MatmulConfig结构体。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        mm_cfg = asc.adv.get_normal_config()
        mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, mm_cfg)
        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)   
        mm.set_bias(gm_bias)
        mm.iterate_all(gm_c)
                
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def get_mdl_config_docstring():
    func_introduction = """
    用于配置MDL模板的参数，获取自定义MDL模板。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ constexpr MatmulConfig GetMDLConfig(const bool intrinsicsLimit = false, const bool batchLoop = false, 
        const uint32_t doMTE2Preload = 0, const bool isVecND2NZ = false, bool isPerTensor = false, 
        bool hasAntiQuantOffset = false, const bool enUnitFlag = false, const bool isMsgReuse = true, 
        const bool enableUBReuse = true, const bool enableL1CacheUB = false, const bool enableMixDualMaster = false, 
        const bool enableKdimReorderLoad = false)
    """

    param_list = """
    **参数说明**
    
    - intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
      - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
      - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
    - batchLoop: 用于设置参数isNBatch。参数取值如下：
      - False：不使能多Batch（默认值）。
      - True：使能多Batch。
    - do_mte2_pre_load: 用于设置参数enVecND2NZ。参数取值如下：
      - False：不使能通过vector指令进行ND2NZ（默认值）。
      - True：使能通过vector指令进行ND2NZ。
    - is_vec_nd2_nz: 用于设置参数enVecND2NZ。参数取值如下：
      - False：不使能通过vector指令进行ND2NZ（默认值）。
      - True：使能通过vector指令进行ND2NZ。
    - is_per_tensor: 用于设置参数isPerTensor。参数取值如下：
      - True：per tensor量化。
      - False：per channel量化。
    - has_anti_quant_offset: 用于设置参数hasAntiQuantOffset。
    - en_unit_flag: 用于设置参数enUnitFlag。参数取值如下：
      - False：不使能UnitFlag功能。
      - True：使能UnitFlag功能。
    - is_msg_reuse: 用于设置参数enableReuse。参数取值如下：
      - True：直接传递计算数据，仅限单个值。
      - False：传递GM上存储的数据地址信息。
    - enable_ub_reuse: 用于设置参数enableUBReuse。参数取值如下：
      - True：使能Unified Buffer复用。
      - False：不使能Unified Buffer复用。
    - enable_l1_cache_ub: 用于设置参数enableL1CacheUB。参数取值如下：
      - True：使能L1 Buffer缓存Unified Buffer计算块。
      - False：不使能L1 Buffer缓存Unified Buffer计算块。
    - enable_mix_dual_master: 用于设置参数enableMixDualMaster。
    - enable_kdim_reorder_load: 用于设置参数enableKdimReorderLoad。
    """

    return_list = """
    **返回值说明**

    MatmulConfig结构体。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        mm_cfg = asc.adv.get_mdl_config()
        mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, mm_cfg)
        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)   
        mm.set_bias(gm_bias)
        mm.iterate_all(gm_c)
                
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def get_special_mdl_config_docstring():
    func_introduction = """
    用于配置SpecialMDL模板的参数，获取自定义SpecialMDL模板。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ constexpr MatmulConfig GetSpecialMDLConfig(const bool intrinsicsLimit = false, const bool batchLoop = false, 
        const uint32_t doMTE2Preload = 0, const bool isVecND2NZ = false, bool isPerTensor = false, bool hasAntiQuantOffset = false)
    """

    param_list = """
    **参数说明**
    
    - intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
      - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
      - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
    - do_mte2_pre_load: 用于设置参数enVecND2NZ。参数取值如下：
      - False：不使能通过vector指令进行ND2NZ（默认值）。
      - True：使能通过vector指令进行ND2NZ。
    - is_vec_nd2_nz: 用于设置参数enVecND2NZ。参数取值如下：
      - False：不使能通过vector指令进行ND2NZ（默认值）。
      - True：使能通过vector指令进行ND2NZ。
    - batch_loop: 用于设置参数isNBatch。参数取值如下：
      - False：不使能多Batch（默认值）。
      - True：使能多Batch。
    - is_per_tensor: 用于设置参数isPerTensor。参数取值如下：
      - True：per tensor量化。
      - False：per channel量化。
    - has_anti_quant_offset: 用于设置参数hasAntiQuantOffset。
    """

    return_list = """
    **返回值说明**

    MatmulConfig结构体。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        mm_cfg = asc.adv.get_special_mdl_config()
        mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, mm_cfg)
        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)   
        mm.set_bias(gm_bias)
        mm.iterate_all(gm_c)
                
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def get_ib_share_norm_config_docstring():
    func_introduction = """
    用于配置IBShare模板的参数，获取自定义IBShare模板。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ constexpr MatmulConfig GetIBShareNormConfig(const bool intrinsicsLimit = false, const bool batchLoop = false, 
        const bool isVecND2NZ = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1, const bool isDoubleCache = false, 
        const bool enUnitFlag = true)

    """

    param_list = """
    **参数说明**
    
    - intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
      - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
      - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
    - bmm_mode: 用于设置参数batchMode。参数取值如下：
      - batchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size。
      - batchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size。
      - batchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。
    - batch_loop: 用于设置参数isNBatch。参数取值如下：
      - False：不使能多Batch（默认值）。
      - True：使能多Batch。
    - is_vec_nd2_nz: 用于设置参数enVecND2NZ。参数取值如下：
      - False：不使能通过vector指令进行ND2NZ（默认值）。
      - True：使能通过vector指令进行ND2NZ。
    - is_double_cache: 用于设置参数enableDoubleCache。参数取值如下：
      - False：L1 Buffer上同时缓存一块数据（默认值）。
      - True：使能L1 Buffer上同时缓存两块数据。
    - en_unit_flag: 用于设置参数enUnitFlag。参数取值如下：
      - False：不使能UnitFlag功能。
      - True：使能UnitFlag功能。
    """

    return_list = """
    **返回值说明**

    MatmulConfig结构体。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        mm_cfg = asc.adv.get_ib_share_norm_config()
        mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, mm_cfg)
        asc.adv.register_matmul(pipe, workspace, mm, tiling)
        mm.set_tensor_a(gm_a)
        mm.set_tensor_b(gm_b)   
        mm.set_bias(gm_bias)
        mm.iterate_all(gm_c)
                
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def get_mm_config_docstring():
    func_introduction = """
    灵活的自定义Matmul模板参数配置。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <MatmulConfigMode configMode, typename... ArgTypes>
        __aicore__ inline constexpr MatmulConfig GetMMConfig(ArgTypes&&... args)

    """

    param_list = """
    **参数说明**
    
    MatmulShapeParams参数：

    - single_core_m: 单核内M轴shape大小，以元素为单位。
    - single_core_n: 单核内N轴shape大小，以元素为单位。
    - single_core_n: 单核内K轴shape大小，以元素为单位。
    - basic_m: Matmul计算时base块M轴长度，以元素为单位。
    - basic_n: Matmul计算时base块N轴长度，以元素为单位。
    - basic_k: Matmul计算时base块K轴长度，以元素为单位。

    MatmulQuantParams参数：

    - is_per_tensor: A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否为per tensor。

      - True：per tensor量化。
      - False：per channel量化。

    - has_anti_quant_offset: A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否使用offset系数。

    MatmulBatchParams参数：

    - is_n_batch: 是否多Batch输入多Batch输出。仅对BatchMatmul有效。参数取值如下：

      - False：不使能多Batch（默认值）。
      - True：使能多Batch。
      
    - batch_mode: 用于设置参数BatchMode。
      batchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与L1 Buffer的大小关系。
      参数取值如下：
      - batchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size；
      - batchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size；
      - batchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。
    - is_bias_batch: 批量多Batch的Matmul场景，即BatchMatmul场景，Bias的大小是否带有Batch轴。参数取值如下：
      - True: Bias带有Batch轴，Bias大小为Batch * N（默认值）。
      - False: Bias不带Batch轴，Bias大小为N，多Batch计算Matmul时，会复用Bias。
      - MatmulFuncParams参数：
    - intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
      - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）；
      - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
    - en_vec_nd2_nz: 使能通过vector指令进行ND2NZ。参数取值如下：
      - False：不使能通过vector指令进行ND2NZ（默认值）；
      - True：使能通过vector指令进行ND2NZ。
    - enable_l1_cache: 是否使能L1 Buffer缓存Unified Buffer计算块。参数取值如下：
      - True: 使能L1 Buffer缓存Unified Buffer计算块。
      - False: 不使能L1 Buffer缓存Unified Buffer计算块。
    - do_mte2_preload: 在MTE2流水间隙较大，且M/N数值较大时可通过该参数开启对应M/N方向的预加载功能，开启后能减小MTE2间隙，提升性能。
      预加载功能仅在MDL模板有效（不支持SpecialMDL模板）。参数取值如下：
      - 0： 不开启（默认值）。
      - 1: 开启M方向preload。
      - 2: 开启N方向preload。
    - iterate_order: 用于设置参数iterateOrder。
    - schedule_type: 用于设置参数scheduleType。配置Matmul数据搬运模式。参数取值如下：
      - scheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运；
      - scheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运。
    - enable_reuse: SetSelfDefineData函数设置的回调函数中的dataPtr是否直接传递计算数据。参数取值如下：
      - False：直接传递计算数据，仅限单个值。
      - True：传递GM上存储的数据地址信息。
    - enable_ub_reuse: 是否使能Unified Buffer复用。参数取值如下：
      - False：使能Unified Buffer复用。
      - True：不使能Unified Buffer复用。
    - is_partial_output: 是否开启PartialOutput功能。参数取值如下：
      - False：开启PartialOutput功能，一次Iterate的K轴不进行累加计算，Matmul每次计算输出局部baseK的baseM * baseN大小的矩阵分片。
      - True：不开启PartialOutput功能，一次Iterate的K轴进行累加计算，Matmul每次计算输出SingleCoreK长度的baseM * baseN大小的矩阵分片。
    - is_a2_b2_shared: 是否开启A2和B2的全局管理，即控制所有Matmul对象是否共用A2和B2的double buffer机制。参数取值如下：
      - True：开启。
      - False：关闭（默认值）。
    - is_enable_channel_split: 是否使能channel_split功能。参数取值如下：
      - False：默认值，不使能channel_split功能，输出的分形为16*16。
      - True：使能channel_split功能，输出的分形为16*8。
    - enable_kdim_reorder_load: 是否使能K轴错峰加载数据。
      - False：默认值，关闭K轴错峰加载数据的功能。
      - True：开启K轴错峰加载数据的功能。
    """

    return_list = """
    **返回值说明**

    MatmulConfig结构体。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # 获取MatmulConfig模板为Norm模板
        config_mode = asc.adv.MatmulConfigMode.CONFIG_NORM
        # single_core_m、single_core_n、single_core_k、basic_m、basic_n、basic_k
        shape_params = asc.adv.MatmulShapeParams(128, 128, 128, 64, 64, 64)
        # B矩阵量化时为per channel且不适用offset系数
        quant_params = asc.adv.MatmulQuantParams(False, False)
        # 不使能多Batch
        batch_params = asc.adv.MatmulBatchParams(False)
        #不进行芯片指令搬运地址偏移量校验，使能通过vector进行ND2NZ
        func_params = asc.adv.MatmulFuncParams(False, True)
        mm_config = asc.adv.get_mm_config(shape_params, quant_params, batch_params, func_params, config_mode)
                
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def get_init_docstring():
    func_introduction = """
    灵活的自定义Matmul模板参数配置。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe = nullptr)

    """

    param_list = """
    **参数说明**
    
    - cube_tiling: Matmul Tiling参数.
    - tpipe: Tpipe对象。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        regist_matmul(&pipe, get_sys_workspace_ptr(), mm)
        mm.init(&tiling)
                
    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def set_regist_matmul_docstring():
    func_introduction = """
    主要用于初始化Matmul对象。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        REGIST_MATMUL_OBJ(tpipe, workspace, ...)

    """

    param_list = """
    **参数说明**
    
    - tpipe: Tpipe对象。
    - workspace: 系统workspace指针。
    - &args: 可变参数，传入Matmul对象和与之对应的Tiling结构。
    """

    constraint_list = """
    **约束说明**

    - 在分离模式中，本接口必须在init_buffer接口前调用。
    - 在程序中，最多支持定义4个Matmul对象。
    - 当代码中只有一个Matmul对象时，本接口可以不传入tiling参数，通过init接口单独传入tiling参数。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        pipe = asc.Tpipe()
        # 推荐：初始化单个matmul对象，传入tiling参数
        mm.register_matmul(pipe, workspace, mm, tiling)
        # 初始化单个matmul对象，未传入tiling参数。注意，该场景下需要使用Init接口单独传入tiling参数。这种方式将matmul对象的初始化和tiling的设置分离，比如，Tiling可变的场景，可通过这种方式多次对Tiling进行重新设置
        mm.register_matmul(pipe, workspace, mm)
        mm.init(&tiling)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


DOC_HANDLES = {
    "async_get_tensor_c": async_get_tensor_c_docstring,
    "disable_bias": disable_bias_docstring,
    "end": end_docstring,
    "get_basic_config": get_basic_config_docstring,
    "get_batch_tensor_c": get_batch_tensor_c_docstring,
    "get_ib_share_norm_config": get_ib_share_norm_config_docstring,
    "get_matmul_api_tiling": get_matmul_api_tiling_docstring,
    "get_mdl_config": get_mdl_config_docstring,
    "get_mm_config": get_mm_config_docstring,
    "get_normal_config": get_normal_config_docstring,
    "get_offset_c": get_offset_c_docstring,
    "get_special_basic_config": get_special_basic_config_docstring,
    "get_special_mdl_config": get_special_mdl_config_docstring,
    "get_tensor_c": get_tensor_c_docstring,
    "init": get_init_docstring,
    "iterate": iterate_docstring,
    "iterate_all": iterate_all_docstring,
    "iterate_batch": iterate_batch_docstring,
    "iterate_n_batch": iterate_n_batch_docstring,
    "register_matmul": set_regist_matmul_docstring,
    "set_batch_num": set_batch_num_docstring,
    "set_bias": set_bias_docstring,
    "set_hf32": set_hf32_docstring,
    "set_org_shape": set_org_shape_docstring,
    "set_quant_scalar": set_quant_scalar_docstring,
    "set_quant_vector": set_quant_vector_docstring,
    "set_self_define_data": set_self_define_data_docstring,
    "set_single_shape": set_single_shape_docstring,
    "set_sparse_index": set_sparse_index_docstring,
    "set_tail": set_tail_docstring,
    "set_tensor_a": set_tensor_a_docstring,
    "set_tensor_b": set_tensor_b_docstring,
    "set_user_def_info": set_user_def_info_docstring,
    "set_workspace": set_workspace_docstring,
    "wait_get_tensor_c": wait_get_tensor_c_docstring,
    "wait_iterate_all": wait_iterate_all_docstring,
    "wait_iterate_batch": wait_iterate_batch_docstring,
}


def set_matmul_docstring(api_name: Optional[str] = None) -> Callable[[T], T]:
    func_introduction = ""
    cpp_signature = ""
    param_list = ""
    return_list = ""
    constraint_list = ""
    py_example = ""

    if DOC_HANDLES.get(api_name) is None:
        raise RuntimeError(f"Invalid matmul api name {api_name}")

    handler = DOC_HANDLES.get(api_name)
    func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = handler()

    docstr = f"""
    {func_introduction}
    {cpp_signature}
    {param_list}
    {return_list}
    {constraint_list}
    {py_example}
    """

    def decorator(fn: T) -> T:
        fn.__doc__ = docstr
        return fn

    return decorator


def set_math_docstring(api_name: Optional[str] = None, append_text: str = "") -> Callable[[T], T]:

    func_introduction = f"""
    {append_text}
    """
    cpp_signature = """
    **对应的Ascend C函数原型**
    """

    cpp_signature0 = ""
    if api_name != 'Log':
        cpp_signature0 = f"""

    .. code-block:: c++

        template <typename T, bool isReuseSource = false>
        __aicore__ inline void {api_name}(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, 
                                            const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)

        """

    cpp_signature1 = ""
    if api_name not in {"Log", "Floor", "Ceil", "Round", "Lgamma"}:
        cpp_signature1 = f"""

    .. code-block:: c++

        template <typename T, bool isReuseSource = false>
        __aicore__ inline void {api_name}(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                            const LocalTensor<uint8_t>& sharedTmpBuffer)

        """

    cpp_signature2 = f"""

    .. code-block:: c++

        template <typename T, bool isReuseSource = false>
        __aicore__ inline void {api_name}(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, 
                                            const uint32_t calCount)

    """

    cpp_signature3 = ""
    if api_name not in {"Floor", "Ceil", "Round", "Lgamma"}:
        cpp_signature3 = f"""
        
    .. code-block:: c++

        template <typename T, bool isReuseSource = false>
        __aicore__ inline void {api_name}(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)

        """

    param_list = """
    **参数说明**
    
    - is_reuse_source：是否允许修改源操作数。
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。
    - temp_buffer：临时缓存。
    - count：参与计算的元素个数。
    """

    constraint_list = """
    **约束说明**

    - 不支持源操作数与目的操作数地址重叠。
    - 不支持temp_buffer与源操作数和目的操作数地址重叠。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = f"""
    **调用示例**

    .. code-block:: python

        pipe = asc.Tpipe()
        tmp_que = asc.TQue(asc.TPosition.VECCALC, 1)
        pipe.init_buffer(que=tmp_que, num=1, len=buffer_size)   # buffer_size 通过Host侧tiling参数获取
        shared_tmp_buffer = tmp_que.alloc_tensor(asc.uint8)
        # 输入tensor长度为1024，算子输入的数据类型为half，实际计算个数为512
        asc.adv.{api_name}(dst, src, count=512, temp_buffer=shared_tmp_buffer)

    """

    if api_name == 'Log':
        py_example = """
    **调用示例**

    .. code-block:: python

        asc.adv.log(dst, src)
        """
        constraint_list = """
    **约束说明**

    - 不支持源操作数与目的操作数地址重叠。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
        """
    docstr = f"""
    {func_introduction}
    {cpp_signature}
    {cpp_signature0}
    {cpp_signature1}
    {cpp_signature2}
    {cpp_signature3}
    {param_list}
    {constraint_list}
    {py_example}
    """

    def decorator(fn: T) -> T:
        fn.__doc__ = docstr
        return fn

    return decorator


def concat_docstring():
    func_introduction = """
    对数据进行预处理，将要排序的源操作数src一一对应的合入目标数据concat中，数据预处理完后，可以进行Sort。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void Concat(LocalTensor<T> &concat, const LocalTensor<T> &src,
                                        const LocalTensor<T> &tmp, const int32_t repeatTime)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数。
    - src：源操作数。
    - tmp：输入，临时空间，用于接口内部复杂计算的中间存储。数据类型与src一致。
    - repeat_time：输入，重复迭代次数，每次迭代处理16个元素，下次迭代跳至相邻的下一组16个元素。取值范围：[0, 255]。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        concat_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=128)
        value_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=128)
        concat_tmp_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=256)
        asc.concat(dst=concat_local, src=value_local, tmp=concat_tmp_local, repeat_time=8)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def extract_docstring():
    func_introduction = """
    处理Sort的结果数据，输出排序后的value和index。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void Extract(const LocalTensor<T> &dstValue,
                                        const LocalTensor<uint32_t> &dstIndex,
                                        const LocalTensor<T> &sorted,
                                        const int32_t repeatTime)
    """

    param_list = """
    **参数说明**

    - dst_value：排序结果的数值部分。
    - dst_index：排序结果的索引部分。
    - src：源操作数。
    - repeat_time：重复迭代次数。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dst_value_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=128)
        dst_index_local = asc.LocalTensor(dtype=asc.uint32, pos=asc.TPosition.VECOUT, addr=0, tile_size=128)
        sort_tmp_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=256)
        asc.extract(dst_value=dst_value_local, dst_index=dst_index_local, src=sort_tmp_local, repeat_time=4)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_sort_docstring(api_name: Optional[str] = None) -> Callable[[T], T]:
    func_introduction = ""
    cpp_signature = ""
    param_list = ""
    return_list = ""
    py_example = ""

    if api_name == "concat":
        func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = concat_docstring()
    elif api_name == "extract":
        func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = extract_docstring()
    else:
        raise RuntimeError(f"Invalid sort api name {api_name}")

    docstr = f"""
    {func_introduction}
    {cpp_signature}
    {param_list}
    {return_list}
    {constraint_list}
    {py_example}
    """

    def decorator(fn: T) -> T:
        fn.__doc__ = docstr
        return fn

    return decorator
