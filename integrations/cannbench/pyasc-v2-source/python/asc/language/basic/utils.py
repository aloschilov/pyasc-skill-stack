# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.ir_value import PlainValue, RuntimeInt, RuntimeNumeric, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import DefaultValued, OverloadDispatcher
from ..core.types import BinaryRepeatParams, UnaryRepeatParams

T = TypeVar("T", bound=Callable)


def print_valid_type(origin):
    converted = {}
    for key, types in origin.items():
        converted[key] = [t.name for t in types]
    return converted


def check_type(callee: str, dst: LocalTensor, src0: LocalTensor, src1: LocalTensor) -> None:
    valids = {"src": [KT.float16, KT.float32, KT.int16, KT.int32], "dst": [KT.float16, KT.float32, KT.int16, KT.int32]}
    valids_relu = {"src": [KT.float16, KT.float32, KT.int16], "dst": [KT.float16, KT.float32, KT.int16]}
    valids_relu_cast = {"src": [KT.float16, KT.float32, KT.int16], "dst": [KT.int8, KT.float16]}
    valids_int = {"src": [KT.int16, KT.uint16], "dst": [KT.int16, KT.uint16]}
    valids_float = {"src": [KT.float16, KT.float32], "dst": [KT.float16, KT.float32]}

    valids_map = {
        "add": valids,
        "add_deq_relu": {"src": [KT.int32], "dst": [KT.float16]},
        "add_relu": valids_relu,
        "add_relu_cast": valids_relu_cast,
        "bilinear_interpolation": {"src": [KT.float16], "dst": [KT.float16]},
        "bitwise_and": valids_int,
        "bitwise_or": valids_int,
        "div": valids_float,
        "fused_mul_add": valids_float,
        "fused_mul_add_relu": valids_float,
        "max": valids,
        "min": valids,
        "mul": valids,
        "mul_add_dst": valids_float,
        "mul_cast": {"src": [KT.float16], "dst": [KT.int8, KT.uint8]},
        "sub": valids,
        "sub_relu": valids_relu,
        "sub_relu_cast": valids_relu_cast,
    }

    check_dst_src = {"add_deq_relu", "add_relu_cast", "bilinear_interpolation", "mul_cast", "sub_relu_cast"}

    if dst.dtype not in valids_map.get(callee).get("dst"):
        raise TypeError(f"Invalid dst data type, got {dst.dtype}, expect {print_valid_type(valids_map.get(callee))}")
    if src0.dtype not in valids_map.get(callee).get("src"):
        raise TypeError(f"Invalid src0 data type, got {dst.dtype}, expect {print_valid_type(valids_map.get(callee))}")
    if src1.dtype not in valids_map.get(callee).get("src"):
        raise TypeError(f"Invalid src1 data type, got {dst.dtype}, expect {print_valid_type(valids_map.get(callee))}")
    if src0.dtype != src1.dtype:
        raise TypeError("Src0 and src1 must be same type.")
    if callee not in check_dst_src:
        if not (dst.dtype == src0.dtype and dst.dtype == src1.dtype):
            raise TypeError("Src0, src1 and dst must be same type.")


def check_type_transpose(callee: str, dst: LocalTensor, src: LocalTensor, *args) -> None:
    if dst.dtype != src.dtype:
        raise TypeError(f"For {callee}, dst and src tensor must have the same dtype, "
                        f"got dst: {dst.dtype}, src: {src.dtype}")

    if args:
        shared_tmp_buffer = args[0]
        if not isinstance(shared_tmp_buffer, LocalTensor):
            raise TypeError("shared_tmp_buffer must be a LocalTensor")
        if shared_tmp_buffer.dtype != KT.uint8:
            raise TypeError(f"shared_tmp_buffer must have dtype uint8, got {shared_tmp_buffer.dtype}")


def check_type_5hd(callee: str, dst_or_list, src_or_list) -> None:
    if isinstance(dst_or_list, LocalTensor):
        if dst_or_list.dtype != KT.uint64 or src_or_list.dtype != KT.uint64:
            raise TypeError(f"For {callee} with LocalTensor inputs, dtype must be uint64.")
    elif isinstance(dst_or_list, list):
        if not dst_or_list or not src_or_list:
            return

        if len(dst_or_list) != len(src_or_list):
            raise ValueError("For {callee}, dst_list and src_list must have the same length.")

        if isinstance(dst_or_list[0], LocalTensor):
            first_dtype = dst_or_list[0].dtype
            if any(t.dtype != first_dtype for t in dst_or_list) or \
               any(t.dtype != first_dtype for t in src_or_list):
                raise TypeError(f"For {callee}, all tensors in dst_list and src_list must have the same dtype.")
        else:
            if not all(isinstance(x, (PlainValue, int)) for x in dst_or_list) or \
               not all(isinstance(x, (PlainValue, int)) for x in src_or_list):
                raise TypeError(f"For {callee}, address lists must contain only RuntimeInt.")
    else:
        raise TypeError(f"Unsupported input types for {callee}: {type(dst_or_list)}")


def op_impl(callee: str, dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, args: Tuple[Any],
            kwargs: Dict[str, Any], build_l0: Callable, build_l1: Callable, build_l2: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")
    dispatcher = OverloadDispatcher(callee)

    check_type(callee, dst, src0, src1)

    @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, repeat_params=BinaryRepeatParams,
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams, is_set_mask: bool = True):
        build_l0(dst.to_ir(), src0.to_ir(), src1.to_ir(),
                 _mat(mask, KT.int64).to_ir(),
                 _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(mask=list, repeat_times=RuntimeInt, repeat_params=BinaryRepeatParams,
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: list, repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams, is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src0.to_ir(), src1.to_ir(), mask,
                 _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(count=RuntimeInt, is_set_mask=DefaultValued(bool, True))
    def _(count: RuntimeInt, is_set_mask: bool = True):
        build_l2(dst.to_ir(), src0.to_ir(), src1.to_ir(), _mat(count, KT.int32).to_ir())

    dispatcher(*args, **kwargs)


def vec_binary_scalar_op_impl(callee: str, dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, args: Tuple[Any],
                              kwargs: Dict[str,
                                           Any], build_l0: Callable, build_l1: Callable, build_l2: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")
    scalar = _mat(scalar, src.dtype).to_ir()
    dispatcher = OverloadDispatcher(callee)

    @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams,
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        build_l0(dst.to_ir(), src.to_ir(), scalar,
                 _mat(mask, KT.int64).to_ir(),
                 _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(mask=list, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams,
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: list, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src.to_ir(), scalar, mask,
                 _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(count=RuntimeInt, is_set_mask=DefaultValued(bool, True))
    def _(count: RuntimeInt, is_set_mask: bool = True):
        build_l2(dst.to_ir(), src.to_ir(), scalar, _mat(count, KT.int32).to_ir(), is_set_mask)

    dispatcher(*args, **kwargs)


def vec_ternary_scalar_op_impl(callee: str, dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric,
                               args: Tuple[Any], kwargs: Dict[str, Any], build_l0: Callable, build_l1: Callable,
                               build_l2: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")
    scalar = _mat(scalar, src.dtype).to_ir()
    dispatcher = OverloadDispatcher(callee)

    @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams,
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        build_l0(dst.to_ir(), src.to_ir(), scalar,
                 _mat(mask, KT.int64).to_ir(),
                 _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(mask=list, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams,
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: list, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src.to_ir(), scalar, mask,
                 _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(count=RuntimeInt)
    def _(count: RuntimeInt):
        build_l2(dst.to_ir(), src.to_ir(), scalar, _mat(count, KT.int32).to_ir())

    dispatcher(*args, **kwargs)


def cast_docstring():
    func_introduction = """
    根据源操作数和目的操作数Tensor的数据类型进行精度转换。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U>
        __aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& round_mode, const uint32_t count)

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& round_mode, const uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& round_mode, const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)

    """

    param_list = """
    **参数说明**

    - T: 目的操作数数据类型。支持的数据类型根据产品型号不同而有所区别。
    - U: 源操作数数据类型。支持的数据类型根据产品型号不同而有所区别。
    - is_set_mask: 是否在接口内部设置mask。true表示在接口内部设置mask，false表示在接口外部设置mask。
    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。起始地址需要32字节对齐。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。起始地址需要32字节对齐。
    - round_mode: 精度转换处理模式，类型是RoundMode枚举类，控制精度转换处理模式：
      - CAST_NONE: 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不舍入。
      - CAST_RINT: rint，四舍六入五成双舍入。
      - CAST_FLOOR: floor，向负无穷舍入。
      - CAST_CEIL: ceil，向正无穷舍入。
      - CAST_ROUND: round，四舍五入舍入。
      - CAST_TRUNC: trunc，向零舍入。
      - CAST_ODD: Von Neumann rounding，最近邻奇数舍入。
    - count: 参与计算的元素个数。
    - mask/mask[]: 用于控制每次迭代内参与计算的元素。支持连续模式和逐bit模式。
    - repeat_time: 重复迭代次数，取值范围[0,255]。
    - repeat_params: 控制操作数地址步长的参数，UnaryRepeatParams类型。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。特别地，对于长度较小的数据类型转换为长度较大的数据类型时，地址重叠可能会导致结果错误。
    - 每个repeat能处理的数据量取决于数据精度、AI处理器型号。
    - 当源操作数和目的操作数位数不同时，计算输入参数以数据类型的字节较大的为准。
    - 当dst或src为int4b_t时，申请Tensor空间时只需申请相同数量的int8_t数据空间的一半。
    - 当dst或src为int4b_t时，tensor高维切分计算接口的连续模式的mask与tensor前n个数据计算接口的count必须为偶数；对于逐bit模式，对应同一字节的相邻两个比特位的数值必须一致。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 256 // asc.int32.sizeof()
          params = asc.UnaryRepeatParams(1, 1, 8, 4)
          asc.cast(dst, src, asc.RoundMode.CAST_CEIL, mask=mask, repeat_times=8, params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [0, 0xFFFFFFFFFFFFFFFF]
          params = asc.UnaryRepeatParams(1, 1, 8, 4)
          asc.cast(dst, src, asc.RoundMode.CAST_CEIL, mask=mask, repeat_times=8, params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.cast(dst, src, asc.RoundMode.CAST_CEIL, count=512)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def copy_docstring():
    func_introduction = """
    在 Vector Core 的不同内部存储单元（VECIN, VECCALC, VECOUT）之间进行数据搬运。

    这是一个矢量指令，支持通过掩码（mask）进行灵活的数据块选择，并通过重复参数（repeat parameters）
    实现高效的间隔操作和高维数据处理。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    该接口支持两种掩码（mask）模式，以进行高维切分计算。

    1. **mask 为逐 bit 模式**

       .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                    const uint64_t mask[], const uint8_t repeatTime,
                                    const CopyRepeatParams& repeatParams)

    2. **mask 为连续模式**

       .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                    const uint64_t mask, const uint8_t repeatTime,
                                    const CopyRepeatParams& repeatParams)
    """

    param_list = """
    **参数说明**

    - dst (asc.LocalTensor): 目标操作数。

      - 必须是 `LocalTensor`。
      - 支持的 TPosition 为 `asc.Position.VECIN`, `asc.Position.VECCALC`, `asc.Position.VECOUT`。
      - 起始地址需要 32 字节对齐。

    - src (asc.LocalTensor): 源操作数。

      - 必须是 `LocalTensor`，且数据类型与 `dst` 保持一致。
      - 支持的 TPosition 为 `asc.Position.VECIN`, `asc.Position.VECCALC`, `asc.Position.VECOUT`。
      - 起始地址需要 32 字节对齐。

    - mask (Union[int, List[int]]): 掩码，用于控制在单次迭代中哪些元素参与搬运。

      - **连续模式** (当 `mask` 为 `int`): 表示从起始位置开始，连续搬运多少个元素。

        - 当数据类型为 16-bit (如 `fp16`) 时，取值范围是 [1, 128]。
        - 当数据类型为 32-bit (如 `fp32`) 时，取值范围是 [1, 64]。

      - **逐 bit 模式** (当 `mask` 为 `List[int]`): 掩码数组中的每个 bit 对应一个元素，bit 为 1 表示搬运，0 表示跳过。

        - 当数据类型为 16-bit 时，`mask` 是一个长度为 2 的列表，例如 `mask=[mask0, mask1]`。
        - 当数据类型为 32-bit 时，`mask` 是一个长度为 1 的列表，例如 `mask=[mask0]`。

    - repeat_time (int): 重复迭代次数。矢量计算单元每次处理一个数据块（256字节），此参数指定了处理整个 Tensor 需要重复迭代的次数。
    - repeat_params (asc.CopyRepeatParams): 控制地址步长的数据结构，用于处理高维或非连续数据。

      - `dstStride`, `srcStride`: 设置同一次迭代内，不同数据块（DataBlock）之间的地址步长。
      - `dstRepeatSize`, `srcRepeatSize`: 设置相邻两次迭代之间的地址步长。

    - is_set_mask (bool, 可选): 模板参数，默认为 `True`。

      - `True`: 在接口内部设置 `mask` 值。
      - `False`: 在接口外部通过 `asc.set_vector_mask` 接口设置 `mask`，此时 `mask` 参数必须为占位符 `asc.MASK_PLACEHOLDER`。
    """

    constraint_list = """
    **约束说明**

    - 源操作数和目的操作数的起始地址需要保证32字节对齐。
    - Copy和矢量计算API一样，支持和掩码操作API配合使用。但Counter模式配合高维切分计算API时，和通用的Counter模式有一定差异。具体差异如下：
      - 通用的Counter模式：Mask代表整个矢量计算参与计算的元素个数，迭代次数不生效。
      - Counter模式配合Copy高维切分计算API，Mask代表每次Repeat中处理的元素个数，迭代次数生效。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        TILE_LENGTH = 1024
        # 1. 定义源和目标 LocalTensor
        src_tensor = asc.LocalTensor(asc.fp16, asc.Position.VECIN, size=TILE_LENGTH)
        dst_tensor = asc.LocalTensor(asc.fp16, asc.Position.VECOUT, size=TILE_LENGTH)

        ...

        # 2. 定义地址步长参数
        # 示例：实现一个交错拷贝，源地址每次迭代跳 256 字节，目标地址连续
        params = asc.CopyRepeatParams(
            dstStride=1,       # 迭代内，目标 datablock 连续
            srcStride=2,       # 迭代内，源 datablock 间隔为 1 个 datablock
            dstRepeatSize=8,   # 迭代间，目标地址步长为 8 个元素
            srcRepeatSize=16   # 迭代间，源地址步长为 16 个元素
        )

        # 3. 使用连续模式调用 Copy
        # 每次迭代处理 128 个元素（一个 256 字节的 block），重复 4 次
        asc.copy(dst_tensor, src_tensor, mask=128, repeat_time=4, repeat_params=params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_wait_flag_docstring():
    func_introduction = """
    同一核内不同流水线之间的同步指令，具有数据依赖的不同流水指令之间需要插此同步。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetFlag(TEventID id)
        __aicore__ inline void WaitFlag(TEventID id)

    """

    param_list = """
    **参数说明**

    - id: 事件ID，由用户自己指定。
    """

    constraint_list = """
    **约束说明**

    - set_flag/wait_flag必须成对出现。
    - 禁止用户在使用set_flag和wait_flag时，自行指定event_id，容易与框架同步事件冲突，导致卡死问题。event_id需要通过alloc_event_id或者fetch_event_id来获取。
    """

    py_example = """
    **调用示例**

    如data_copy需要等待set_value执行完成后才能执行，需要插入PIPE_S到PIPE_MTE3的同步。

    .. code-block:: python

        dst = asc.GlobalTensor()
        src = asc.LocalTensor()
        src.set_value(0, 0)
        data_size = 512
        event_id = global_pipe.fetch_event_id(event=asc.HardEvent.S_MTE3)
        asc.set_flag(event=asc.HardEvent.S_MTE3, event_id=event_id)
        asc.wait_flag(event=asc.HardEvent.S_MTE3, event_id=event_id)
        asc.data_copy(dst, src, data_size)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def data_sync_barrier_docstring():
    func_introduction = """
    用于阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <MemDsbT arg0>
        __aicore__ inline void DataSyncBarrier()

    """

    param_list = """
    **参数说明**

    - arg0: 模板参数，表示需要等待的内存位置，类型为MemDsbT，可取值为：

      - ALL，等待所有内存访问指令。
      - DDR，等待GM访问指令。
      - UB，等待UB访问指令。
      - SEQ，预留参数，暂未启用，为后续的功能扩展做保留。
    """

    constraint_list = """
    **约束说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.mmad(...)
        asc.data_sync_barrier(asc.MemDsbT.ALL)
        asc.fixpipe(...)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def pipe_barrier_docstring():
    func_introduction = """
    阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <pipe_t pipe>
        __aicore__ inline void PipeBarrier()

    """

    param_list = """
    **参数说明**

    - pipe: 模板参数，表示阻塞的流水类别。
    """

    constraint_list = """
    **约束说明**

    Scalar流水之间的同步由硬件自动保证，调用pipe_barrier(PIPE_S)会引发硬件错误。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.add(dst0, src0, src1, 512)
        asc.pipe_barrier(asc.PipeID.PIPE_V)
        asc.mul(dst1, dst0, src2, 512)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def ib_set_docstring():
    func_introduction = """
    当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。
    调用ib_set设置某一个核的标志位，与ib_wait成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <bool isAIVOnly = true>
        __aicore__ inline void IBSet(
            const GlobalTensor<int32_t>& gmWorkspace,
            const LocalTensor<int32_t>&  ubWorkspace,
            int32_t blockIdx,
            int32_t eventID)
    """

    param_list = """
    **参数说明**

    - gmWorkspace: 外部存储核状态的公共缓存，类型为GlobalTensor。
    - ubWorkspace: 存储当前核状态的公共缓存。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - blockIdx: 表示等待核的idx号，取值范围：[0, 核数-1]。
    - eventID: 用来控制当前核的set、wait事件。
    - isAIVOnly:控制是否为AIVOnly模式，默认为true。
    """

    constraint_list = """
    **约束说明**

    - gmWorkspace申请的空间最少要求为：核数 * 32Bytes * eventID_max + blockIdx_max * 32Bytes + 32Bytes。（eventID_max和blockIdx_max分别指eventID、blockIdx的最大值 ）；
    - 注意：如果是AIVOnly模式，核数 = GetBlockNum()；如果是MIX模式，核数 = GetBlockNum() * 2；
    - ubWorkspace申请的空间最少要求为：32Bytes；
    - gmWorkspace缓存的值需要初始化为0。
    - 使用该接口进行多核控制时，算子调用时指定的逻辑blockNum必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        gm = asc.GlobalTensor()
        gm.set_global_buffer(x)
        ub = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=32)
        asc.ib_set(gm, ub, block_idx=0, event_id=0)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def ib_wait_docstring():
    func_introduction = """
    当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。
    ib_wait与ib_set成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。
    """

    cpp_signature = """
    **对应的Ascend C函数原型（软同步）**

    .. code-block:: c++

        template <bool isAIVOnly = true>
        __aicore__ inline void IBWait(
            const GlobalTensor<int32_t>& gmWorkspace,
            const LocalTensor<int32_t>&  ubWorkspace,
            int32_t blockIdx,
            int32_t eventID)
    """

    param_list = """
    **参数说明**

    - gmWorkspace: 外部存储核状态的公共缓存，类型为GlobalTensor。
    - ubWorkspace: 存储当前核状态的公共缓存。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - blockIdx: 表示等待核的idx号，取值范围：[0, 核数-1]。
    - eventID: 用来控制当前核的set、wait事件。
    - isAIVOnly:控制是否为AIVOnly模式，默认为true。
    """

    constraint_list = """
    **约束说明**

    - gmWorkspace申请的空间最少要求为：核数 * 32Bytes * eventID_max + blockIdx_max * 32Bytes + 32Bytes。（eventID_max和blockIdx_max分别指eventID、blockIdx的最大值 ）
    - ubWorkspace申请的空间最少要求为：32Bytes。
    - 使用该接口进行多核控制时，算子调用时指定的逻辑blockNum必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        gm = asc.GlobalTensor()
        gm.set_global_buffer(x)
        ub = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=32)
        asc.ib_wait(gm, ub, block_idx=0, event_id=0)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def sync_all_docstring():
    func_introduction = """
    当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。
    目前多核同步分为硬同步和软同步，硬件同步是利用硬件自带的全核同步指令由硬件保证多核同步，软件同步是使用软件算法模拟实现。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // 软同步
        template <bool isAIVOnly = true>
        __aicore__ inline void SyncAll(
            const GlobalTensor<int32_t>& gmWorkspace,
            const LocalTensor<int32_t>&  ubWorkspace,
            const int32_t usedCores = 0)

    .. code-block:: c++

        // 硬同步
        template <bool isAIVOnly = true>
        __aicore__ inline void SyncAll()
    """

    param_list = """
    **参数说明**

    - gmWorkspace: gmWorkspace为用户定义的全局Global空间，作为所有核共用的缓存，用于保存每个核的状态标记，类型为GlobalTensor，支持的数据类型为int32_t。
    - ubWorkspace: ubWorkspace为用户定义的局部Local空间，每个核单独自用，用于标记当前核的状态。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT，支持的数据类型为int32_t。
    - usedCores: 指定多少个核之间的同步，传入数值不能超过算子调用时指定的逻辑blockNum。此参数为默认参数，不传此参数表示全核软同步。
    - isAIVOnly: 控制SyncAll作用于纯Vector算子或融合（Cube和Vector融合）算子。可选值：
      - true（默认值）：纯Vector算子的全核同步，仅执行Vector核的全核同步。
      - false：融合算子的全核同步，先分别完成Vector核和Cube核的全核同步，再执行两者之间的同步（软同步接口不支持此功能）。
    """

    constraint_list = """
    **约束说明**

    - gmWorkspace缓存申请的空间大小要求大于等于核数*32Bytes，并且缓存的值需要初始化为0。目前常见的有两种初始化方式：
      - 通过在host侧进行初始化操作，确保传入该接口时，gmWorkspace缓存已经初始化为0；
      - 在kernel侧初始化的时候对gmWorkspace缓存初始化，需要注意的是，每个核上都需要初始化全部的gmWorkspace缓存空间。
    - ubWorkspace申请的空间大小要求大于等于核数*32Bytes。
    - 使用该接口进行多核控制时，算子调用时指定的逻辑blockNum必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。
    - 在分离模式下，建议使用硬同步接口而非软同步接口。软同步接口仅适用于纯Vector场景，且性能较低。使用硬同步接口时，需根据场景设置Kernel类型：
      - 在纯Vector/Cube场景下，需设置Kernel类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0。
      - 对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。
    """

    py_example = """
    **调用示例**

    - 软同步

        .. code-block:: python

            gm = asc.GlobalTensor()
            gm.set_global_buffer(x)
            ub = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=32)
            asc.sync_all(gm, ub, used_cores=0)

    - 硬同步

        .. code-block:: python

            asc.sync_all()
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def cross_core_set_flag_docstring():
    func_introduction = """
    面向分离架构的核间同步控制接口。
    该接口和cross_core_wait_flag接口配合使用。使用时需传入核间同步的标记ID(flagId)，每个ID对应一个初始值为0的计数器。执行cross_core_set_flag后ID对应的计数器增加1；执行cross_core_wait_flag时如果对应的计数器数值为0则阻塞不执行；如果对应的计数器大于0，则计数器减一，同时后续指令开始执行。
    同步控制分为以下几种模式：
    - 模式0：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到cross_core_set_flag时，cross_core_wait_flag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到cross_core_set_flag时，cross_core_wait_flag后续的指令才会执行。
    - 模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了cross_core_set_flag，cross_core_wait_flag后续的指令才会执行。
    - 模式2：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行cross_core_set_flag之后，两个AIV上cross_core_wait_flag后续的指令才会继续执行；两个AIV都执行cross_core_set_flag后，AIC上cross_core_wait_flag后续的指令才能执行。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <uint8_t modeId, pipe_t pipe>
        __aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
    """

    param_list = """
    **参数说明**

    - modeId: 核间同步的模式，取值如下：
      - 模式0：AI Core核间的同步控制。
      - 模式1：AI Core内部，Vector核（AIV）之间的同步控制。
      - 模式2：AI Core内部，Cube核（AIC）与Vector核（AIV）之间的同步控制。
    - pipe: 设置这条指令所在的流水类型。
    - flagId: 核间同步的标记，取值范围是0-10。
    """

    constraint_list = """
    **约束说明**

    - 使用该同步接口时，需要按照如下规则设置Kernel类型：
      - 在纯Vector/Cube场景下，需设置Kernel类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0。
      - 对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。
    - 因为Matmul高阶API内部实现中使用了本接口进行核间同步控制，所以不建议开发者同时使用该接口和Matmul高阶API，否则会有flagID冲突的风险。
    - 同一flagId的计数器最多设置15次。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.cross_core_set_flag(flag_id=0, mode_id=0, pipe=asc.PipeID.PIPE_V)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def cross_core_wait_flag_docstring():
    func_introduction = """
    面向分离架构的核间同步控制接口。该接口和cross_core_set_flag接口配合使用。具体使用方法请参考cross_core_set_flag。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <uint8_t modeId, pipe_t pipe>
        __aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
    """

    param_list = """
    **参数说明**

    - modeId: 核间同步的模式，取值如下：
      - 模式0：AI Core核间的同步控制。
      - 模式1：AI Core内部，Vector核（AIV）之间的同步控制。
      - 模式2：AI Core内部，Cube核（AIC）与Vector核（AIV）之间的同步控制。
    - pipe: 设置这条指令所在的流水类型。
    - flagId: 核间同步的标记。取值范围是0-10。
    """

    constraint_list = """
    **约束说明**

    - 使用该同步接口时，需要按照如下规则设置Kernel类型：
      - 在纯Vector/Cube场景下，需设置Kernel类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0。
      - 对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。
    - CrossCoreWaitFlag必须与CrossCoreSetFlag接口配合使用，避免计算核一直处于阻塞阶段。
    - 如果执行CrossCoreWaitFlag时该flagId的计数器的值为0，则CrossCoreWaitFlag之后的所有指令都将被阻塞，直到该flagId的计数器的值不为0。同一个flagId的计数器最多设置15次。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.cross_core_wait_flag(flag_id=0, mode_id=0, pipe=asc.PipeID.PIPE_V)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def data_cache_clean_and_invalid_docstring():
    func_introduction = """
    用来刷新Cache，保证Cache与Global Memory之间的数据一致性。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, CacheLine entireType, DcciDst dcciDst>
        __aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)

    .. code-block:: c++

        template <typename T, CacheLine entireType, DcciDst dcciDst>
        __aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dst)

    .. code-block:: c++

        template <typename T, CacheLine entireType>
        __aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)

    """

    param_list = """
    **参数说明**

    - entire_type：指令操作模式，类型为CacheLine枚举值：
      - SINGLE_CACHE_LINE：只刷新传入地址所在的Cache Line（若非64B对齐，仅操作对齐范围内部分）。
      - ENTIRE_DATA_CACHE：刷新整个Data Cache（耗时较大，性能敏感场景慎用）。
    - dcci_dst：指定Data Cache与哪种存储保持一致性，类型为DcciDst枚举类：
      - CACHELINE_ALL：与CACHELINE_OUT效果一致。
      - CACHELINE_UB：预留参数，暂未支持。
      - CACHELINE_OUT：保证Data Cache与Global Memory一致。
      - CACHELINE_ATOMIC：部分Atlas产品上为预留参数，暂未支持。
    - dst：	需要刷新Cache的Tensor。
    """

    py_example = """
    **调用示例**

    - 支持通过配置dcciDst确保Data Cache与GM存储的一致性

      .. code-block:: python

          asc.data_cache_clean_and_invalid(entire_type=asc.CacheLine.SINGLE_CACHE_LINE,
                                          dcci_dst=asc.DcciDst.CACHELINE_OUT, dst=dst)

    - 不支持配置dcciDst，仅支持保证Data Cache与GM的一致性

      .. code-block:: python

          asc.data_cache_clean_and_invalid(entire_type=asc.CacheLine.SINGLE_CACHE_LINE, dst=dst)

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def data_cache_preload_docstring():
    func_introduction = """
    从源地址所在的特定GM地址预加载数据到data cache中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

        .. code-block:: c++

            template <typename T>
            __aicore__ inline void DataCachePreload(const GlobalTensor<uint64_t> &src, const T cacheOffset)
    """

    param_list = """
    **参数说明**

    - src (asc.GlobalTensor): 源操作数，代表一个 Global Memory 地址空间。
    - cache_offset (int): 内存偏移量，表示从 `src` 的基地址开始，偏移 `cache_offset` 个字节的位置开始预加载数据。
    """

    return_value = """
    **返回值**

    无。
    """

    constraint_list = """
    **约束说明**

    频繁调用此接口可能导致保留站拥塞，这种情况下，此指令将被视为NOP指令，阻塞Scalar流水。因此不建议频繁调用该接口。
    """

    py_example = """
    **调用示例**

        .. code-block:: python

            weight_gm = asc.GlobalTensor()
            weight_gm.set_global_buffer(weight_gm_addr)

            asc.data_cache_preload(src=weight_gm, cache_offset=0)

    """

    return [func_introduction, cpp_signature, param_list, return_value, constraint_list, py_example]


def data_copy_docstring():
    func_introduction = """
    DataCopy系列接口提供全面的数据搬运功能，支持多种数据搬运场景，并可在搬运过程中实现随路格式转换和量化激活等操作。
    该接口支持Local Memory与Global Memory之间的数据搬运，以及Local Memory内部的数据搬运。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                        const uint32_t count)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                            const DataCopyParams& repeatParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                        const uint32_t count)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                        const DataCopyParams& repeatParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                        const uint32_t count)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                        const DataCopyParams& repeatParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                        const DataCopyParams& intriParams,
                                        const DataCopyEnhancedParams& enhancedParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                        const DataCopyParams& intriParams,
                                        const DataCopyEnhancedParams& enhancedParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                        const DataCopyParams& intriParams,
                                        const DataCopyEnhancedParams& enhancedParams)

    .. code-block:: c++

        template <typename T, typename U>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src,
                                        const DataCopyParams& intriParams,
                                        const DataCopyEnhancedParams& enhancedParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                        const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[],
                                        const uint32_t dimValue = 1)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                                        const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[],
                                        const uint32_t dimValue = 1)

    """

    param_list = """
    **参数说明**

    - dst: 目的操作数，类型为LocalTensor或GlobalTensor。
    - src：源操作数，类型为LocalTensor或GlobalTensor。
    - params：搬运参数，DataCopyParams类型。
    - count：参与搬运的元素个数。
    - enhanced_params：增强信息参数。
    - slice_list1/slice_list2：目的操作数/源操作数切片信息，SliceInfo类型。
    - dim_value：操作数维度信息，默认值为1。
    """

    constraint_list = """
    **约束说明**

    - 如果需要执行多个data_copy指令，且data_copy的目的地址存在重叠，需要通过调用pipe_barrier(ISASI)来插入同步指令，保证多个data_copy指令的串行化，防止出现异常数据。
    - 在跨卡通信算子开发场景，data_copy类接口支持跨卡数据搬运，仅支持HCCS物理链路，不支持其他通路；开发者开发过程中，需要关注涉及卡间通信的物理通路，可通过npu-smi info -t topo命令查询HCCS物理链路。
    """

    py_example = """
    **调用示例**

    - 基础数据搬运

      .. code-block:: python

          pipe = asc.Tpipe()
          in_queue_src = asc.TQue(asc.TPosition.VECIN, 1)
          out_queue_dst = asc.TQue(asc.TPosition.VECOUT, 1)
          src_global = asc.GlobalTensor()
          dst_global = asc.GlobalTensor()
          pipe.init_buffer(que=in_queue_src, num=1, len=512 * asc.half.sizeof())
          pipe.init_buffer(que=out_queue_dst, num=1,len=512 * asc.half.sizeof())
          src_local = in_queue_src.alloc_tensor(asc.half)
          dst_local = out_queue_dst.alloc_tensor(asc.half)
          # 使用传入count参数的搬运接口，完成连续搬运
          asc.data_copy(src_local, src_global, count=512)
          asc.data_copy(dst_local, src_local, count=512)
          asc.data_copy(dst_global, dst_local, count=512)
          # 使用传入DataCopyParams参数的搬运接口，支持连续和非连续搬运
          intri_params = asc.DataCopyParams()
          asc.data_copy(src_local, src_global, params=intri_params)
          asc.data_copy(dst_local, src_local, params=intri_params)
          asc.data_copy(dst_global, dst_local, params=intri_params)

    - 增强数据搬运

      .. code-block:: python

          pipe = asc.Tpipe()
          in_queue_src = asc.TQue(asc.TPosition.CO1, 1)
          out_queue_dst = asc.TQue(asc.TPosition.CO2, 1)
          ...
          src_local = in_queue_src.alloc_tensor(asc.half)
          dst_local = out_queue_dst.alloc_tensor(asc.half)
          intri_params = asc.DataCopyParams()
          enhanced_params = asc.DataCopyEnhancedParams()
          asc.data_copy(dst_local, src_local, params=intri_params, enhanced_params=enhanced_params)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def data_copy_pad_docstring():
    func_introduction = """
    DataCopyPad接口提供数据非对齐搬运的功能，其中从Global Memory搬运数据至Local Memory时，可以根据开发者的需要自行填充数据。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    通路：Global Memory->Local Memory

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopyPad(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                            const DataCopyExtParams &dataCopyParams, const DataCopyPadExtParams<T> &padParams)

    通路：Local Memory->Global Memory

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopyPad(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                                            const DataCopyExtParams &dataCopyParams)

    通路：Local Memory->Local Memory，实际搬运过程是VECIN/VECOUT->GM->TSCM

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopyPad(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                            const DataCopyExtParams &dataCopyParams, const Nd2NzParams &nd2nzParams)

    通路：Global Memory->Local Memory (DataCopyParams版本)

    .. code-block:: c++

        template<typename T>
        __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                            const DataCopyParams& dataCopyParams, const DataCopyPadParams& padParams)

    通路：Local Memory->Global Memory (DataCopyParams版本)

    .. code-block:: c++

        template<typename T>
        __aicore__ inline void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                            const DataCopyParams& dataCopyParams)

    通路：Local Memory->Local Memory，实际搬运过程是VECIN/VECOUT->GM->TSCM (DataCopyParams版本)

    .. code-block:: c++

        template<typename T>
        __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                            const DataCopyParams& dataCopyParams, const Nd2NzParams& nd2nzParams)
    """

    param_list = """
    **参数说明**

    - dst: 目的操作数，类型为LocalTensor或GlobalTensor。
      LocalTensor的起始地址需要保证32字节对齐。
      GlobalTensor的起始地址无地址对齐约束。

    - src: 源操作数，类型为LocalTensor或GlobalTensor。
      LocalTensor的起始地址需要保证32字节对齐。
      GlobalTensor的起始地址无地址对齐约束。

    - dataCopyParams: 搬运参数。
      DataCopyExtParams类型：支持更大的操作数步长等参数取值范围
      DataCopyParams类型：标准搬运参数

    - padParams: 从Global Memory搬运数据至Local Memory时，用于控制数据填充过程的参数。
      DataCopyPadExtParams<T>类型：支持泛型填充值
      DataCopyPadParams类型：仅支持uint64_t数据类型且填充值只能为0

    - nd2nzParams: 从VECIN/VECOUT->TSCM进行数据搬运时，用于控制数据格式转换的参数。
      Nd2NzParams类型，ndNum仅支持设置为1。
    """

    constraint_list = """
    **约束说明**

    - leftPadding、rightPadding的字节数均不能超过32Bytes。
    - 当数据类型长度为64位时，paddingValue只能设置为0。
    - 不同产品型号对函数原型的支持存在差异，请参考官方文档选择产品型号支持的函数原型进行开发。
    """

    py_example = """
    **调用示例**

    GM->VECIN搬运数据并填充：

    .. code-block:: python

        # 从GM->VECIN搬运，使用DataCopyParams和DataCopyPadParams
        src_local = in_queue_src.alloc_tensor(asc.half)
        copy_params = asc.DataCopyParams(1, 20 * asc.half.sizeof(), 0, 0)
        pad_params = asc.DataCopyPadParams(True, 0, 2, 0)
        asc.data_copy_pad(src_local, src_global, copy_params, pad_params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def dump_acc_chk_point_docstring():
    func_introduction = """
    基于算子工程开发的算子，可以使用该接口 Dump 指定 Tensor 的内容。
    同时支持打印自定义的附加信息（仅支持 uint32_t 类型的信息），
    例如用于打印当前执行位置、行号等调试信息。
    与 dump_tensor 不同的是，该接口支持指定 Tensor 的偏移位置进行 Dump，适用于精细化调试和问题定位。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DumpAccChkPoint(
            const LocalTensor<T>& tensor,
            uint32_t index,
            uint32_t countOff,
            uint32_t dumpSize);

        template <typename T>
        __aicore__ inline void DumpAccChkPoint(
            const GlobalTensor<T>& tensor,
            uint32_t index,
            uint32_t countOff,
            uint32_t dumpSize);
    """

    param_list = """
    **参数说明**

    - tensor：
      需要 Dump 的 Tensor，支持 LocalTensor 和 GlobalTensor。

    - index：
      Dump 检查点索引编号，用于区分不同 Dump 位置。

    - count_off：
      自定义附加信息，仅支持 uint32_t 类型，通常用于记录行号、
      步骤编号等调试信息。

    - dump_size：
      Dump 的元素个数，从 Tensor 指定偏移位置开始连续 Dump。
    """

    constraint_list = """
    **约束说明**

    - 该接口主要用于调试和问题定位，建议仅在 Debug 场景下使用。
    - 附加信息 count_off 仅支持 uint32_t 类型。
    - Dump 行为可能影响性能，不建议在性能敏感路径中频繁调用。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        x_local = asc.LocalTensor(
            dtype=asc.float16,
            pos=asc.TPosition.VECIN,
            addr=0,
            tile_size=512,
        )
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)

        asc.dump_acc_chk_point(tensor=x_local, index=0, count_off=1, dump_size=5)
        asc.dump_acc_chk_point(tensor=x_gm, index=0, count_off=1, dump_size=5)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def dump_tensor_docstring():
    func_introduction = """
    基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DumpTensor(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize)
        template <typename T>
        __aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc, uint32_t dumpSize)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DumpTensor(const LocalTensor<T>& tensor, uint32_t desc,
        uint32_t dumpSize, const ShapeInfo& shapeInfo)
        template <typename T>
        __aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc,
        uint32_t dumpSize, const ShapeInfo& shapeInfo)

    """

    param_list = """
    **参数说明**

    - tensor：需要dump的Tensor。
    - desc：用户自定义附加信息（行号或其他自定义数字）。
    - dump_size：需要dump的元素个数。
    - shape_info：传入Tensor的shape信息，可按照shape信息进行打印。
    """

    constraint_list = """
    **约束说明**

    - 该功能仅用于NPU上板调试，且仅在如下场景支持：
      - 通过Kernel直调方式调用算子。
      - 通过单算子API调用方式调用算子。
      - 间接调用单算子API(aclnnxxx)接口：Pytorch框架单算子直调的场景。
    - 当前仅支持打印存储位置为Unified Buffer/L1 Buffer/L0C Buffer/Global Memory的Tensor信息。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量（包括信息头）不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
    """

    py_example = """
    **调用示例**

    - 无Tensor shape的打印

      .. code-block:: python

          asc.dump_tensor(src_local, 5, date_len)

    - 带Tensor shape的打印

      .. code-block:: python

          shape_info = asc.ShapeInfo()
          asc.dump_tensor(x, 2, 64, shape_info)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def metrics_prof_start_docstring():
    func_introduction = """
    用于设置性能数据采集信号启动，和asc.metrics_prof_stop()配合使用。
    使用msProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用asc.metrics_prof_start()和asc.metrics_prof_stop()来指定需要调优的代码段范围。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void MetricsProfStart()

    """

    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.metrics_prof_start()

    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def metrics_prof_stop_docstring():
    func_introduction = """
    设置性能数据采集信号停止，和asc.metrics_prof_start()配合使用。
    使用msProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用asc.metrics_prof_start()和asc.metrics_prof_stop()来指定需要调优的代码段范围。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void MetricsProfStop()

    """

    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.metrics_prof_stop()

    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def printf_docstring():
    func_introduction = """
    该接口提供CPU域/NPU域调试场景下的格式化输出功能。
    在算子kernel侧实现代码中需要输出日志信息的地方调用printf接口打印相关内容。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        void printf(__gm__ const char* fmt, Args&&... args)
        void PRINTF(__gm__ const char* fmt, Args&&... args)

    """

    param_list = """
    **参数说明**

    - fmt：格式控制字符串，包含两种类型的对象：普通字符和转换说明。
    - args：附加参数，个数和类型可变的参数列表。
    """

    constraint_list = """
    **约束说明**

    - 本接口不支持打印除换行符之外的其他转义字符。
    - 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
    - 算子入图场景，若一个动态Shape模型中有可下沉的部分，框架内部会将模型拆分为动态调度和下沉调度（静态子图）两部分，静态子图中的算子不支持该printf特性。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        #整型打印
        x = 10
        asc.printf("%d", x)
        #浮点型打印
        x = 3.14
        asc.printf("%f", x)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def print_time_stamp_docstring():
    func_introduction = """
    提供时间戳打点功能，用于在算子Kernel代码中标记关键执行点。调用后会打印如下信息：

    - desc_id： 用户自定义标识符，用于区分不同打点位置；
    - rsv ：保留值，默认为0，无需关注；
    - time_stamp ： 当前系统cycle数，用于计算时间差，时间换算规则可参考GetSystemCycle(ISASI)；
    - pc_ptr：pc指针数值，若无特殊需求，用户无需关注。
    - entry：预留字段，用户无需关注。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void PrintTimeStamp(uint32_t descId)
    """

    param_list = """
    **参数说明**

    - desc_id：用户自定义标识符（自定义数字），用于区分不同打点位置。[0, 0xffff]是预留给Ascend C内部各个模块使用的id值，用户自定义的desc_id建议使用大于0xffff的数值。
    """

    constraint_list = """
    **约束说明**

    - 该功能仅用于NPU上板调试。
    - 暂不支持算子入图场景的打印。
    - 单次调用本接口打印的数据总量不可超过1MB（还包括少量框架需要的头尾信息，通常可忽略）。使用时应注意，如果超出这个限制，则数据不会被打印。在使用自定义算子工程进行工程化算子开发时，一个算子所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1MB。请开发者自行控制待打印的内容数据量，超出则不会打印。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.print_time_stamp(65577)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_block_num_docstring():
    func_introduction = """
    获取当前任务配置的核数，用于代码内部的多核逻辑控制等。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline int64_t GetBlockNum()

    """

    param_list = """
    **参数说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        loop_size = total_size // asc.get_block_num()

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def get_block_idx_docstring():
    func_introduction = """
    获取当前核的index，用于代码内部的多核逻辑控制及多核偏移量计算等。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline int64_t GetBlockIdx()

    """

    param_list = """
    **参数说明**

    无。
    """

    constraint_list = """
    **约束说明**

    GetBlockIdx为一个系统内置函数，返回当前核的index。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        src0_global.set_global_buffer(src0_gm + asc.get_block_idx() * single_core_offset)
        src1_global.set_global_buffer(src1_gm + asc.get_block_idx() * single_core_offset)
        dst_global.set_global_buffer(dst_gm + asc.get_block_idx() * single_core_offset)
        pipe.init_buffer(que=in_queue_src0, num=1, len=256*asc.float.sizeof())
        pipe.init_buffer(que=in_queue_src1, num=1, len=256*asc.float.sizeof())
        pipe.init_buffer(que=sel_mask, num=1, len=256)
        pipe.init_buffer(que=out_queue_dst, num=1, len=256*asc.float.sizeof())

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_data_block_size_in_bytes_docstring():
    func_introduction = """
    获取当前芯片版本一个data_block的大小，单位为byte。
    开发者可以根据data_block的大小来计算API指令中待传入的repeat_time、
    data_block、stride、repeat_stride等参数值。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline constexpr int16_t GetDataBlockSizeInBytes()
    """

    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

    当前芯片版本一个data_block的大小，单位为byte。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        size = asc.get_data_block_size_in_bytes()

    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def get_icache_preload_status_docstring():
    func_introduction = """
    获取ICACHE的PreLoad的状态。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline int64_t GetICachePreloadStatus();
    """

    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

    int64_t类型，0表示空闲，1表示忙。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        cache_preload_status = asc.get_icache_preload_status()
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def get_mrg_sort_result_docstring():
    func_introduction = """
    获取mrg_sort已经处理过的队列里的Region Proposal个数，并依次存储在四个出参中。

    本接口和mrg_sort相关指令的配合关系如下：

    - 配合mrg_sort_4指令使用，获取mrg_sort_4指令处理过的队列里的Region Proposal个数。使用时，需要将mrg_sort_4中的mrg_sort_4_info.if_exhausted_suspension参数配置为TTrue，该配置模式下某条队列耗尽后，mrg_sort_4指令即停止。
      以上说明适用于如下型号：

      - Atlas 推理系列产品AI Core

    - 配合mrg_sort指令使用，获取mrg_sort指令处理过的队列里的Region Proposal个数。使用时，需要将mrg_sort中的mrg_sort_4_info.if_exhausted_suspension参数配置为True，该配置模式下某条队列耗尽后，mrg_sort指令即停止。
      以上说明适用于如下型号：

      - Atlas A3 训练系列产品/Atlas A3 推理系列产品
      - Atlas A2 训练系列产品/Atlas A2 推理系列产品
      - Atlas 200I/500 A2 推理产品

    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void GetMrgSortResult(uint16_t &mrgSortList1, uint16_t &mrgSortList2, uint16_t &mrgSortList3, uint16_t &mrgSortList4)


    """

    param_list = """
    **参数说明**

    无。

    """

    return_list = """
    **返回值说明**

    - mrg_sort_list1（第一个返回值）：类型为uint16_t，表示mrg_sort第一个队列里已经处理过的Region Proposal个数。
    - mrg_sort_list2（第二个返回值）：类型为uint16_t，表示mrg_sort第二个队列里已经处理过的Region Proposal个数。
    - mrg_sort_list3（第三个返回值）：类型为uint16_t，表示mrg_sort第三个队列里已经处理过的Region Proposal个数。
    - mrg_sort_list4（第四个返回值）：类型为uint16_t，表示mrg_sort第四个队列里已经处理过的Region Proposal个数。
    """

    constraint_list = """
    **约束说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        src1 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        src2 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=512, tile_size=512)
        src3 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=1024, tile_size=512)
        src4 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=1536, tile_size=512)
        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=2048)
        element_count_list = [128, 128, 128, 128]
        sorted_num = [0, 0, 0, 0]
        asc.mrg_sort(dst, sort_list, element_count_list, sorted_num, valid_bit=15, repeat_time=1)
        asc.mrg_sort(dst, sort_list, element_count_list, sorted_num, valid_bit=15, repeat_time=1, is_exhausted_suspension=True)
        mrg_sort4_info = asc.MrgSort4Info(element_count_list, if_exhausted_suspension=False, valid_bit=7, repeat_times=1)
        asc.mrg_sort(dst, sort_list, mrg_sort4_info)

        mrg1, mrg2, mrg3, mrg4 = asc.get_mrg_sort_result()

    """

    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def get_program_counter_docstring():
    func_introduction = """
    获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

         __aicore__ inline int64_t GetProgramCounter()

    """

    param_list = """
    **参数说明**

    无。
    """
    py_example = """
    **调用示例**

    .. code-block:: python

        pc = asc.get_program_counter()

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def get_sort_len_docstring():
    func_introduction = """
    Get sort length in the sorted structure.
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

       namespace AscendC {

       template <typename T>
       __aicore__ inline uint32_t GetSortLen(const uint32_t elemCount);

       } // namespace AscendC
    """

    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

        int: The sort length.
    """

    constraint_list = """
    **约束说明**

        - elem_count must be a 32-bit unsigned integer.
    """

    py_example = """
    **调用示例**

        .. code-block:: python

           length = asc.get_sort_len(100)
    """
    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def get_sort_offset_docstring():
    func_introduction = """
    Get sort offset in the sorted structure.
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

       namespace AscendC {

       template <typename T>
       __aicore__ inline uint32_t GetSortOffset(const uint32_t elemOffset);

       } // namespace AscendC
    """

    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

        int: The sort offset.
    """

    constraint_list = """
    **约束说明**

        - elem_offset must be a 32-bit unsigned integer.
    """

    py_example = """
    **调用示例**

        .. code-block:: python

           offset = asc.get_sort_offset(10)
    """
    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def get_sub_block_idx_docstring():
    func_introduction = """
    获取 AI Core 上 Vector 核的 ID。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline int64_t GetSubBlockIdx();
    """

    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

    返回 Vector 核 ID。
    """

    constraint_list = """
    **约束说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        sub_block_id = asc.get_sub_block_idx()
    """

    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def get_system_cycle_docstring():
    func_introduction = """
    获取当前系统cycle数，若换算成时间需要按照50MHz的频率，时间单位为us，换算公式为：time = (cycle数/50) us 。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline int64_t GetSystemCycle()

    """

    param_list = """
    **参数说明**

    无。
    """

    constraint_list = """
    **约束说明**

    该接口是PIPE_S流水，若需要测试其他流水的指令时间，需要在调用该接口前通过pipe_barrier插入对应流水的同步
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        cycle = asc.get_system_cycle()

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_task_ratio_docstring():
    func_introduction = """
    分离模式下，获取一个AI Core上Cube Core（AIC）或者Vector Core（AIV）的数量与AI Core数量的比例。耦合模式下，固定返回1。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline int64_t GetTaskRatio();
    """

    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

    针对分离模式，不同Kernel类型下（通过设置Kernel类型设置），在AIC和AIV上调用该接口的返回值如下：

    表1 返回值列表
    .. list-table::
       :header-rows: 1

       * - Kernel 类型
         - KERNEL_TYPE_AIV_ONLY
         - KERNEL_TYPE_AIC_ONLY
         - KERNEL_TYPE_MIX_AIC_1_2
         - KERNEL_TYPE_MIX_AIC_1_1
         - KERNEL_TYPE_MIX_AIC_1_0
         - KERNEL_TYPE_MIX_AIV_1_0
       * - AIV
         - 1
         - －
         - 2
         - 1
         - －
         - 1
       * - AIC
         - －
         - 1
         - 1
         - 1
         - 1
         - －

    针对耦合模式，固定返回 1。
    """

    constraint_list = """
    **约束说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        ratio = asc.get_task_ratio()
    """

    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def icache_preload_docstring():
    func_introduction = """
    从指令所在DDR地址预加载指令到ICache中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void ICachePreLoad(const int64_t preFetchLen);
    """

    param_list = """
    **参数说明**

    - pre_fetch_len：预取长度。
    """

    return_list = """
    **返回值说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        pre_fetch_len = 2
        asc.icache_preload(pre_fetch_len)
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def init_const_value_docstring():
    func_introduction = """
    将特定TPosition的LocalTensor初始化为某一具体数值。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U = PrimT<T>, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
        __aicore__ inline void InitConstValue(const LocalTensor<T> &dst, const InitConstValueParams<U> &initConstValueParams)

    """

    param_list = """
    **参数说明**

    - dst：目的操作数，类型为 LocalTensor，支持的 TPosition 为 A1/A2/B1/B2。如果TPosition为A1/B1，起始地址需要满足32B对齐；如果TPosition为A2/B2，起始地址需要满足512B对齐。

    - init_const_value_params：初始化相关参数，类型为InitConstValueParams。仅支持配置迭代次数（repeat_times）和初始化值（init_value）场景下，其他参数配置无效。每次迭代处理固定数据量（512字节），迭代间无间隔。支持配置所有参数场景下，支持配置迭代次数（repeat_times）、初始化值（init_value）、每个迭代处理的数据块个数（block_num）和迭代间间隔（dst_gap）。

      - repeat_times：迭代次数。默认值为0。

        - 仅支持配置迭代次数（repeat_times）和初始化值（init_value）场景下，repeat_times∈[0, 255]。
        - 支持配置所有参数场景下，repeat_times∈[0, 32767]。

      - block_num：每次迭代初始化的数据块个数，取值范围：block_num∈[0, 32767] 。默认值为0。

        - dst的位置为A1/B1时，每一个block（数据块）大小是32B；
        - dst的位置为A2/B2时，每一个block（数据块）大小是512B。

      - dst_gap：目的操作数前一个迭代结束地址到后一个迭代起始地址之间的距离。取值范围：dst_gap∈[0, 32767] 。默认值为0。

        - dst的位置为A1/B1时，单位是32B；
        - dst的位置为A2/B2时，单位是512B。

      - init_value：初始化的value值，支持的数据类型与dst保持一致。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        dst = asc.LocalTensor(dtype=asc.float32, pos=asc.TPosition.A1, addr=0, tile_size=128)
        params = asc.InitConstValueParams(repeat_times=1, block_num=2, dst_gap=0, init_value=2.2)
        asc.init_const_value(dst=dst, init_const_value_params=params)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def load_data_docstring():
    func_introduction = """
    源操作数/目的操作数的数据类型为uint8_t/int8_t时，分形矩阵大小在A1/A2上为16*32， 在B1/B2上为32*16。
    源操作数/目的操作数的数据类型为uint16_t/int16_t/half/bfloat16_t时，分形矩阵在A1/B1/A2/B2上的大小为16*16。
    源操作数/目的操作数的数据类型为uint32_t/int32_t/float时，分形矩阵大小在A1/A2上为16*8， 在B1/B2上为8*16。
    支持如下数据通路：
    GM->A1; GM->B1; GM->A2; GM->B2;
    A1->A2; B1->B2。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const LocalTensor<T>& src,
                                        const LoadData2DParams& loadDataParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const GlobalTensor<T>& src,
                                        const LoadData2DParams& loadDataParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const LocalTensor<T>& src,
                                        const LoadData2DParamsV2& loadDataParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const GlobalTensor<T>& src,
                                        const LoadData2DParamsV2& loadDataParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const LocalTensor<T>& src,
                                        const LoadData3DParamsV2Pro& loadDataParams)

    .. code-block:: c++

        template <typename T,
                const IsResetLoad3dConfig &defaultConfig = IS_RESER_LOAD3D_DEFAULT_CONFIG,
                typename U = PrimT<T>,
                typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const LocalTensor<T>& src,
                                        const LoadData3DParamsV1<U>& loadDataParams)

    .. code-block:: c++

        template <typename T,
                const IsResetLoad3dConfig &defaultConfig = IS_RESER_LOAD3D_DEFAULT_CONFIG,
                typename U = PrimT<T>,
                typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const LocalTensor<T>& src,
                                        const LoadData3DParamsV2<U>& loadDataParams)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数，类型为 LocalTensor。
      - 作为二维数据加载的目标 Tensor。
      - 支持的 TPosition 为 VECIN/VECCALC/VECOUT。
      - 起始地址需要 32 字节对齐。

    - src：源操作数，类型为 LocalTensor 或 GlobalTensor。
      - 当为 LocalTensor 时，表示在芯片内部不同本地存储单元之间按 2D 方式搬运。
      - 当为 GlobalTensor 时，表示从 Global Memory 按 2D 方式加载数据到 LocalTensor。
      - 元素数据类型需与 dst 保持一致。

    - params：类型为下面结构体
      - LoadData2DParams 结构体
        - start_index：分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：start_index∈[0, 65535] 。单位：512B。默认为0。
        - repeat_times：迭代次数，每个迭代可以处理512B数据。取值范围：repeat_times∈[1, 255]。
        - src_stride：相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B。取值范围：src_stride∈[0, 65535]。默认为0。
        - sid：预留参数，配置为0即可。
        - dst_gap：相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B。取值范围：dst_gap∈[0, 65535]。默认为0。
        - if_transpose：是否启用转置功能，对每个分形矩阵进行转置，默认为false。
        - addr_mode：预留参数，配置为0即可。

      - LoadData2DParamsV2 结构体
        - m_start_position：M维起始位置，取值范围：m_start_position∈[0, 65535]。默认为0。
        - k_start_position：K维起始位置，取值范围：k_start_position∈[0, 65535]。默认为0。
        - m_step：M维步长，取值范围：m_step∈[0, 65535]。默认为0。
        - k_step：K维步长，取值范围：k_step∈[0, 65535]。默认为0。
        - src_stride：源操作数步长，取值范围：src_stride∈[-2147483648, 2147483647]。默认为0。
        - dst_stride：目的操作数步长，取值范围：dst_stride∈[0, 65535]。默认为0。
        - if_transpose：是否启用转置功能，默认为false。
        - sid：流ID，取值范围：sid∈[0, 255]。默认为0。

      - LoadData3DParamsV2Pro 结构体
        - channel_size：通道大小，取值范围：channel_size∈[0, 65535]。默认为0。
        - en_transpose：是否启用转置功能，默认为false。
        - en_small_k：是否启用小K优化，默认为false。
        - filter_size_w：是否启用滤波器宽度优化，默认为false。
        - filter_size_h：是否启用滤波器高度优化，默认为false。
        - f_matrix_ctrl：是否启用矩阵控制，默认为false。
        - ext_config：扩展配置，取值范围：ext_config∈[0, 18446744073709551615]。默认为0。
        - filter_config：滤波器配置，取值范围：filter_config∈[0, 18446744073709551615]。默认为0x10101010101。

      - LoadData3DParamsV1 结构体
        - pad_list：padding列表，顺序为[padding_left, padding_right, padding_top, padding_bottom]，每个元素取值范围：[0, 255]。
        - l1_h：源操作数height，取值范围：[1, 32767]。
        - l1_w：源操作数width，取值范围：[1, 32767]。
        - c1_index：卷积窗口在源Tensor C1维度的起点，取值范围：[0, 4095]。
        - fetch_filter_w：卷积窗口在filter W维度的起始位置，取值范围：[0, 254]。
        - fetch_filter_h：卷积窗口在filter H维度的起始位置，取值范围：[0, 254]。
        - left_top_w：卷积窗口在源Tensor W维度的起点，取值范围：[-255, 32767]。
        - left_top_h：卷积窗口在源Tensor H维度的起点，取值范围：[-255, 32767]。
        - stride_w：卷积核在W维的滑动步长，取值范围：[1, 63]。
        - stride_h：卷积核在H维的滑动步长，取值范围：[1, 63]。
        - filter_w：卷积核width，取值范围：[1, 255]。
        - filter_h：卷积核height，取值范围：[1, 255]。
        - dilation_filter_w：卷积核W维膨胀系数，取值范围：[1, 255]。
        - dilation_filter_h：卷积核H维膨胀系数，取值范围：[1, 255]。
        - jump_stride：迭代之间目的操作数地址递增步长，取值范围：[1, 127]。
        - repeat_mode：迭代模式，取值范围：[0, 1]，默认为0。
        - repeat_time：迭代次数，取值范围：[1, 255]。
        - c_size：通道展开优化控制参数，取值范围：[0, 1]，默认为0。
        - pad_value：padding填充值，需与src数据类型一致，默认为0。

      - LoadData3DParamsV2 结构体
        - pad_list：padding列表，顺序为[padding_left, padding_right, padding_top, padding_bottom]，每个元素取值范围：[0, 255]。
        - l1_h：源操作数height，取值范围：[1, 32767]。
        - l1_w：源操作数width，取值范围：[1, 32767]。
        - channel_size：通道大小，不同数据类型与平台存在对齐约束。
        - k_extension：K维扩展长度，取值范围：[1, 65535]。
        - m_extension：M维扩展长度，取值范围：[1, 65535]。
        - k_start_pt：K维起始位置，取值范围：[0, 65535]。
        - m_start_pt：M维起始位置，取值范围：[0, 65535]。
        - stride_w：卷积核在W维的滑动步长，取值范围：[1, 63]。
        - stride_h：卷积核在H维的滑动步长，取值范围：[1, 63]。
        - filter_w：卷积核width，取值范围：[1, 255]。
        - filter_h：卷积核height，取值范围：[1, 255]。
        - dilation_filter_w：卷积核W维膨胀系数，取值范围：[1, 255]。
        - dilation_filter_h：卷积核H维膨胀系数，取值范围：[1, 255]。
        - en_transpose：是否启用转置功能，取值为bool，默认为false。
        - pad_value：padding填充值，需与src数据类型一致，默认为0。
        - filter_size_w：是否在filterW基础上增加256元素，默认为false。
        - filter_size_h：是否在filterH基础上增加256元素，默认为false。
        - f_matrix_ctrl：FeatureMap矩阵控制开关，默认为false。
      """

    constraint_list = """
    **约束说明**

    - dst 与 src 的数据需要满足起始地址对齐要求，具体可查看文档。
    - 不使用或者不想改变的配置，建议保持默认值，有助于性能提升。

    """

    py_example = """
    **调用示例**

    - Local Memory 内部 2D 搬运（Local -> Local）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_l2l(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)
              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              params = asc.LoadData2DParams(0, 4, 0, 0, 0, 0, 0)

              asc.load_data(y_local, x_local, params)

    - Global Memory 到 Local Memory 的 2D 搬运（Global -> Local）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_g2l(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)
              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              x_gm = asc.GlobalTensor()
              x_gm.set_global_buffer(x)

              params = asc.LoadData2DParams(0, 4, 0, 0, 0, 0, 0)

              asc.load_data(y_local, x_local, params)
              asc.load_data(x_local, x_gm, params)

    - Local Memory 内部 2D 搬运（V2版本，Local -> Local）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_l2l_v2(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)
              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              params_v2 = asc.LoadData2DParamsV2(0, 0, 16, 16, 0, 0, False, 0)

              asc.load_data(y_local, x_local, params_v2)

    - Global Memory 到 Local Memory 的 2D 搬运（V2版本，Global -> Local）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_g2l_v2(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)
              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              x_gm = asc.GlobalTensor()
              x_gm.set_global_buffer(x)

              params_v2 = asc.LoadData2DParamsV2(0, 0, 16, 16, 0, 0, False, 0)

              asc.load_data(y_local, x_local, params_v2)
              asc.load_data(x_local, x_gm, params_v2)

    - Local Memory 内部 3D 搬运（V2Pro版本，Local -> Local）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_3d_v2pro(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)
              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              params_3d_v2_pro = asc.LoadData3DParamsV2Pro(16, False, False, False, False, False, 0, 0x10101010101)

              asc.load_data(y_local, x_local, params_3d_v2_pro)

    - Local Memory 内部 3D 搬运（LoadData3DParamsV1）

      .. code-block:: python

          def test_load_data_v1(mock_launcher_run):

            @asc.jit
            def kernel_load_data_v1(x: asc.GlobalAddress) -> None:
                x_local = asc.LocalTensor(
                    dtype=asc.float16,
                    pos=asc.TPosition.VECIN,
                    addr=0,
                    tile_size=512,
                )
                y_local = asc.LocalTensor(
                    dtype=asc.float16,
                    pos=asc.TPosition.VECOUT,
                    addr=0,
                    tile_size=512,
                )
                x_gm = asc.GlobalTensor()
                x_gm.set_global_buffer(x)

                params_3d_v1 = asc.LoadData3DParamsV1(
                    [0, 0, 0, 0],
                    16, 16,
                    0,
                    0, 0,
                    0, 0,
                    1, 1,
                    3, 3,
                    1, 1,
                    1,
                    0, 1,
                    0,
                    0,
                )

                asc.load_data(y_local, x_local, params_3d_v1)

            x = MockTensor(asc.float16)
            kernel_load_data_v1[1](x)
            assert mock_launcher_run.call_count == 1

    - Local Memory 内部 3D 搬运（LoadData3DParamsV2）

      .. code-block:: python

          def test_load_data_v2(mock_launcher_run):

            @asc.jit
            def kernel_load_data_v2(x: asc.GlobalAddress) -> None:
            x_local = asc.LocalTensor(
                    dtype=asc.float16,
                    pos=asc.TPosition.VECIN,
                    addr=0,
                    tile_size=512,
                )
                y_local = asc.LocalTensor(
                    dtype=asc.float16,
                    pos=asc.TPosition.VECOUT,
                    addr=0,
                    tile_size=512,
                )
                x_gm = asc.GlobalTensor()
                x_gm.set_global_buffer(x)

                params_3d_v2 = asc.LoadData3DParamsV2(
                    [0, 0, 0, 0],
                    16, 16,
                    16,
                    16, 16,
                    0, 0,
                    1, 1,
                    3, 3,
                    1, 1,
                    False,
                    0,
                    False, False,
                    False,
                )

                asc.load_data(y_local, x_local, params_3d_v2)

            x = MockTensor(asc.float16)
            kernel_load_data_v2[1](x)
            assert mock_launcher_run.call_count == 1
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def load_data_with_transpose_docstring():
    func_introduction = """
    该接口实现带转置的2D格式数据从A1/B1到A2/B2的加载。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadDataWithTranspose(const LocalTensor<T>& dst,
                                                     const LocalTensor<T>& src,
                                                     const LoadData2dTransposeParams& loadDataParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadDataWithTranspose(const LocalTensor<T>& dst,
                                                     const LocalTensor<T>& src,
                                                     const LoadData2dTransposeParamsV2& loadDataParams)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数，类型为 LocalTensor。
      - 用于接收转置后的二维数据。
      - 存储位置需属于 VECIN / VECCALC / VECOUT 中的一种。
      - 起始地址需满足 32 字节对齐要求。

    - src：源操作数，类型为 LocalTensor。
      - 作为 2D 输入块的提供者。
      - 仅支持 Local → Local（A1/B1 → A2/B2），不支持 GlobalTensor。
      - 数据类型必须与 dst 一致。

    - params：二维转置加载参数，类型为 LoadData2dTransposeParams 或 LoadData2dTransposeParamsV2。
      - LoadData2dTransposeParams 结构体
        - start_index：方块矩阵ID，搬运起始位置为源操作数中第几个方块矩阵（0 为源操作数中第1个方块矩阵）。取值范围：start_index∈[0, 65535] 。默认为0。
        - repeat_times：迭代次数，取值范围：repeat_times∈[0, 255]。默认为0。
        - src_stride：相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔。这里的单位实际上是拼接后的方块矩阵的大小。取值范围：src_stride∈[0, 65535]。默认为0。
        - dst_gap：相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔，单位：512B。取值范围：dst_gap∈[0, 65535]。默认为0。
        - dst_frac_gap：每个迭代内目的操作数转置前一个分形结束地址与后一个分形起始地址的间隔，单位：512B，仅在数据类型为float/int32_t/uint32_t/uint8_t/int8_t/int4b_t时有效。取值范围：dst_frac_gap∈[0, 65535]。默认为0。
        - addr_mode：预留参数

      - LoadData2dTransposeParamsV2 结构体
        - start_index：方块矩阵ID，搬运起始位置为源操作数中第几个方块矩阵（0 为源操作数中第1个方块矩阵）。取值范围：start_index∈[0, 65535] 。默认为0。
        - repeat_times：迭代次数，取值范围：repeat_times∈[0, 255]。默认为0。
        - src_stride：源操作数步长，取值范围：src_stride∈[0, 65535]。默认为0。
        - dst_gap：目的操作数间隔，取值范围：dst_gap∈[0, 65535]。默认为0。
        - dst_frac_gap：分形间隔，取值范围：dst_frac_gap∈[0, 65535]。默认为0。
        - src_frac_gap：源分形间隔，取值范围：src_frac_gap∈[0, 65535]。默认为0。
        - addr_mode：地址模式，取值范围：addr_mode∈[0, 255]。默认为0。
    """

    constraint_list = """
    **约束说明**

    - repeatTimes 为 0 时表示不执行搬运操作。
    - 开发者需要保证目的操作数转置后的分形没有重叠。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。

    - repeat_times 为 0 时表示不执行搬运操作。
    - 开发者需要保证目的操作数转置后的分形没有重叠。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = """
    **调用示例**

    - 调用示例（V1版本）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_with_transpose(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)

              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              params = asc.LoadData2dTransposeParams(0, 4, 0, 0, 0, 0)

              asc.load_data_with_transpose(y_local, x_local, params)

    - 调用示例（V2版本）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_with_transpose_v2(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                      pos=asc.TPosition.VECIN,
                                      addr=0, tile_size=512)

              y_local = asc.LocalTensor(dtype=asc.float16,
                                      pos=asc.TPosition.VECOUT,
                                      addr=0, tile_size=512)

              params_v2 = asc.LoadData2dTransposeParamsV2(0, 4, 0, 0, 0, 0, 0)

              asc.load_data_with_transpose(y_local, x_local, params_v2)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def mmad_docstring():
    func_introduction = """
    完成矩阵乘加（C += A * B）操作。矩阵ABC分别为A2/B2/CO1中的数据。
    ABC矩阵的数据排布格式分别为ZZ，ZN，NZ。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <typename T, typename U, typename S>
        __aicore__ inline void Mmad(const LocalTensor<T>& dst,
                                    const LocalTensor<U>& fm,
                                    const LocalTensor<S>& filter,
                                    const MmadParams& mmadParams)

    .. code-block:: c++

        template <typename T, typename U, typename S, typename V>
        __aicore__ inline void Mmad(const LocalTensor<T>& dst,
                                    const LocalTensor<U>& fm,
                                    const LocalTensor<S>& filter,
                                    const LocalTensor<V>& bias,
                                    const MmadParams& mmadParams)
    """

    param_list = """
    **参数说明**

    - dst：结果输出 Tensor，类型为 LocalTensor。
      - 用于存放矩阵乘累加的结果。
      - **必须位于 CO1 存储位置（TPosition.CO1）**。
      - 元素数据类型需与累加结果类型匹配。

    - fm：左矩阵（A 矩阵）输入，类型为 LocalTensor。
      - 表示矩阵乘法中的左操作数。
      - **必须位于 A2 存储位置（TPosition.A2）**。
      - 需要按照满足 Mmad 格式要求的 A2 布局存储。

    - filter：右矩阵（B 矩阵）输入，类型为 LocalTensor。
      - 表示矩阵乘法中的右操作数。
      - **必须位于 B2 存储位置（TPosition.B2）**。
      - 需要按照符合指令格式的 B2 分块布局排布。

    - bias（可选）：偏置项，类型为 LocalTensor。
      - 用于执行 `dst += fm × filter + bias` 的计算。
      - 当提供 bias 时，将使用带偏置版本的指令。

    - params：MmadParams 类型的矩阵乘参数。
      - m：左矩阵Height，取值范围：m∈[0, 4095] 。默认值为0。
      - n：右矩阵Width，取值范围：n∈[0, 4095] 。默认值为0。
      - k：左矩阵Width、右矩阵Height，取值范围：k∈[0, 4095] 。默认值为0。
      - cmatrixInitVal：配置C矩阵初始值是否为0。默认值true。
      - cmatrixSource：配置C矩阵初始值是否来源于C2（存放Bias的硬件缓存区）。默认值为false。
      - isBias：该参数废弃，新开发内容不要使用该参数。
      - fmOffset：预留参数。
      - enSsparse：预留参数。
      - enWinogradA：预留参数。
      - enWinogradB：预留参数。
      - unitFlag：预留参数。
      - kDirectionAlign：预留参数。
    """

    constraint_list = """
    **约束说明**

    - dst只支持位于CO1，fm只支持位于A2，filter只支持位于B2。
    - 当M、K、N中的任意一个值为0时，该指令不会被执行。
    - 当M = 1时，会默认开启GEMV（General Matrix-Vector Multiplication）功能。在这种情况下，Mmad API从L0A Buffer读取数据时，会以ND格式进行读取，而不会将其视为ZZ格式。所以此时左矩阵需要直接按照ND格式进行排布。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = """
    **调用示例**

    - 基本 Mmad（无 bias）

      .. code-block:: python

          @asc.jit
          def kernel_mmad_basic():
              dst = asc.LocalTensor(dtype=asc.float16,
                                    pos=asc.TPosition.CO1,
                                    addr=0, tile_size=1024)

              fm = asc.LocalTensor(dtype=asc.float16,
                                   pos=asc.TPosition.A2,
                                   addr=0, tile_size=1024)

              filter = asc.LocalTensor(dtype=asc.float16,
                                       pos=asc.TPosition.B2,
                                       addr=0, tile_size=1024)

              params = asc.MmadParams(4, 4, 4)

              asc.mmad(dst, fm, filter, params)

    - Mmad 带 bias

      .. code-block:: python

          @asc.jit
          def kernel_mmad_bias():
              dst = asc.LocalTensor(dtype=asc.float16,
                                    pos=asc.TPosition.CO1,
                                    addr=0, tile_size=1024)

              fm = asc.LocalTensor(dtype=asc.float16,
                                   pos=asc.TPosition.A2,
                                   addr=0, tile_size=1024)

              filter = asc.LocalTensor(dtype=asc.float16,
                                       pos=asc.TPosition.B2,
                                       addr=0, tile_size=1024)

              bias = asc.LocalTensor(dtype=asc.float16,
                                     pos=asc.TPosition.VECIN,
                                     addr=0, tile_size=1024)

              params = asc.MmadParams(4, 4, 4)

              asc.mmad(dst, fm, filter, bias, params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def load_image_to_local_docstring():
    func_introduction = """
    将图像数据从GM搬运到A1/B1。 搬运过程中可以完成图像预处理操作：包括图像翻转，改变图像尺寸（抠图，裁边，缩放，伸展），以及色域转换，类型转换等。
    图像预处理的相关参数通过set_aipp_functions进行配置。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadImageToLocal(const LocalTensor<T>& dst, const LoadImageToLocalParams& loadDataParams)
    """

    param_list = """
    **参数说明**

    - dst：输出，目的操作数，类型为LocalTensor，支持的TPosition为A1/B1。LocalTensor的起始地址需要保证32字节对齐。不同产品支持的数据类型不同：
      - Atlas A3 训练/推理系列产品：int8_t/half
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：int8_t/half
      - Atlas 200I/500 A2 推理产品：uint8_t/int8_t/half
    - load_data_params：输入，LoadData参数结构体，类型为LoadImageToLocalParams，包含以下子参数：
      - horiz_size：输入，从源图中加载图片的水平宽度，单位为像素，取值范围：horiz_size∈[2, 4095]。
      - vert_size：输入，从源图中加载图片的垂直高度，单位为像素，取值范围：vert_size∈[2, 4095]。
      - horiz_start_pos：输入，加载图片在源图片上的水平起始地址，单位为像素，取值范围：horiz_start_pos∈[0, 4095]，默认为0。注意：当输入图片为YUV420SP、XRGB8888、RGB888和YUV400格式时，该参数需要是偶数。
      - vert_start_pos：输入，加载图片在源图片上的垂直起始地址，单位为像素，取值范围：vert_start_pos∈[0, 4095]，默认为0。注意：当输入图片为YUV420SP格式时，该参数需要是偶数。
      - src_horiz_size：输入，源图像水平宽度，单位为像素，取值范围：src_horiz_size∈[2, 4095]。注意：当输入图片为YUV420SP格式时，该参数需要是偶数。
      - top_pad_size：输入，目的图像顶部填充的像素数，取值范围：top_pad_size∈[0, 32]，默认为0。进行数据填充时使用，需要先调用SetAippFunctions(ISASI)通过AippPaddingParams配置填充的数值，再通过topPadSize、botPadSize、leftPadSize、rightPadSize配置填充的大小范围。
      - bot_pad_size：输入，目的图像底部填充的像素数，取值范围：bot_pad_size∈[0, 32]，默认为0。
      - left_pad_size：输入，目的图像左边填充的像素数，取值范围：left_pad_size∈[0, 32]，默认为0。
      - right_pad_size：输入，目的图像右边填充的像素数，取值范围：right_pad_size∈[0, 32]，默认为0。
      - sid：输入，预留参数，为后续功能保留，开发者暂时无需关注，使用默认值即可。
    """

    return_list = """
    **返回值说明**

    无
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 加载到dst的图片的大小加padding的大小必须小于等于L1的大小。
    - 对于XRGB输入格式的数据，芯片在处理的时候会默认丢弃掉第四个通道的数据，所以需要在set_aipp_functions接口里设置好通道交换的参数后输出RGB格式的数据。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.A1, addr=0, tile_size=128)
        load_data_params = asc.LoadImageToLocalParams(2, 2, 0, 0, 2, 0, 0, 0, 0)
        asc.load_image_to_local(dst, load_data_params)
    """
    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def proposal_concat_docstring():
    func_introduction = """
    将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposals的对应位置里。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void ProposalConcat(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t modeNumber)

    """

    param_list = """
    **参数说明**

    - dst：目的操作数。
    - src：源操作数。数据类型需要与dst保持一致。
    - repeat_time：重复迭代次数。每次迭代完成16个元素合入到16个Region Proposals里，下次迭代跳至相邻的下一组16个Region Proposals和下一组16个元素。取值范围：repeatTime∈[0,255]。

    - mode_number：合入位置参数，取值范围：modeNumber∈[0, 5]
      - 0：合入x1
      - 1：合入y1
      - 2：合入x2
      - 3：合入y2
      - 4：合入score
      - 5：合入label
    """

    constraint_list = """
    **约束说明**

    - 用户需保证dst中存储的proposal数目大于等于实际所需数目，否则会存在tensor越界错误。
    - 用户需保证src中存储的元素大于等于实际所需数目，否则会存在tensor越界错误。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.proposal_concat(dst, src, repeat_time=2, mode_number=4)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def proposal_extract_docstring():
    func_introduction = """
    与proposal_concat功能相反，从Region Proposals内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void ProposalExtract(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t modeNumber)

    """

    param_list = """
    **参数说明**

    - dst：目的操作数。
    - src：源操作数，数据类型需与dst一致。
    - repeat_time：重复迭代次数。每次迭代处理16个Region Proposals的元素抽取并重排，下次迭代跳至相邻的下一组16个Region Proposals。取值范围：repeatTime∈[0,255]。
    - mode_number：抽取位置参数，取值范围：modeNumber∈[0,5]
      - 0：抽取x1
      - 1：抽取y1
      - 2：抽取x2
      - 3：抽取y2
      - 4：抽取score
      - 5：抽取label
    """

    constraint_list = """
    **约束说明**

    - 用户需保证src中存储的proposal数量不小于实际所需数量，否则可能发生tensor越界。
    - 用户需保证dst中可容纳的元素数量不小于实际抽取数量。
    - 操作数地址需满足通用对齐约束（32字节对齐）。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.proposal_extract(dst, src, repeat_time=2, mode_number=4)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def trap_docstring():
    func_introduction = """
    在Kernel侧调用，NPU模式下会中断AI Core的运行，CPU模式下等同于assert。可用于Kernel侧异常场景的调试。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void Trap()

    """

    param_list = """
    **参数说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.trap()

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def duplicate_docstring():
    func_introduction = """
    将一个变量或立即数复制多次并填充到向量中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count)

        template <typename T, bool isSetMask = true>
        void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)

        template <typename T, bool isSetMask = true>
        void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)

    """

    param_list = """
    **参数说明**

    - dst：目的操作数。
    - scalar：被复制的源操作数，支持输入变量和立即数，数据类型需与dst中元素的数据类型保持一致。
    - count：参与计算的元素个数。
    - mask：mask用于控制每次迭代内参与计算的元素。
    - repeat_time：矢量计算单元，每次读取连续的8个datablock（每个block32Bytes，共256Bytes）数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeat_time表示迭代的次数。
    - dst_block_stride：单次迭代内，矢量目的操作数不同datablock间地址步长。
    - dst_repeat_stride：相邻迭代间，矢量目的操作数相同datablock地址步长。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          scalar = 18.0
          asc.duplicate(dst_local, scalar, mask=mask, repeat_times=2, dst_block_stride=1, dst_repeat_stride=8)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          scalar = 18.0
          asc.duplicate(dst_local, scalar, mask=mask, repeat_times=2, dst_block_stride=1, dst_repeat_stride=8)

    - tensor前n个数据计算样例，源操作数为标量

      .. code-block:: python

          scalar = 18.0
          asc.duplicate(dst_local, scalar, count=src_data_size)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def mmad_with_sparse_docstring():
    func_introduction = """
    完成矩阵乘加操作，传入的左矩阵A为稀疏矩阵， 右矩阵B为稠密矩阵 。
    对于矩阵A，在mmad_with_sparse计算时完成稠密化；
    对于矩阵B，在计算执行前的输入数据准备时自行完成稠密化（按照下文中介绍的稠密算法进行稠密化），
    所以输入本接口的B矩阵为稠密矩阵。B稠密矩阵需要通过调用load_data_with_sparse载入，同时加载索引矩阵，
    索引矩阵在矩阵B稠密化的过程中生成，再用于A矩阵的稠密化。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T = int32_t, typename U = int8_t, typename Std::enable_if<Std::is_same<PrimT<T>, int32_t>::value, bool>::type = true, typename Std::enable_if<Std::is_same<PrimT<U>, int8_t>::value, bool>::type = true>
        __aicore__ inline void MmadWithSparse(const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<U>& filter, const MmadParams& mmadParams)

    """

    param_list = """
    **参数说明**

    - dst：输出，目的操作数，结果矩阵，类型为LocalTensor，支持的TPosition为CO1。起始地址需要256个元素（1024字节）对齐。

    - fm：输入，源操作数，左矩阵A，类型为LocalTensor，支持的TPosition为A2。LocalTensor的起始地址需要512字节对齐。

    - filter：输入，源操作数，右矩阵B，类型为LocalTensor，支持的TPosition为B2。LocalTensor的起始地址需要512字节对齐。

    - mmad_params：输入，矩阵乘相关参数，类型为MmadParams。

      - m：左矩阵Height，取值范围：m∈[0, 4095] 。默认值为0。
      - n：右矩阵Width，取值范围：n∈[0, 4095] 。默认值为0。
      - k：左矩阵Width、右矩阵Height，取值范围：k∈[0, 4095] 。默认值为0。
      - cmatrix_init_val：配置C矩阵初始值是否为0。默认值true。

        - true：C矩阵初始值为0；
        - false：C矩阵初始值通过cmatrix_source参数进行配置。

      - cmatrix_source：配置C矩阵初始值是否来源于C2（存放Bias的硬件缓存区）。默认值为false。

        - false：来源于CO1；
        - true：来源于C2。

      - is_bias：该参数废弃，新开发内容不要使用该参数。如果需要累加初始矩阵，请使用带bias的接口来实现；也可以通过cmatrix_init_val和cmatrix_source参数配置C矩阵的初始值来源来实现。推荐使用带bias的接口，相比于配置cmatrix_init_val和cmatrix_source参数更加简单方便。配置是否需要累加初始矩阵，默认值为false，取值说明如下：

        - false：矩阵乘，无需累加初始矩阵，C = A * B。
        - true：矩阵乘加，需要累加初始矩阵，C += A * B。

      - fm_offset：预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。
      - en_ssparse：预留参数。
      - en_winograd_a：预留参数。
      - en_winograd_b：预留参数。
      - unit_flag：预留参数。
      - k_direction_align：预留参数。
    """

    constraint_list = """
    **约束说明**

    - 原始稀疏矩阵B每4个元素中应保证最多2个非零元素，如果存在3个或更多非零元素，则仅使用前2个非零元素。
    - 当M、K、N中的任意一个值为0时，该指令不会被执行。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        dst = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.CO1, addr=0, tile_size=400)
        fm = asc.LocalTensor(dtype=asc.int8, pos=asc.TPosition.A2, addr=0, tile_size=400)
        filter = asc.LocalTensor(dtype=asc.int8, pos=asc.TPosition.B2, addr=0, tile_size=400)
        mmad_params = asc.MmadParams(
            m=20,
            n=20,
            k=20,
            is_bias=False,
            fm_offset=0,
            en_ssparse=False,
            en_winograd_a=False,
            en_winograd_b=False
        )
        asc.mmad_with_sparse(dst, fm, filter, mmad_params)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def pair_reduce_sum_docstring():
    func_introduction = """
    PairReduceSum：相邻两个（奇偶）元素求和。例如，对于序列 (a1, a2, a3, a4, a5, a6, ...)，
    相邻两个数据求和为 (a1+a2, a3+a4, a5+a6, ...)。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // mask 逐bit模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const uint64_t mask[],
                                            const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);

    .. code-block:: c++

        // mask 连续模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask,
                                            const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);
    """

    param_list = """
    **参数说明**

    - dst：输出操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。LocalTensor 的起始地址需要 32 字节对齐。

    - src：输入操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。LocalTensor 的起始地址需要 32 字节对齐。

    - repeat_time：迭代次数，取值范围 [0, 255]。关于该参数的具体描述请参考如何使用 Tensor 高维切分计算 API。

    - mask：控制每次迭代内参与计算的元素。

      **逐比特模式**
      数组形式，按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。
      - 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
      - 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
      - 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]

      **连续模式**
      整数形式，表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。
      - 16 位：mask ∈ [1, 128]
      - 32 位：mask ∈ [1, 64]
      - 64 位：mask ∈ [1, 32]

    - dst_rep_stride：
      目的操作数相邻迭代间的地址步长，以一个 repeat 归约后的长度为单位。
      - PairReduce 完成后，一个 repeat 的长度减半。
      - 注意：Atlas 训练系列产品不支持配置 0。

    - src_blk_stride：单次迭代内数据 block 的地址步长，详细说明请参考 data_block_stride。

    - src_rep_stride：源操作数相邻迭代间的地址步长，即每次迭代跳过的 data block 数目。详细说明请参考 repeatStride。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 如果两两相加的两个元素mask位未配置（即当前两个元素不参与运算）。

      - 对于 Atlas 200I/500 A2 推理产品 ，对应的目的操作数中的值会置为0。
      - 对于其他产品型号，对应的目的操作数中的值不会变化。

    - 比如float场景下对64个数使用当前指令，mask配置为62，表示最后两个元素不参与运算。

      - 对于 Atlas 200I/500 A2 推理产品 ，目的操作数中最后一个值会返回0。
      - 对于其他产品型号，目的操作数中最后一个值不会变化。
    """

    py_example = """
    **调用示例**

    - mask 连续模式

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.pair_reduce_sum(z_local, x_local, repeat_time=2, mask=128,
                              dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 逐bit模式

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.pair_reduce_sum(z_local, x_local, repeat_time=2, mask=mask,
                              dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def repeat_reduce_sum_docstring():
    func_introduction = """
    对每个 repeat 内的所有数据进行求和。
    与 whole_reduce_sum 接口相比，不支持 mask 逐比特模式。
    建议使用功能更全面的 whole_reduce_sum 接口。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void RepeatReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask,
                                                const int32_t dstBlkStride, const int32_t srcBlkStride, const int32_t dstRepStride, const int32_t srcRepStride);
    """

    param_list = """
    **参数说明**

    - dst：
      输出操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 起始地址需保证 2 字节对齐（half 数据类型）或 4 字节对齐（float 数据类型）。
      - 数据类型根据产品支持情况：half / float。

    - src：
      输入操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 起始地址需 32 字节对齐。
      - 数据类型需与 dst 保持一致。

    - repeat_time：
      重复迭代次数，取值范围 [0, 255]。
      - 矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代才能完成所有数据的读取与计算。repeatTime表示迭代的次数。
      - 具体描述请参考 Tensor 高维切分计算 API。

    - mask：
      控制每次迭代内连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。
      - 操作数为 16 位：mask ∈ [1, 128]
      - 操作数为 32 位：mask ∈ [1, 64]

    - dst_blk_stride：此参数无效，可配置任意值。

    - src_blk_stride：单次迭代内数据 data_block 的地址步长。详细说明请参考 data_block_stride。

    - dst_rep_stride：
      目的操作数相邻迭代间的地址步长，以一个 repeat 归约后的长度为单位。
      - 单位为 dst 数据类型所占字节长度。比如当dst为half时，单位为2Bytes。
      - 注意：Atlas 训练系列产品不支持配置 0。

    - src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的data_block数目。详细说明请参考 repeat_stride。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    - 对于RepeatReduceSum，其内部的相加方式采用二叉树方式，两两相加
      假设源操作数为128个half类型的数据[data0,data1,data2...data127]，一个repeat可以计算完，计算过程如下。
      1. data0和data1相加得到data00，data2和data3相加得到data01...data124和data125相加得到data62，data126和data127相加得到data63；
      2. data00和data01相加得到data000，data02和data03相加得到data001...data62和data63相加得到data031；
      3. 以此类推，得到目的操作数为1个half类型的数据[data]。
      4. 需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。例如源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，第二步计算-30000+100=-29900，第四步计算65504-29900=35604。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.repeat_reduce_sum(z_local, x_local, repeat_time=4, mask=128,
                              dst_blk_stride=0, src_blk_stride=1, dst_rep_stride=8, src_rep_stride=8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def whole_reduce_max_docstring():
    func_introduction = """
    每个repeat内所有数据求最大值以及其索引index，返回的索引值为每个repeat内部索引。
    归约指令的总体介绍请参考如何使用归约指令。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // mask 逐bit模式
        template <typename T, bool isSetMask = true>
         __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask[], const int32_t repeatTime,
                                                const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
                                                ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);


    .. code-block:: c++

        // mask 连续模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t mask, const int32_t repeatTime,
                                                const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
                                                ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);
    """

    param_list = """
    **参数说明**

    - dst：
      输出，目的操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 的起始地址需 4 字节对齐（half 数据类型）或 8 字节对齐（float 数据类型）。
      - 数据类型根据产品支持情况：half / float。

    - src：
      输入，源操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 起始地址需 32 字节对齐。
      - 源操作数的数据类型需要与目的操作数保持一致。

    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]

          - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
          - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]

    - repeat_time：迭代次数，取值范围 [0, 255]。具体描述请参考 如何使用Tensor 高维切分计算API。

    - dst_rep_stride：
      目的操作数相邻迭代间地址步长。
      - 以一个 repeat 归约后的长度为单位。
      - 返回索引和最值时，单位为 dst 数据类型所占字节长度的两倍，比如当dst为half时，单位为4Bytes。
      - 仅返回最值时，单位为 dst 数据类型所占字节长度。
      - 仅返回索引时，单位为 uint32_t 类型所占字节长度。
      - 注意：Atlas 训练系列产品不支持配置 0。

    - src_blk_stride：单次迭代内数据 block 的地址步长。详细说明请参考 data_block_stride。

    - src_rep_stride：源操作数相邻迭代间地址步长，即每次迭代跳过的 data_block 数目。详细说明请参考 repeat_stride。

    - order
      可选参数，指定 dst 中 index 与 value 的相对位置以及返回结果行为，类型为 ReduceOrder。
      - 默认值为 asc.ReduceOrder.ORDER_VALUE_INDEX。
      - asc.ReduceOrder.ORDER_VALUE_INDEX：value 位于低半部，返回顺序 [value, index]。
      - asc.ReduceOrder.ORDER_INDEX_VALUE：index 位于低半部，返回顺序 [index, value]。
      - asc.ReduceOrder.ORDER_ONLY_VALUE：只返回最值，顺序 [value]。
      - asc.ReduceOrder.ORDER_ONLY_INDEX：只返回索引，顺序 [index]。
      - 910B，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。
      - 910C，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    - dst结果存储顺序由order决定，默认为最值、最值索引。返回结果中索引index数据按照dst的数据类型进行存储，比如dst使用half类型时，index按照half类型进行存储，读取时需要使用reinterpret_cast方法转换到整数类型。若输入数据类型是half，需要使用reinterpret_cast<uint16_t*>，若输入是float，需要使用reinterpret_cast<uint32_t*>。比如完整样例中，前两个计算结果为[9.980e-01 5.364e-06]，5.364e-06需要使用reinterpret_cast方法转换得到索引值90。
    - 针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考选择低延迟指令，优化归约操作性能，具体样例请参考ReduceCustom。
    """

    py_example = """
    **调用示例**

    - mask 连续模式，默认 ORDER_VALUE_INDEX

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.whole_reduce_max(z_local, x_local, mask=128, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 连续模式，ORDER_INDEX_VALUE

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.whole_reduce_max(z_local, x_local, mask=128, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8,
                               order=asc.ReduceOrder.ORDER_INDEX_VALUE)

    - mask 逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.whole_reduce_max(z_local, x_local, mask=mask, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 逐bit模式，ORDER_INDEX_VALUE

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.whole_reduce_max(z_local, x_local, mask=mask, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8,
                               order=asc.ReduceOrder.ORDER_INDEX_VALUE)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def whole_reduce_min_docstring():
    func_introduction = """
    每个repeat内所有数据求最小值以及其索引index，返回的索引值为每个repeat内部索引。
    归约指令的总体介绍请参考如何使用归约指令。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // mask 逐bit模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask[], const int32_t repeatTime,
                                            const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
                                            ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);

    .. code-block:: c++

        // mask 连续模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t mask, const int32_t repeatTime,
                                            const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
                                            ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);
    """

    param_list = """
    **参数说明**

    - dst：
      输出，目的操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 的起始地址需 4 字节对齐（half 数据类型）或 8 字节对齐（float 数据类型）。
      - 数据类型根据产品支持情况：half / float。

    - src：
      输入，源操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 起始地址需 32 字节对齐。
      - 源操作数的数据类型需要与目的操作数保持一致。

    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**

        mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]

    - repeat_time：迭代次数，取值范围 [0, 255]。具体描述请参考 如何使用Tensor 高维切分计算API。

    - dst_rep_stride：
      目的操作数相邻迭代间地址步长。
      - 以一个 repeat 归约后的长度为单位。
      - 返回索引和最值时，单位为 dst 数据类型所占字节长度的两倍，比如当dst为half时，单位为4Bytes。
      - 仅返回最值时，单位为 dst 数据类型所占字节长度。
      - 仅返回索引时，单位为 uint32_t 类型所占字节长度。
      - 注意：Atlas 训练系列产品不支持配置 0。

    - src_blk_stride：单次迭代内数据 block 的地址步长。详细说明请参考 data_block_stride。

    - src_rep_stride：源操作数相邻迭代间地址步长，即每次迭代跳过的 data_block 数目。详细说明请参考 repeat_stride。

    - order
      可选参数，指定 dst 中 index 与 value 的相对位置以及返回结果行为，类型为 ReduceOrder。
      - 默认值为 asc.ReduceOrder.ORDER_VALUE_INDEX。
      - asc.ReduceOrder.ORDER_VALUE_INDEX：value 位于低半部，返回顺序 [value, index]。
      - asc.ReduceOrder.ORDER_INDEX_VALUE：index 位于低半部，返回顺序 [index, value]。
      - asc.ReduceOrder.ORDER_ONLY_VALUE：只返回最值，顺序 [value]。
      - asc.ReduceOrder.ORDER_ONLY_INDEX：只返回索引，顺序 [index]。
      - 910B，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。
      - 910C，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    - dst结果存储顺序由order决定，默认为最值、最值索引。返回结果中索引index数据按照dst的数据类型进行存储，比如dst使用half类型时，index按照half类型进行存储，读取时需要使用reinterpret_cast方法转换到整数类型。若输入数据类型是half，需要使用reinterpret_cast<uint16_t*>，若输入是float，需要使用reinterpret_cast<uint32_t*>。比如完整样例中，前两个计算结果为[9.980e-01 5.364e-06]，5.364e-06需要使用reinterpret_cast方法转换得到索引值90。
    - 针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考选择低延迟指令，优化归约操作性能，具体样例请参考ReduceCustom。
    """

    py_example = """
    **调用示例**

    - mask 连续模式，默认 ORDER_VALUE_INDEX

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.whole_reduce_min(z_local, x_local, mask=128, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 连续模式，ORDER_INDEX_VALUE

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.whole_reduce_min(z_local, x_local, mask=128, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8,
                               order=asc.ReduceOrder.ORDER_INDEX_VALUE)

    - mask 逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.whole_reduce_min(z_local, x_local, mask=mask, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 逐bit模式，ORDER_INDEX_VALUE

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.whole_reduce_min(z_local, x_local, mask=mask, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8,
                               order=asc.ReduceOrder.ORDER_INDEX_VALUE)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def whole_reduce_sum_docstring():
    func_introduction = """
    每个迭代内所有数据求和。归约指令的总体介绍请参考如何使用归约指令。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // mask 逐比特模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const uint64_t mask[],
                                              const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);

    .. code-block:: c++

        // mask 连续模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask,
                                              const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);
    """

    param_list = """
    **参数说明**

    - dst：
      输出，目的操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 的起始地址需保证 2 字节对齐（half 数据类型）或 4 字节对齐（float 数据类型）。
      - 数据类型根据产品支持情况：half / float。

    - src：
      输入，源操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 起始地址需 32 字节对齐。
      - 源操作数的数据类型需要与目的操作数保持一致。

    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]

    - repeat_time：迭代次数，取值范围 [0, 255]。具体描述请参考 如何使用Tensor 高维切分计算API。

    - dst_rep_stride：
      目的操作数相邻迭代间地址步长，以一个 repeat 归约后的长度为单位。
      - 单位为 dst 数据类型所占字节长度。比如当dst为half时，单位为2Bytes。
      - 注意：Atlas 训练系列产品不支持配置 0。

    - src_blk_stride：单次迭代内datablock的地址步长。详细说明请参考data_block_stride。

    - src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的data_block数目。详细说明请参考repeat_stride。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    - 对于whole_reduce_sum，其内部的相加方式采用二叉树方式，两两相加
      假设源操作数为128个half类型的数据[data0,data1,data2...data127]，一个repeat可以计算完，计算过程如下。
      1. data0和data1相加得到data00，data2和data3相加得到data01...data124和data125相加得到data62，data126和data127相加得到data63；
      2. data00和data01相加得到data000，data02和data03相加得到data001...data62和data63相加得到data031；
      3. 以此类推，得到目的操作数为1个half类型的数据[data]。
      4. 需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。例如源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，第二步计算-30000+100=-29900，第四步计算65504-29900=35604。
    """

    py_example = """
    **调用示例**

    - mask 连续模式

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.whole_reduce_sum(z_local, x_local, mask=128, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.whole_reduce_sum(z_local, x_local, mask=mask, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def load_data_with_sparse_docstring():
    func_introduction = """
    用于搬运存放在B1里的512B的稠密权重矩阵到B2里，同时读取128B的索引矩阵用于稠密矩阵的稀疏化。
    索引矩阵的数据类型为int2，需要拼成int8的数据类型，再传入接口。
    索引矩阵在一个int8的地址中的排布是逆序排布的，例如：索引矩阵1 2 0 1 0 2 1 0，
    在地址中的排布为1 0 2 1 0 1 2 0，其中1 0 2 1（对应索引矩阵前四位1 2 0 1）为一个int8，0 1 2 0（对应索引矩阵后四位0 2 1 0）为一个int8。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T = int8_t, typename U = uint8_t, typename Std::enable_if<Std::is_same<PrimT<T>, int8_t>::value, bool>::type = true, typename Std::enable_if<Std::is_same<PrimT<U>, uint8_t>::value, bool>::type = true>
        __aicore__ inline void LoadDataWithSparse(const LocalTensor<T> &dst, const LocalTensor<T> &src, const LocalTensor<U> &idx, const LoadData2dParams &loadDataParam)
    """

    param_list = """
    **参数说明**

    - dst：输出，目的操作数，类型为LocalTensor，支持的TPosition为B2，LocalTensor的起始地址需要512字节对齐。支持的数据类型为int8_t。数据连续排列顺序要求为小N大Z格式。

    - src：输入，源操作数，类型为LocalTensor，支持的TPosition为B1，LocalTensor的起始地址需要32字节对齐。支持的数据类型为int8_t。

    - idx：输入，源操作数，类型为LocalTensor，支持的TPosition为B1，LocalTensor的起始地址需要32字节对齐。支持的数据类型为int8_t。

    - load_data_param：输入，LoadData参数结构体，LoadData2DParams类型。

      - start_index：分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：start_index∈[0, 65535] 。单位：512B。默认为0。
      - repeat_times：迭代次数，每个迭代可以处理512B数据。取值范围：repeat_times∈[1, 255]。
      - src_stride：相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B。取值范围：src_stride∈[0, 65535]。默认为0。
      - sid：预留参数，配置为0即可。
      - dst_gap：相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B。取值范围：dst_gap∈[0, 65535]。默认为0。
      - if_transpose：是否启用转置功能，对每个分形矩阵进行转置，默认为false（注意：只有A1->A2和B1->B2通路才能使能转置，使能转置功能时，源操作数、目的操作数仅支持uint16_t/int16_t/half数据类型）:

        - true：启用
        - false：不启用

      - addr_mode：预留参数，配置为0即可。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - repeat_times=0表示不执行。
    - 每次迭代中的start_index不能小于零。
    - 不支持转置功能。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        dst = asc.LocalTensor(dtype=asc.int8, pos=asc.TPosition.B2, addr=0, tile_size=512)
        src = asc.LocalTensor(dtype=asc.int8, pos=asc.TPosition.B1, addr=0, tile_size=512)
        idx = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.B1, addr=0, tile_size=512)
        params = asc.LoadData2DParams(repeat_times=1, src_stride=0, if_transpose=False)
        asc.load_data_with_sparse(dst=dst, src=src, idx=idx, load_data_param=params)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def scatter_docstring():
    func_introduction = """
    给定一个连续的输入张量和一个目的地址偏移张量，scatter指令根据偏移地址生成新的结果张量后将输入张量分散到结果张量中。
    将源操作数src中的元素按照指定的位置（由dst_offset和dst_base共同作用）分散到目的操作数dst中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                      const LocalTensor<uint32_t>& dstOffset,
                                      const uint32_t dstBaseAddr, const uint32_t count)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                        const LocalTensor<uint32_t>& dstOffset,
                                        const uint32_t dstBaseAddr, const uint64_t mask[],
                                        const uint8_t repeatTime, const uint8_t srcRepStride)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                        const LocalTensor<uint32_t>& dstOffset,
                                        const uint32_t dstBaseAddr, const uint64_t mask,
                                        const uint8_t repeatTime, const uint8_t srcRepStride)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数。
    - src：源操作数，数据类型需与dst保持一致。
    - dst_offset：用于存储源操作数的每个元素在dst中对应的地址偏移,以字节为单位。
      偏移基于dst的基地址dst_base计算，以字节为单位，取值应保证按dst数据类型位宽对齐。
    - dst_base：dst的起始偏移地址，单位是字节。取值应保证按dst数据类型位宽对齐。
    - count：执行处理的数据个数。
    - mask：控制每次迭代内参与计算的元素，支持连续模式或逐bit模式。
    - repeat_times：指令迭代次数，每次迭代完成8个datablock的数据收集。
    - src_rep_stride：相邻迭代间的地址步长，单位是datablock。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          asc.scatter(dst, src, dst_offset, dst_base=0, mask=128, repeat_times=1, src_rep_stride=8)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask_bits = [uint64_max, uint64_max]
          asc.scatter(dst, src, dst_offset, dst_base=0, mask=mask_bits, repeat_times=1, src_rep_stride=8)

    - tensor前n个数据计算样例，源操作数为标量

      .. code-block:: python

          asc.scatter(dst, src, dst_offset, dst_base=0, count=128)

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def set_deq_scale_docstring():
    func_introduction = """
    设置DEQSCALE寄存器的值。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetDeqScale(half scale)

        __aicore__ inline void SetDeqScale(float scale, int16_t offset, bool signMode)

        template <typename T>
        __aicore__ inline void SetDeqScale(const LocalTensor<T>& vdeq, const VdeqInfo& vdeqInfo)
    """

    param_list = """
    **参数说明**

    - scale(half)：scale量化参数，half类型。
    - scale(float)：scale量化参数，float类型。
    - offset：offset量化参数，int16_t类型，只有前9位有效。
    - sign_mode：bool类型，表示量化结果是否带符号。
    - vdeq: 输入量化tensor，大小为128Byte。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - vdeqInfo: 存储量化tensor信息的数据结构，结构体内包含量化tensor中的16组量化参数。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # Cast
        scale = 1.0
        asc.set_deq_scale(scale)
        asc.cast(cast_dst_local, cast_dsrc_local, asc.RoundMode.CAST_NONE, src_size)
        # CastDeq
        scale = 1.0
        offset = 0
        sign_mode = True
        asc.set_deq_scale(scale, offset, sign_mode)
        asc.cast_deq(dst_local, src_local, count=src_size, is_vec_deq=False, half_block=False)
        # CastVdeq
        vdeq_local = asc.LocalTensor(dtype=asc.uint64, pos=asc.TPosition.VECIN, addr=0, tile_size=16)
        vdeq_scale = [1.0] * 16
        vdeq_offset = [0] * 16
        vdeq_sign_mode = [False] * 16
        vdeq_info = asc.VdeqInfo(vdeq_scale, vdeq_offset, vdeq_sign_mode)
        asc.set_deq_scale(vdeq_local, vdeq_info)
        asc.cast_deq(dst_local, src_local, count=src_size, is_vec_deq=True, half_block=False)
    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def set_pad_value_docstring():
    func_introduction = """
    设置asc.data_copy_pad需要填充的数值。支持的通路如下：GM->VECIN/GM->VECOUT填充值。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, TPosition pos = TPosition::MAX>
        __aicore__ inline void SetPadValue(T paddingValue)

    """

    param_list = """
    **参数说明**

    - padding_value: 输入，asc.data_copy_pad接口填充的数值，数据与asc.data_copy_pad接口搬运的数据类型一致。
    - pos:

      - 输入，用于指定asc.data_copy_pad接口搬运过程中从GM搬运数据到哪一个目的地址，目的地址通过逻辑位置来表达。
      - 默认值为asc.TPosition.MAX，等效于asc.TPosition.VECIN或asc.TPosition.VECOUT。
    """

    return_list = """
    **返回值说明**

    无。
    """

    constraint_list = """
    **约束说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc

        asc.set_pad_value(37)
        asc.set_pad_value(37, asc.TPosition.VECIN)
    """

    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def set_vector_mask_docstring():
    func_introduction = """
    用于在矢量计算时设置mask。使用前需要先调用 set_mask_count/set_mask_norm 设置 mask 模式。
    在不同模式下，mask的含义不同：

    - **Normal 模式**

      mask参数用来控制单次迭代内参与计算的元素个数。此时又可以划分为如下两种模式：

      - **连续模式（len）**：表示单次迭代中前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数为16位时：mask ∈ [1, 128]
        - 操作数为32位时：mask ∈ [1, 64]
        - 操作数为64位时：mask ∈ [1, 32]

      - **逐比特模式（mask_high / mask_low）**：按位控制参与计算的元素，bit位的值为1表示参与计算，0表示不参与。

        分为mask_high（高位mask）和mask_low（低位mask）。参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数为16位时：mask_low、mask_high ∈ [0, 2⁶⁴-1]，并且不同时为 0
        - 操作数为32位时：mask_high = 0，mask_low ∈ (0, 2⁶⁴-1]
        - 操作数为64位时：mask_high = 0，mask_low ∈ (0, 2³²-1]

    - **Counter 模式**

      mask参数表示整个矢量计算参与计算的元素个数。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, MaskMode mode = MaskMode::NORMAL>
        __aicore__ static inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow);

    .. code-block:: c++

        template <typename T, MaskMode mode = MaskMode::NORMAL>
        __aicore__ static inline void SetVectorMask(int32_t len);
    """

    param_list = """
    **参数说明**

    - mask_high

      - Normal模式：对应Normal模式下的逐比特模式，可以按位控制哪些元素参与计算。传入高位mask值。
      - Counter模式：需要置0，本入参不生效。

    - mask_low

      - Normal模式：对应Normal模式下的逐比特模式，可以按位控制哪些元素参与计算。传入低位mask值。
      - Counter模式：整个矢量计算过程中，参与计算的元素个数。

    - len

      - Normal模式：对应Normal模式下的mask连续模式，表示单次迭代内表示前面连续的多少个元素参与计算。
      - Counter模式：整个矢量计算过程中，参与计算的元素个数。

    - dtype：矢量计算操作数的数据类型，由 Python 前端显式指定，用于推导 C++ 模板参数 T。

    - mode：
      mask 模式，类型为 MaskMode 枚举值
      - asc.MaskMode.NORMAL：Normal 模式，支持连续模式与逐比特模式。
      - asc.MaskMode.COUNTER：Counter 模式，mask 参数表示整个矢量计算参与的总元素个数。

    """

    constraint_list = """
    **约束说明**

    该接口仅在矢量计算API的isSetMask模板参数为false时生效，使用完成后需要使用ResetMask将mask恢复为默认值。
    """

    py_example = """
    **调用示例**

    - Counter 模式：整个计算中参与 128 个元素

      .. code-block:: python

          len = 128
          asc.set_mask_count()
          asc.set_vector_mask(len, dtype=asc.float16, mode=asc.MaskMode.COUNTER)
          asc.reset_mask()

    - Normal 模式（逐bit模式）：使用 bitmask 控制参与计算的元素

      .. code-block:: python

          mask_high = 2**64 - 1
          mask_low = 2**64 - 1
          asc.set_mask_norm()
          asc.set_vector_mask(mask_high, mask_low, dtype=asc.float16, mode=asc.MaskMode.NORMAL)
          asc.reset_mask()

    - Normal 模式（连续模式）：前 64 个元素参与每次迭代计算

      .. code-block:: python

          len = 64
          asc.set_mask_norm()
          asc.set_vector_mask(len, dtype=asc.float32, mode=asc.MaskMode.NORMAL)
          asc.reset_mask()
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def reset_mask_docstring():
    func_introduction = """
    恢复mask的值为默认值（全1），表示矢量计算中每次迭代内的所有元素都将参与运算。

    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void ResetMask()

    """

    param_list = """
    **参数说明**

    无。
    """

    constraint_list = """
    **约束说明**

    无。
    """

    py_example = """
    **调用示例**

      .. code-block:: python

          len = 128
          asc.set_mask_count()
          asc.set_vector_mask(len, dtype=asc.float16, mode=asc.MaskMode.COUNTER)
          asc.reset_mask()

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_hccl_context_docstring():
    func_introduction = """
    获取指定Index通信域的context（消息区）地址。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <uint32_t index>
        __aicore__ inline __gm__ uint8_t* __gm__ GetHcclContext(void)
    """

    param_list = """
    **参数说明**

    - index: 模板参数，用来表示要设置的通信域ID，当前只支持2个通信域，index只能为0/1。
    """

    return_list = """
    **返回值说明**

    指定通信域的context（消息区）地址。
    """

    constraint_list = """
    **约束说明**

    当前最多只支持2个通信域。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        ctx = asc.get_hccl_context(1)
    """

    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def set_hccl_context_docstring():
    func_introduction = """
    设置通算融合算子每个通信域对应的context（消息区）地址。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <uint32_t index>
        __aicore__ inline void SetHcclContext(__gm__ uint8_t* context)
    """

    param_list = """
    **参数说明**

    - index: 模板参数，用来表示要设置的通信域ID，当前只支持2个通信域，index只能为0/1。
    - context: 对应通信域的context（消息区）地址。
    """

    constraint_list = """
    **约束说明**

    当前最多只支持2个通信域。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.set_hccl_context(0, x)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_sys_workspace_docstring():
    func_introduction = """
    获取系统workspace指针。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline __gm__ uint8_t* __gm__ GetSysWorkSpacePtr()

    """

    param_list = """
    **参数说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        workspace = asc.get_sys_workspace()

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def transpose_docstring():
    func_introduction = """
    用于实现16 * 16的二维矩阵数据块转置或者[N,C,H,W]与[N,H,W,C]数据格式互相转换。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // 普通转置，支持16 * 16的二维矩阵数据块进行转置
        template <typename T>
        __aicore__ inline void Transpose(const LocalTensor<T>& dst, const LocalTensor<T>& src)

        // 增强转置，支持16 * 16的二维矩阵数据块转置，支持[N,C,H,W]与[N,H,W,C]互相转换
        template <typename T>
        __aicore__ inline void Transpose(const LocalTensor<T>& dst, const LocalTensor<T> &src,
                                        const LocalTensor<uint8_t> &sharedTmpBuffer,
                                        const TransposeParamsExt &transposeParams)
    """

    param_list = """
    **参数说明**

    - dst: 目的操作数，类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT，起始地址需要32字节对齐
    - src: 源操作数，类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT，起始地址需要32字节对齐，数据类型需要与dst保持一致
    - shared_tmp_buffer: 共享的临时Buffer，大小根据transposeType确定
    - params: 控制Transpose的数据结构，包含输入的shape信息和transposeType参数
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 普通转置接口支持src和dst复用。
    - 增强转置接口，transposeType为TRANSPOSE_ND2ND_B16时支持src和dst复用，transposeType为TRANSPOSE_NCHW2NHWC、TRANSPOSE_NHWC2NCHW时不支持src和dst复用。
    """

    py_example = """
    **调用示例**

    - 基础转置样例

      .. code-block:: python

          pipe = asc.TPipe()
          in_queue_x = asc.TQue(asc.TPosition.VECIN, buffer_num)
          out_queue_z = asc.TQue(asc.TPosition.VECOUT, buffer_num)
          ...
          x_local = in_queue_x.alloc_tensor(asc.float16)
          z_local = out_queue_z.alloc_tensor(asc.float16)
          asc.transpose(z_local, x_local)

    - 增强转置样例

      .. code-block:: python

        pipe = asc.TPipe()
        in_queue_x = asc.TQue(asc.TPosition.VECIN, buffer_num)
        out_queue_z = asc.TQue(asc.TPosition.VECOUT, buffer_num)
        in_queue_tmp = asc.TQue(asc.TPosition.VECIN, buffer_num)
        ...
        x_local = in_queue_x.alloc_tensor(asc.float16)
        z_local = out_queue_z.alloc_tensor(asc.float16)
        tmp_buffer = in_queue_tmp.alloc_tensor(asc.uint8)

        params = asc.TransposeParamsExt(
            n_size=1,
            c_size=16,
            h_size=4,
            w_size=4,
            transpose_type=asc.TransposeType.TRANSPOSE_NCHW2NHWC
        )

        asc.transpose(z_local, x_local, tmp_buffer, params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def trans_data_to_5hd_docstring():
    func_introduction = """
    数据格式转换，一般用于将NCHW格式转换成NC1HWC0格式，也可用于二维矩阵数据块的转置。
    相比于transpose接口，本接口单次repeat内可处理512Byte的数据（16个datablock），
    支持不同shape的矩阵转置，还可以支持多次repeat操作。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // 使用LocalTensor数组版本
        template <typename T>
        __aicore__ inline void TransDataTo5HD(const LocalTensor<T> (&dstList)[NCHW_CONV_ADDR_LIST_SIZE],
                                            const LocalTensor<T> (&srcList)[NCHW_CONV_ADDR_LIST_SIZE],
                                            const TransDataTo5HDParams& nchwconvParams)

        // 使用地址值数组版本（性能更优）
        template<typename T>
        __aicore__ inline void TransDataTo5HD(uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE],
                                            uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE],
                                            const TransDataTo5HDParams& nchwconvParams)

        // 使用连续存储地址值版本
        template <typename T>
        __aicore__ inline void TransDataTo5HD(const LocalTensor<uint64_t>& dst,
                                            const LocalTensor<uint64_t>& src,
                                            const TransDataTo5HDParams& nchwconvParams)
    """

    param_list = """
    **参数说明**

    - dst_or_list：目的操作数地址序列，类型为LocalTensor数组、地址值数组或连续存储地址值的LocalTensor
    - src_or_list：源操作数地址序列，类型与dst_or_list对应，数据类型需要与目的操作数保持一致
    - params：控制参数结构体，包含读取写入位置控制、迭代次数、地址步长等参数
      - dst_high_half：指定数据存储到datablock的高半部还是低半部（仅支持int8_t/uint8_t）
      - src_high_half：指定数据从datablock的高半部还是低半部读取（仅支持int8_t/uint8_t）
      - repeat_times：重复迭代次数，取值范围[0,255]
      - dst_rep_stride：相邻迭代间目的操作数相同datablock地址步长
      - src_rep_stride：相邻迭代间源操作数相同datablock地址步长
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 普通转置接口支持src和dst复用。
    - 增强转置接口，transposeType为TRANSPOSE_ND2ND_B16时支持src和dst复用，transposeType为TRANSPOSE_NCHW2NHWC、TRANSPOSE_NHWC2NCHW时不支持src和dst复用。
    """

    py_example = """
    **调用示例**

    此接口通过不同的方式构造源和目的操作数序列，以实现灵活的数据重组。

    - `dst_list`, `src_list`：定义了源数据块和目标数据块。它们可以是包含 `LocalTensor` 物理地址的 `list`/`tuple`，
      也可以是包含 `LocalTensor` 视图对象的 `list`/`tuple`，或者是将地址值连续存储的 `LocalTensor<uint64_t>`。

      .. code-block:: python

          params = asc.TransDataTo5HDParams(
              dst_high_half=False,
              src_high_half=False,
              repeat_times=4,
              dst_rep_stride=8,
              src_rep_stride=8
          )

          asc.trans_data_to_5hd(dst_list, src_list, params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


NAME_TRANS = {
    "Add": "add",
    "AddDeqRelu": "add_deq_relu",
    "AddRelu": "add_relu",
    "AddReluCast": "add_relu_cast",
    "And": "bitwise_and",
    "Or": "bitwise_or",
    "Div": "div",
    "FusedMulAdd": "fused_mul_add",
    "FusedMulAddRelu": "fused_mul_add_relu",
    "Max": "max",
    "Min": "min",
    "Mul": "mul",
    "MulAddDst": "mul_add_dst",
    "MulCast": "mul_cast",
    "Sub": "sub",
    "SubRelu": "sub_relu",
    "SubReluCast": "sub_relu_cast",
    "Adds": "adds",
    "LeakyRelu": "leaky_relu",
    "Maxs": "maxs",
    "Mins": "mins",
    "Muls": "muls",
    "ShiftLeft": "shift_left",
    "ShiftRight": "shift_right",
    "Abs": "abs",
    "Exp": "exp",
    "Ln": "ln",
    "Not": "bitwise_not",
    "Reciprocal": "reciprocal",
    "Relu": "relu",
    "Rsqrt": "rsqrt",
    "Sqrt": "sqrt",
}


def set_aipp_functions_docstring():
    func_introduction = """
    设置图片预处理（AIPP，AI core pre-process）相关参数。和load_image_to_local(ISASI)接口配合使用。
    设置后，调用load_image_to_local(ISASI)接口可在搬运过程中完成图像预处理操作。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    输入图片格式为YUV400、RGB888、XRGB8888:

    .. code-block:: c++

        template<typename T, typename U>
        void SetAippFunctions(const GlobalTensor<T>& src0, AippInputFormat format, AippParams<U> config)

    输入图片格式为YUV420 Semi-Planar:

    .. code-block:: c++

        template<typename T, typename U>
        void SetAippFunctions(const GlobalTensor<T>& src0, const GlobalTensor<T>& src1, AippInputFormat format, AippParams<U> config)
    """

    param_list = """
    **参数说明**

    - src0：源图片在Global Memory上的矩阵
    - src1：源图片格式为YUV420SP时，表示UV维度在Global Memory上的矩阵
    - input_format：源图片的图片格式
    - config：图片预处理的相关参数，类型为AippParams
    """

    constraint_list = """
    **约束说明**

    - src0、src1在Global Memory上的地址对齐要求如下：
      - YUV420SP：src0必须2Bytes对齐，src1必须2Bytes对齐
      - XRGB8888：src0必须4Bytes对齐
      - RGB888：src0无对齐要求
      - YUV400：src0无对齐要求
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        swap_settings = asc.AippSwapParams(is_swap_rb=True)
        cpad_settings = asc.AippChannelPaddingParams(c_padding_mode=0, c_padding_value=-1)

        aipp_config_int8 = asc.AippParams(
            dtype=asc.int8,
            swap_params=swap_settings,
            c_padding_params=cpad_settings
        )

        asc.set_aipp_functions(rgb_gm, asc.AippInputFormat.RGB888_U8, aipp_config_int8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_binary_docstring(cpp_name: Optional[str] = None, append_text: str = "") -> Callable[[T], T]:
    func_introduction = f"""
    {append_text}
    """

    cpp_signature = f"""
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                            const LocalTensor<T>& src1, const int32_t& count);

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
         __aicore__ inline void {cpp_name}(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                            const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTimes,
                                            const BinaryRepeatParams& repeatParams);

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                            const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTimes,
                                            const BinaryRepeatParams& repeatParams);

    """

    param_list = """
    **参数说明**

    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0, src1：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - count：参与计算的元素个数。
    - mask：用于控制每次迭代内参与计算的元素。
    - repeat_times：重复迭代次数。
    - params：控制操作数地址步长的参数。
    """

    set_mask_param = ""
    if cpp_name != 'MulCast':
        set_mask_param = """
    - is_set_mask: 是否在接口内部设置mask。
        """
    api_name = NAME_TRANS[cpp_name]
    py_example = f"""
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.{api_name}(dst, src0, src1, mask=mask, repeat_times=4, repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.{api_name}(dst, src0, src1, mask=mask, repeat_times=4, repeat_params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.{api_name}(dst, src0, src1, count=512)

    """

    if api_name == 'add_deq_relu':
        py_example = f"""
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          scale = 0.1
          asc.set_deq_scale(scale)
          asc.{api_name}(dst, src0, src1, mask=mask, repeat_times=4, repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          scale = 0.1
          asc.set_deq_scale(scale)
          asc.{api_name}(dst, src0, src1, mask=mask, repeat_times=4, repeat_params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          scale = 0.1
          asc.set_deq_scale(scale)
          asc.{api_name}(dst, src0, src1, count=512)

    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    - 使用整个tensor参与计算接口符号重载时，运算量为目的LocalTensor的总长度。
    """
    docstr = f"""
    {func_introduction}
    {cpp_signature}
    {param_list}
    {set_mask_param}
    {constraint_list}
    {py_example}
    """

    def decorator(fn: T) -> T:
        fn.__doc__ = docstr
        return fn

    return decorator


def set_binary_scalar_docstring(cpp_name: Optional[str] = None, append_text: str = "") -> Callable[[T], T]:
    func_introduction = f"""
    {append_text}
    """

    cpp_signature = f"""
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                            const T& scalarValue, const int32_t& calCount)

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                            const T& scalarValue, uint64_t mask[], const uint8_t repeatTimes,
                                            const UnaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                            const T& scalarValue, uint64_t mask, const uint8_t repeatTimes,
                                            const UnaryRepeatParams& repeatParams)

    """

    param_list = """
    **参数说明**

    - is_set_mask：是否在接口内部设置mask模式和mask值。
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - scalar：源操作数，数据类型需要与目的操作数中的元素类型保持一致。
    - count：参与计算的元素个数。
    - mask：用于控制每次迭代内参与计算的元素。
    - repeat_times：重复迭代次数。
    - params：元素操作控制结构信息。
    """
    api_name = NAME_TRANS[cpp_name]
    py_example = f"""
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          scalar = 2
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.{api_name}(dst, src, scalar, mask=mask, repeat_times=4, repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          scalar = 2
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.{api_name}(dst, src, scalar, mask=mask, repeat_times=4, repeat_params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.{api_name}(dst, src, scalar, count=512)

    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    """

    docstr = f"""
    {func_introduction}
    {cpp_signature}
    {param_list}
    {constraint_list}
    {py_example}
    """

    def decorator(fn: T) -> T:
        fn.__doc__ = docstr
        return fn

    return decorator


def set_hf32_mode_docstring():
    func_introduction = """
    调用该接口后，可设置 Mmad 计算是否开启 HF32 模式。
    开启 HF32 模式后，L0A/L0B 中的 FP32 数据在参与矩阵乘法计算之前将被舍入为 HF32 精度。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline void SetHF32Mode(bool hf32Mode);
    """

    param_list = """
    **参数说明**

    - hf32_mode：
      Mmad HF32 模式控制参数，bool 类型。
      - True：L0A/L0B 中的 FP32 数据将在矩阵乘法之前被舍入为 HF32。
      - False：执行常规的 FP32 矩阵乘法计算。
    """

    constraint_list = """
    **约束说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.set_hf32_mode(True)
        asc.set_hf32_mode(False)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_hf32_trans_mode_docstring():
    func_introduction = """
    调用该接口后，可设置 Mmad 的 HF32 取整模式，仅在 HF32 模式开启时有效。
    在 HF32 模式下，将按照给定模式对 FP32 数据进行舍入。
    """
    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline void SetHF32TransMode(bool hf32TransMode);
    """
    param_list = """
    **参数说明**

    - hf32_trans_mode：
      Mmad HF32 取整模式控制参数，bool 类型。
      - True：FP32 将以向零靠近的方式舍入为 HF32。
      - False：FP32 将以最接近偶数的方式舍入为 HF32。
    """
    constraint_list = """
    **约束说明**

    无。
    """
    py_example = """
    **调用示例**

    .. code-block:: python

        asc.set_hf32_trans_mode(True)
        asc.set_hf32_trans_mode(False)
    """
    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_mask_count_docstring():
    func_introduction = """
    设置掩码模式为 Counter 模式。在该模式下，
    矢量计算时不需要开发者显式指定迭代次数和处理非对齐尾块，只需调用 SetMaskCount 即可自动推断。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline void SetMaskCount();
    """

    param_list = """
    **参数说明**

    - 无
    """

    constraint_list = """
    **约束说明**

    - 设置为 Counter 模式后，建议在矢量计算完成后调用 set_mask_norm 恢复 Normal 模式。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.set_mask_count()
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_mask_norm_docstring():
    func_introduction = """
    设置掩码模式为 Normal 模式，这是掩码操作的默认模式。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline void SetMaskNorm();
    """

    param_list = """
    **参数说明**

    - 无
    """

    constraint_list = """
    **约束说明**

    - 无
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.set_mask_norm()
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_mm_layout_transform_docstring():
    func_introduction = """
    调用该接口后，可设置 Mmad 的 M/N 方向优先顺序，
    控制矩阵乘加计算时先按 N 再按 M 方向还是先按 M 再按 N 方向。
    """
    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline void SetMMLayoutTransform(bool mmLayoutMode);
    """
    param_list = """
    **参数说明**

    - mm_layout_mode：
      Mmad M/N 方向控制参数，bool 类型。
      - True：代表 CUBE 将首先通过 N 方向，然后通过 M 方向产生结果。
      - False：代表 CUBE 将首先通过 M 方向，然后通过 N 方向产生结果。
    """
    constraint_list = """
    **约束说明**

    无。
    """
    py_example = """
    **调用示例**

    .. code-block:: python

        asc.set_mm_layout_transform(True)
        asc.set_mm_layout_transform(False)
    """
    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_unary_docstring(cpp_name: Optional[str] = None, append_text: str = "") -> Callable[[T], T]:
    func_introduction = f"""
    {append_text}
    """

    cpp_signature = f"""
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                            const int32_t& calCount)

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                            uint64_t mask[], const uint8_t repeatTimes,
                                            const UnaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                            uint64_t mask, const uint8_t repeatTimes,
                                            const UnaryRepeatParams& repeatParams)

        """

    param_list = """
    **参数说明**

    - is_set_mask：是否在接口内部设置mask。
    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - count: 参与计算的元素个数。
    - mask: 用于控制每次迭代内参与计算的元素。
    - repeat_times: 重复迭代次数。
    - params: 控制操作数地址步长的参数。
    """
    api_name = NAME_TRANS[cpp_name]
    py_example = f"""
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 256 // asc.half.sizeof()
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.{api_name}(dst, src, mask=mask, repeat_times=4, repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.{api_name}(dst, src, mask=mask, repeat_times=4, repeat_params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.{api_name}(dst, src, count=512)

    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    """

    docstr = f"""
    {func_introduction}
    {cpp_signature}
    {param_list}
    {constraint_list}
    {py_example}
    """

    def decorator(fn: T) -> T:
        fn.__doc__ = docstr
        return fn

    return decorator


def gather_mask_docstring():
    func_introduction = """
    以内置固定模式对应的二进制或者用户自定义输入的Tensor数值对应的二进制为gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U, GatherMaskMode mode = defaultGatherMaskMode>
        __aicore__ inline void GatherMask(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                          const LocalTensor<U>& src1Pattern, const bool reduceMode,
                                          const uint32_t mask, const GatherMaskParams& gatherMaskParams,
                                          uint64_t& rsvdCnt)

    .. code-block:: c++

        template <typename T, GatherMaskMode mode = defaultGatherMaskMode>
        __aicore__ inline void GatherMask(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                          const uint8_t src1Pattern, const bool reduceMode,
                                          const uint32_t mask, const GatherMaskParams& gatherMaskParams,
                                          uint64_t& rsvdCnt)

    """

    param_list = """
    **参数说明**

    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型需要与目的操作数保持一致。
    - src1_pattern: gather mask（数据收集的掩码），分为内置固定模式和用户自定义模式两种：

      - 内置固定模式：src1_pattern数据类型为uint8_t，取值范围为[1,7]，所有repeat迭代使用相同的gather mask。不支持配置src1_repeat_stride。

        1：01010101…0101 # 每个repeat取偶数索引元素
        2：10101010…1010 # 每个repeat取奇数索引元素
        3：00010001…0001 # 每个repeat内每四个元素取第一个元素
        4：00100010…0010 # 每个repeat内每四个元素取第二个元素
        5：01000100…0100 # 每个repeat内每四个元素取第三个元素
        6：10001000…1000 # 每个repeat内每四个元素取第四个元素
        7：11111111...1111 # 每个repeat内取全部元素

      - 用户自定义模式：src1_pattern数据类型为LocalTensor，迭代间间隔由src1_repeat_stride决定，迭代内src1_pattern连续消耗。

    - reduce_mode: 用于选择mask参数模式，数据类型为bool，支持如下取值：

      - False：Normal模式。该模式下，每次repeat操作256Bytes数据，总的数据计算量为repeat_times * 256Bytes。mask参数无效，建议设置为0。按需配置repeat_times、src0_block_stride、src0_repeat_stride参数。支持src1_pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1_repeat_stride。
      - True：Counter模式。根据mask等参数含义的不同，该模式有以下两种配置方式：

        配置方式一：每次repeat操作mask个元素，总的数据计算量为repeat_times * mask个元素。mask值配置为每一次repeat计算的元素个数。按需配置repeat_times、src0_block_stride、src0_repeat_stride参数。支持src1_pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1_repeat_stride。
        配置方式二：总的数据计算量为mask个元素。mask配置为总的数据计算量。repeat_times值不生效，指令的迭代次数由源操作数和mask共同决定。按需配置src0_block_stride、src0_repeat_stride参数。支持src1_pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1_repeat_stride。

    - mask: 用于控制每次迭代内参与计算的元素。根据reduce_mode，分为两种模式：
      - Normal模式：mask无效，建议设置为0。
      - Counter模式：取值范围[1, 232 – 1]。不同的版本型号Counter模式下，mask参数表示含义不同。具体配置规则参考上文reduce_mode参数描述。
    - params: 控制操作数地址步长的数据结构，GatherMaskParams类型。具体参数包括：
      - src0_block_stride: 用于设置src0同一迭代不同DataBlock间的地址步长。
      - repeat_times: 迭代次数。
      - src0_repeat_stride: 用于设置src0相邻迭代间的地址步长。
      - src1_repeat_stride: 用于设置src1相邻迭代间的地址步长。
    - mode: 模板参数，用于指定gather_mask的模式，当前仅支持默认模式GatherMaskMode.DEFAULT，为后续功能做预留。
    """

    return_list = """
    **返回值说明**

    该条指令筛选后保留下来的元素计数，对应dst_local中有效元素个数，数据类型为uint64_t。

    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    - 若调用该接口前为Counter模式，在调用该接口后需要显式设置回Counter模式（接口内部执行结束后会设置为Normal模式）。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        src0_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        dst_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        pattern_value = 2
        reduce_mode = False
        gather_mask_mode = asc.GatherMaskMode.DEFAULT
        mask = 0
        params = asc.GatherMaskParams(src0_block_stride=1, repeat_times=1, src0_repeat_stride=0, src1_repeat_stride=0)
        rsvd_cnt = asc.gather_mask(dst_local, src0_local, pattern_value, reduce_mode, mask, params, gather_mask_mode)

    """
    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def scalar_cast_docstring():
    func_introduction = """
    对标量的数据类型进行转换。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U, RoundMode roundMode>
        __aicore__ inline U ScalarCast(T valueIn);
    """

    param_list = """
    **参数说明**

    - value_in：被转换数据类型的标量。
    - dtypeL：
      目标数据类型，由Python前端指定。
      - 支持：asc.half、asc.float16、asc.int32。
    - round_mode：
      精度转换处理模式，类型为RoundMode枚举值。
      - asc.RoundMode.CAST_NONE：在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不取整。
      - asc.RoundMode.CAST_RINT：rint，四舍六入五成双取整。
      - asc.RoundMode.CAST_FLOOR：floor，向负无穷取整。
      - asc.RoundMode.CAST_CEIL：ceil，向正无穷取整。
      - asc.RoundMode.CAST_ROUND：round，四舍五入取整。
      - asc.RoundMode.CAST_ODD：Von Neumann rounding，最近邻奇数舍入。

      - 对应支持关系

        - float -> half(f322f16)：asc.RoundMode.CAST_ODD
        - float -> int32(f322s32)：asc.RoundMode.CAST_ROUND、asc.RoundMode.CAST_CEIL、asc.RoundMode.CAST_FLOOR、asc.RoundMode.CAST_RINT

      - ScalarCast的精度转换规则与Cast保持一致
    """

    return_list = """
    **返回值说明**

    返回值为转换后的标量，类型与dtype一致。

    """

    py_example = """
    **调用示例**

    .. code-block:: python

        value_in = 2.5
        dtype = asc.int32
        round_mode = asc.RoundMode.CAST_ROUND
        value_out = asc.scalar_cast(value_in, dtype, round_mode)
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def scalar_get_sff_value_docstring():
    func_introduction = """
    获取一个 uint64_t 类型数字的二进制表示中，从最低有效位（LSB）开始第一个 0 或 1 出现的位置。
    如果未找到指定值，则返回 -1。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <int countValue>
        __aicore__ inline int64_t ScalarGetSFFValue(uint64_t valueIn);
    """

    param_list = """
    **参数说明**

    - value_in：
      输入数据，类型为 uint64_t。
      - 表示待查找的无符号整数。

    - count_value：
      指定要查找的值，类型为 int。
      - 取值为 0 或 1。
      - 0 表示查找从最低有效位开始的第一个 0 出现的位置；
      - 1 表示查找从最低有效位开始的第一个 1 出现的位置。
    """

    return_list = """
    **返回值说明**

    - 返回 int64 类型的数：
      表示 value_in 的二进制表示中，第一个匹配值（0 或 1）出现的位置。
      - 如果未找到，则返回 -1。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        value_in = 28
        count_value = 1
        one_count = asc.scalar_get_sff_value(value_in, count_value)
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def scalar_get_count_of_value_docstring():
    func_introduction = """
    获取一个 uint64_t 类型数字的二进制中 0 或者 1 的个数。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

        .. code-block:: c++

            template <int countValue>
            __aicore__ inline int64_t ScalarGetCountOfValue(uint64_t valueIn);
    """

    param_list = """
    **参数说明**

    - value_in：输入数据
      - 被统计的二进制数字。

    - count_value：指定统计 0 还是统计 1 的个数。
      - 只能输入 0 或 1 。
    """

    return_list = """
    **返回值说明**

    - value_in 中 0 或者 1 的个数。

    """

    py_example = """
    **调用示例**

        .. code-block:: python

            import asc
            value_in = 0xffff
            count_bits = 1
            one_count = asc.scalar_get_count_of_value(value_in, count_bits)
            # 输出数据 oneCount : 16
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def scalar_count_leading_zero_docstring():
    func_introduction = """
    计算一个 uint64_t 类型数字前导 0 的个数
    （二进制从最高位到第一个 1 一共有多少个 0 ）。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

        .. code-block:: c++

            __aicore__ inline int64_t ScalarCountLeadingZero(uint64_t valueIn);
    """

    param_list = """
    **参数说明**

    - value_in：输入数据
      - 被统计的二进制数字。
    """

    return_list = """
    **返回值说明**

    - 返回 value_in 的前导 0 的个数。
    """

    py_example = """
    **调用示例**

        .. code-block:: python

            import asc
            value_in = 0x0fffffffffffffff
            ans = asc.scalar_count_leading_zero(value_in)
            # ans 输出: 4
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def count_bits_cnt_same_as_sign_bit_docstring():
    func_introduction = """
    计算一个 int64_t 类型数字的二进制中，从最高数值位开始与符号位相同的连续比特位的个数。
    当输入是 -1 （比特位全 1 ）或者 0 （比特位全 0 ）时，返回 -1 。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

        .. code-block:: c++

            __aicore__ inline int64_t CountBitsCntSameAsSignBit(int64_t valueIn);
    """

    param_list = """
    **参数说明**

    - value_in：输入数据
      - 数据类型 int64_t 。
    """

    return_list = """
    **返回值说明**

    - 返回从最高数值位开始和符号位相同的连续比特位的个数。
    """

    py_example = """
    **调用示例**

        .. code-block:: python

            import asc
            value_in = 0x0f00000000000000
            ans = asc.count_bits_cnt_same_as_sign_bit(value_in)
            # ans 输出: 3
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def set_atomic_add_docstring():
    func_introduction = """
    调用该接口后，可对后续的从VECOUT/L0C/L1到GM的数据传输开启原子累加，
    通过dtype参数设定不同类型的数据。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void SetAtomicAdd();
    """

    param_list = """
    **参数说明**

    - dtype：
      原子加操作的数据类型，由 Python 前端指定。
      - 支持类型：asc.float16、asc.float32、asc.int32、asc.half。
    """

    constraint_list = """
    **约束说明**

    - 累加操作完成后，建议通过set_atomic_none关闭原子累加，以免影响后续相关指令功能。
    - 该指令执行前不会对GM的数据做清零操作，开发者根据实际的算子逻辑判断是否需要清零，如果需要自行进行清零操作。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dtype = asc.float32
        asc.set_atomic_add(dtype)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_atomic_max_docstring():
    func_introduction = """
    原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较：
    将待拷贝的内容和GM已有内容进行比较，将最大值写入GM。
    可通过设置模板参数来设定不同的数据类型。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void SetAtomicMax();
    """

    param_list = """
    **参数说明**

    - dtype：
      原子 max 操作的数据类型，由 Python 前端指定。
      - 支持类型：asc.float16、asc.float32、asc.int32、asc.half。
    """

    constraint_list = """
    **约束说明**

    - 使用完后，建议通过set_atomic_none关闭原子累加，以免影响后续相关指令功能。
    - 对于910B，目前无法对bfloat16_t类型设置inf/nan模式。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dtype = asc.int32
        asc.set_atomic_max(dtype)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_atomic_min_docstring():
    func_introduction = """
    原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较，
    将待拷贝的内容和GM已有内容进行比较，将最小值写入GM。
    可通过设置模板参数来设定不同的数据类型。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void SetAtomicMin();
    """

    param_list = """
    **参数说明**

    - dtype：
      原子 min 操作的数据类型，由 Python 前端指定。
      - 支持类型：asc.float16、asc.float32、asc.int32、asc.half。
    """

    constraint_list = """
    **约束说明**

    使用完后，建议通过set_atomic_none关闭原子累加，以免影响后续相关指令功能。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dtype = asc.float16
        asc.set_atomic_min(dtype)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_atomic_none_docstring():
    func_introduction = """
    清空原子操作的状态。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline void SetAtomicNone();
    """

    param_list = """
    **参数说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.set_atomic_none()
    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def set_atomic_type_docstring():
    func_introduction = """
    通过设置模板参数来设定原子操作不同的数据类型。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void SetAtomicType();
    """

    param_list = """
    **参数说明**

    - dtype：
      原子操作使用的数据类型，由 Python 前端指定。
      - 支持类型：asc.float16、asc.float32、asc.int32、asc.half。
    """

    constraint_list = """
    **约束说明**

    - 需要和set_atomic_add、set_atomic_max、set_atomic_min配合使用。
    - 使用完成后，建议清空原子操作的状态（详见set_atomic_none），以免影响后续相关指令功能。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dtype = asc.float16
        asc.set_atomic_type(dtype)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_load_data_boundary_docstring():
    func_introduction = """
    设置 load_3d 时 A1/B1 边界值。
    如果 load_3d 指令在处理源操作数时，源操作数在 A1/B1 上的地址超出设置的边界，则会从 A1/B1 起始地址开始读取数据。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline void SetLoadDataBoundary(uint32_t boundaryValue)
    """

    param_list = """
    **参数说明**

    - boundary_value
        边界值。
        load_3d_v1 指令：单位是 32 字节。
        load_3d_v2 指令：单位是字节。
    """

    constraint_list = """
    **约束说明**

    - 用于 load_3d_v1 时， boundary_value 的最小值是 16 （单位： 32 字节）；用于 load_3d_v2 时， boundary_value 的最小值是 1024 （单位：字节）。
    - 如果使用 SetLoadDataBoundary 接口设置了边界值，配合 load_3d 指令使用时， load_3d 指令的 A1/B1 初始地址要在设置的边界内。
    - 如果 boundary_value 设置为 0 ，则表示无边界，可使用整个 A1/B1 。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        asc.set_load_data_boundary(1024)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_load_data_padding_value_docstring():
    func_introduction = """
    用于调用 load_3d_v2v1接口/load_3d_v2 接口时设置 Pad 填充的数值。 load_3d_v1/load_3d_v2 的模板参数 isSetPadding 设置为 true 时，用户需要通过本接口设置 Pad 填充的数值，设置为 false 时，本接口设置的填充值不生效。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        _template <typename T>
        __aicore__ inline void SetLoadDataPaddingValue(const T padValue)
    """

    param_list = """
    **参数说明**

    - padValue
        输入， Pad 填充值的数值。
    """

    constraint_list = """
    **约束说明**
        无
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        asc.set_load_data_padding_value(10)
        asc.set_load_data_padding_value(2.0)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_load_data_repeat_docstring():
    func_introduction = """
    用于设置 load_3d_v2 接口的 repeat 参数。设置 repeat 参数后，可以通过调用一次 load_3d_v2 接口完成多个迭代的数据搬运。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline void SetLoadDataRepeat(const LoadDataRepeatParam& repeatParams)
    """

    param_list = """
    **参数说明**

    - repeatParams
        设置load_3d_v22接口的repeat参数，类型为LoadDataRepeatParam。

    - repeatParams
        height/width方向上的迭代次数，取值范围：repeatTime ∈[0, 255] 。默认值为1

    - repeatStride
        height/width方向上的前一个迭代与后一个迭代起始地址的距离，取值范围：n∈[0, 65535]，默认值为0。
        repeatMode为0，repeatStride的单位为16个元素。
        repeatMode为1，repeatStride的单位和具体型号有关。

    - repeatMode
        控制repeat迭代的方向，取值范围：k∈[0, 1] 。默认值为0。
        0：迭代沿height方向；
        1：迭代沿width方向。
    """

    constraint_list = """
    **约束说明**
        无
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        static_param = asc.LoadDataRepeatParam(
            repeatTime=4,
            repeatStride=8,
            repeatMode=0
        )
        asc.set_load_data_repeat(static_param)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def compare_docstring() -> Callable[[T], T]:
    func_introduction = """
    逐元素比较两个tensor大小，如果比较后的结果为真，则输出的结果的对应比特位为1，否则为0。可将结果存入寄存器中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U>
        __aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                        const LocalTensor<T>& src1, CMPMODE cmpMode, uint32_t count);

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Copmare(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                        const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask[],
                                        uint8_t repeatTimes, const BinaryRepeatParams& repeatParams);

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                        const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask,
                                        uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void Copmare(const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode,
                                        const uint64_t mask[], const BinaryRepeatParams& repeatParams);

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void Compare(const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode,
                                        const uint64_t mask, const BinaryRepeatParams& repeatParams);

    """

    param_list = """
    **参数说明**

    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0, src1: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - cmp_mode: CMPMODE类型，表示比较模式。
      - LT: src0小于（less than）src1
      - GT: src0大于（greater than）src1
      - GE: src0大于或等于（greater than or equal to）src1
      - EQ: src0等于（equal to）src1
      - NE: src0不等于（not equal to）src1
      - LE: src0小于或等于（less than or equal to）src1
    - count: 参与计算的元素个数。
    - mask: 用于控制每次迭代内参与计算的元素。
    - repeat_times: 重复迭代次数。
    - repeat_params: 控制操作数地址步长的参数。
    - is_set_mask: 是否在接口内部设置mask。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
    - 使用整个tensor参与计算的运算符重载功能，src0和src1需满足256字节对齐；使用tensor前n个数据参与计算的接口，设置count时，需要保证count个元素所占空间256字节对齐。
    - 将结果存入寄存器的接口没有repeat输入，repeat默认为1，即一条指令计算256B的数据。
    - 将结果存入寄存器的接口会将结果写入128bit的cmpMask寄存器中，可以用GetCmpMask接口获取寄存器保存的数据。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          # repeat_times = 1，一次迭代计算128个数
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare(dst, src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1, repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          # repeat_times = 1，一次迭代计算128个数
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare(dst, src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1, repeat_params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.compare(dst, src0, src1, cmp_mode=asc.CMPMODE.LT, count=512)

    - tensor高维切分计算样例-mask连续模式，结果存入寄存器中

      .. code-block:: python

          mask = 128
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare(src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式，结果存入寄存器中

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare(src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_params=params)

    """

    return func_introduction, cpp_signature, param_list, "", constraint_list, py_example


def mrg_sort_docstring():
    func_introduction = """
    将已经排好序的多个队列合并成一条队列，结果按照指定顺序排序。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void MrgSort(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src,
                                     const uint16_t elementCountList[], uint32_t sortedNum[],
                                     uint16_t validBit, uint16_t repeatTime,
                                     bool isExhaustedSuspension = false)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void MrgSort(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src,
                                     const MrgSort4Info& params)
    """

    param_list = """
    **参数说明**

    - dst (asc.LocalTensor): 目的操作数，存储经过排序后的结果。支持的TPosition为VECIN/VECCALC/VECOUT。
    - src (asc.MrgSortSrcList): 源操作数，多个已经排好序的队列。具体定义如下：

    .. code-block:: python

        class MrgSortSrcList:
            src1: LocalTensor  # 第一个已经排好序的Region Proposals队列
            src2: LocalTensor  # 第二个已经排好序的Region Proposals队列
            src3: LocalTensor  # 第三个已经排好序的Region Proposals队列
            src4: LocalTensor  # 第四个已经排好序的Region Proposals队列

    - element_count_list: 各个源队列的长度（元素数目），类型为uint16_t数组。
    - sorted_num: 输出参数，存储各个队列排序后的元素数目，类型为uint32_t数组。
    - valid_bit: 有效队列个数位掩码。
    - repeat_time: 迭代次数。
    - is_exhausted_suspension: 某条队列耗尽后，指令是否需要停止，类型为bool，默认false。
    - params (asc.MrgSort4Info)：排序所需参数。
      - element_lengths: 四个源Region Proposals队列的长度（Region Proposal数目），类型为长度为4的uint16_t数组，每个元素取值范围[0, 4095]。
      - is_exhausted_suspension: 某条队列耗尽后，指令是否需要停止，类型为bool，默认false。
      - valid_bit：有效队列个数。
      - repeat_times：迭代次数，每一次源操作数和目的操作数跳过四个队列总长度。取值范围[1,255]。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 不支持源操作数与目的操作数之间存在地址重叠。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        src_list = asc.MrgSortSrcList(queue1, queue2, queue3, queue4)
        element_counts = [16, 16, 16, 16]
        sorted_nums = [0, 0, 0, 0]
        valid_bit = 15  # 所有4个队列都有效
        repeat_time = 1
        asc.mrg_sort(dst, src_list, element_counts, sorted_nums, valid_bit, repeat_time)

        src_list = asc.MrgSortSrcList(queue1, queue2, queue3, queue4)
        element_lengths = [16, 16, 16, 16]
        params = asc.MrgSort4Info(element_lengths, False, 15, 1)
        asc.mrg_sort(dst, src_list, params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def mrg_sort4_docstring():
    func_introduction = """
    将已经排好序的最多4条Region Proposals队列，排列并合并成1条队列，结果按照score域由大到小排序。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void MrgSort4(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src, const MrgSort4Info& params)
    """

    param_list = """
    **参数说明**

    - dst (asc.LocalTensor): 目的操作数，存储经过排序后的Region Proposals。支持的TPosition为VECIN/VECCALC/VECOUT。
    - src (asc.MrgSortSrcList): 源操作数，多个已经排好序的队列。具体定义如下：
    .. code-block:: python

        class MrgSortSrcList:
            src1: LocalTensor
            src2: LocalTensor
            src3: LocalTensor
            src4: LocalTensor

    - params (asc.MrgSort4Info)
        排序所需参数。
        - element_lengths: 四个源Region Proposals队列的长度（Region Proposal数目），类型为长度为4的uint16_t数组，每个元素取值范围[0, 4095]。
        - is_exhausted_suspension: 某条队列耗尽后，指令是否需要停止，类型为bool，默认false。
        - valid_bit：有效队列个数。
        - repeat_times：迭代次数，每一次源操作数和目的操作数跳过四个队列总长度。取值范围[1,255]。

    """

    constraint_list = """
    **约束说明**

    - 当存在proposal[i]与proposal[j]的score值相同时，如果i>j，则proposal[j]将首先被选出来，排在前面。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 不支持源操作数与目的操作数之间存在地址重叠。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # vconcat_work_local为已经创建并且完成排序的4个Region Proposals，每个Region Proposal数目是16个
        src_list = asc.MrgSortSrcList(vconcat_work_local[0], vconcat_work_local[1], vconcat_work_local[2], vconcat_work_local[3])
        element_lengths = [16, 16, 16, 16]
        src_info = asc.MrgSort4Info(element_lengths, False, 15, 1)
        asc.mrg_sort4(dst_local, src_list, src_info)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def rp_sort16_docstring():
    func_introduction = """
    根据Region Proposals中的score域对其进行排序（score大的排前面），每次排16个Region Proposals。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void RpSort16(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime)
    """

    param_list = """
    **参数说明**

    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - repeat_time: 重复迭代次数。
    """

    constraint_list = """
    **约束说明**

    - 用户需保证src和dst中存储的Region Proposal数目大于实际所需数据，否则会存在tensor越界错误。
    - 当存在proposal[i]与proposal[j]的score值相同时，如果i>j，则proposal[j]将首先被选出来，排在前面。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # repeat_time = 2, 对2个Region Proposal进行排序
        asc.rp_sort16(dst_local, src_local, 2)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def sort32_docstring():
    func_introduction = """
    排序函数，一次迭代可以完成32个数的排序。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void Sort32(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                     const LocalTensor<uint32_t>& src1, const int32_t repeatTime)
    """

    param_list = """
    **参数说明**
    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src1: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - repeat_time: 重复迭代次数。
    """

    constraint_list = """
    **约束说明**

    - 当存在score[i]与score[j]相同时，如果i>j，则score[j]将首先被选出来，排在前面。
    - 每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # repeat_time = 4, 对128个数分成4组进行排序，每次完成1组32个数的排序
        asc.sort32(dst_local, src_local0, src_local1, 4)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def sort_docstring():
    func_introduction = """
    排序函数，按照数值大小进行降序排序。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, bool isFullSort>
        __aicore__ inline void Sort(const LocalTensor<T> &dst, const LocalTensor<T> &concat,
                                   const LocalTensor<uint32_t> &index, LocalTensor<T> &tmp,
                                   const int32_t repeatTime)
    """

    param_list = """
    **参数说明**

    - dst (asc.LocalTensor): 目的操作数，shape为[2n]。
    - concat (asc.LocalTensor): 源操作数，shape为[n]，数据类型与目的操作数保持一致。
    - index (asc.LocalTensor): 源操作数，shape为[n]。固定为uint32_t数据类型。
    - tmp (asc.LocalTensor): 临时空间。接口内部复杂计算时用于存储中间变量，由开发者提供。数据类型与源操作数保持一致。
    - repeat_time (int): 重复迭代次数，int32_t类型。
    - is_full_sort (bool, 可选): 模板参数，是否开启全排序模式。
    """

    constraint_list = """
    **约束说明**

    - 当存在score[i]与score[j]相同时，如果i>j，则score[j]将首先被选出来，排在前面，即index的顺序与输入顺序一致。
    - 非全排序模式下，每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # 处理128个half类型数据
        element_count = 128
        sort_repeat_times = element_count // 32
        extract_repeat_times = element_count // 32
        asc.sort(dst_local, concat_local, index_local, tmp_local, sort_repeat_times, is_full_sort=True)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def compare_scalar_docstring() -> Callable[[T], T]:
    func_introduction = """
    逐元素比较一个tensor中的元素和另一个scalar的大小，如果比较后的结果为真，则输出的结果的对应比特位为1，否则为0。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U>
        __aicore__ inline void CompareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                              const T src1Scalar, CMPMODE cmpMode, uint32_t count);

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void CopmareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                              const T src1Scalar, CMPMODE cmpMode, const uint64_t mask[],
                                              uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void CompareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                              const T src1Scalar, CMPMODE cmpMode, const uint64_t mask,
                                              uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

    """

    param_list = """
    **参数说明**

    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src1_scalar: 源操作数，Scalar标量。数据类型和src0保持一致。
    - cmp_mode: CMPMODE类型，表示比较模式。
      - LT: src0小于（less than）src1
      - GT: src0大于（greater than）src1
      - GE: src0大于或等于（greater than or equal to）src1
      - EQ: src0等于（equal to）src1
      - NE: src0不等于（not equal to）src1
      - LE: src0小于或等于（less than or equal to）src1
    - count: 参与计算的元素个数。
    - mask: 用于控制每次迭代内参与计算的元素。
    - repeat_times: 重复迭代次数。
    - repeat_params: 控制操作数地址步长的参数。
    - is_set_mask: 是否在接口内部设置mask。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
    - 使用tensor前n个数据参与计算的接口，设置count时，需要保证count个元素所占空间256字节对齐。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          # repeat_times = 1，一次迭代计算128个数
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare_scalar(dst, src0, src1_scalar, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1,
                             repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          # repeat_times = 1，一次迭代计算128个数
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare_scalar(dst, src0, src1_scalar, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1,
                             repeat_params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.compare_scalar(dst, src0, src1_scalar, cmp_mode=asc.CMPMODE.LT, count=512)

    """

    return func_introduction, cpp_signature, param_list, "", constraint_list, py_example


def get_cmp_mask_docstring() -> Callable[[T], T]:
    func_introduction = """
    用于获取compare（结果存入寄存器）指令的比较结果。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void GetCmpMask(const LocalTensor<T>& dst)

    """

    param_list = """
    **参数说明**

    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    """

    constraint_list = """
    **约束说明**

    - dst的空间大小不能少于128字节。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.get_cmp_mask(dst)

    """

    return func_introduction, cpp_signature, param_list, "", constraint_list, py_example


def set_cmp_mask_docstring() -> Callable[[T], T]:
    func_introduction = """
    为select不传入mask参数的接口设置比较寄存器。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void SetCmpMask(const LocalTensor<T>& src)

    """

    param_list = """
    **参数说明**

    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        src = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        asc.set_cmp_mask(src)

    """

    return func_introduction, cpp_signature, param_list, "", "", py_example


class ListTensorDescDocstring:

    def __init__(self) -> None:
        ...

    @staticmethod
    def init_docstring():
        func_introduction = """
        初始化函数，用于解析对应的内存排布。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline void Init(__gm__ void* data, uint32_t length = 0xffffffff, uint32_t shapeSize = 0xffffffff)

        """

        param_list = """
        **参数说明**

        - data: 待解析数据的首地址。
        - length: 待解析内存的长度。
        - shape_size: 数据指针的个数。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            x_desc = asc.ListTensorDesc()
            x_desc.init(data=x, length=0xffffffff, shape_size=0xffffffff)

        """

        return [func_introduction, cpp_signature, param_list, "", "", py_example]

    @staticmethod
    def get_desc_docstring():
        func_introduction = """
        根据index获得功能说明图中对应的TensorDesc信息。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template<class T> void GetDesc(TensorDesc<T>& desc, uint32_t index);

        """

        param_list = """
        **参数说明**

        - desc: 出参，解析后的Tensor描述信息。
        - index: 索引值。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            x_desc = asc.ListTensorDesc(data=x, length=0xffffffff, shape_size=0xffffffff)
            y = asc.TensorDesc()
            x_desc.get_desc(y, index=0)

        """

        return [func_introduction, cpp_signature, param_list, "", "", py_example]

    @staticmethod
    def get_data_ptr_docstring():
        func_introduction = """
        根据index获取储存对应数据的地址。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template<class T> __aicore__ inline __gm__ T* GetDataPtr(uint32_t index)

        """

        param_list = """
        **参数说明**

        - index: 索引值。
        - dtype: 输出的指针地址指向的数据的数据类型。
        """

        return_list = """
        **返回值说明**

        储存对应数据的地址。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            x_desc = asc.ListTensorDesc(data=x, length=0xffffffff, shape_size=0xffffffff)
            x_ptr = x_desc.get_data_ptr(index=0, dtype=asc.float16)

        """

        return [func_introduction, cpp_signature, param_list, return_list, "", py_example]

    @staticmethod
    def get_size_docstring():
        func_introduction = """
        获取ListTensor中包含的数据指针的个数。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline uint32_t GetSize()

        """

        param_list = """
        **参数说明**

        无。
        """

        return_list = """
        **返回值说明**

        数据指针的个数。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            x_desc = asc.ListTensorDesc(data=x, length=0xffffffff, shape_size=0xffffffff)
            x_size = x_desc.get_size()

        """

        return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def select_docstring() -> Callable[[T], T]:
    func_introduction = """
    给定两个源操作数src0和src1，根据sel_mask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U>
        __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                                       const LocalTensor<T>& src0, T src1, SELMODE selMode, uint32_t count)

    .. code-block:: c++

        template <typename T, typename U>
        __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                                       const LocalTensor<T>& src0, const LocalTensor<T>& src1,
                                       SELMODE selMode, uint32_t count)

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                                       const LocalTensor<T>& src0, T src1, SELMODE selMode, uint64_t mask[],
                                       uint8_t repeatTime, const BinaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                                       const LocalTensor<T>& src0, T src1, SELMODE selMode, uint64_t mask,
                                       uint8_t repeatTime, const BinaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, typename U>
        __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                                       const LocalTensor<T>& src0, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                                       const LocalTensor<T>& src0, const LocalTensor<T>& src1, SELMODE selMode, uint64_t mask[],
                                       uint8_t repeatTime, const BinaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                                       const LocalTensor<T>& src0, const LocalTensor<T>& src1, SELMODE selMode, uint64_t mask,
                                       uint8_t repeatTime, const BinaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, SELMODE selMode>
        __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1,
                                       uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
    """

    param_list = """
    **参数说明**

    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - sel_mask: 选取mask，类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src1: 源操作数。
      - 当selMode为模式0或模式2时：类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
      - 当selMode为模式1时，类型为T，标量数据类型。
    - sel_mode: SELMODE类型，表示指令模式。
      - VSEL_CMPMASK_SPR: 模式0，根据selMask在两个tensor中选取元素，selMask中有效数据的个数存在限制，具体取决于源操作数的数据类型。
      - VSEL_TENSOR_SCALAR_MODE: 模式1，根据selMask在1个tensor和1个scalar标量中选取元素，selMask无有效数据限制。
      - VSEL_TENSOR_TENSOR_MODE: 模式2，根据selMask在两个tensor中选取元素，selMask无有效数据限制。
    - count: 参与计算的元素个数。
    - mask: 用于控制每次迭代内参与计算的元素。
    - repeat_times: 重复迭代次数。
    - repeat_params: 控制操作数地址步长的参数。
    - is_set_mask: 是否在接口内部设置mask。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    - 对于模式1和模式2，使用时需要预留8K的Unified Buffer空间，作为接口的临时数据存放区。
    """

    py_example = """
    **调用示例**

    - tensor 前 n 个数据计算样例（模式 0 / 模式 2）

      .. code-block:: python

          asc.select(
              z_local,
              y_local,
              x_local,
              p_local,
              sel_mode=asc.SelMode.VSEL_CMPMASK_SPR,
              count=512
          )

    - tensor 前 n 个数据计算样例（模式 1）

      .. code-block:: python

          asc.select(
              z_local,
              y_local,
              x_local,
              0.0,
              sel_mode=asc.SelMode.VSEL_TENSOR_SCALAR_MODE,
              count=512
          )

    - tensor 高维切分计算样例 —— 标量 + mask 逐 bit（模式 1）

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.select(
              z_local,
              y_local,
              x_local,
              0.0,
              sel_mode=asc.SelMode.VSEL_TENSOR_SCALAR_MODE,
              mask=mask,
              repeat_times=1,
              repeat_params=params
          )

    - tensor 高维切分计算样例 —— 标量 + mask 连续（模式 1）

      .. code-block:: python

          mask = 512
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.select(
              z_local,
              y_local,
              x_local,
              0.0,
              sel_mode=asc.SelMode.VSEL_TENSOR_SCALAR_MODE,
              mask=mask,
              repeat_times=1,
              repeat_params=params
          )

    - tensor 高维切分计算样例 —— 不传入 mask（需配合寄存器 mask 使用，模式 1）

      .. code-block:: python

          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.select(
              z_local,
              y_local,
              x_local,
              repeat_times=1,
              repeat_params=params
          )

    - tensor 高维切分计算样例 —— Tensor + mask 逐 bit（模式 0 / 模式 2）

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.select(
              z_local,
              y_local,
              x_local,
              p_local,
              sel_mode=asc.SelMode.VSEL_CMPMASK_SPR,
              mask=mask,
              repeat_times=1,
              repeat_params=params
          )

    - tensor 高维切分计算样例 —— Tensor + mask 连续（模式 0 / 模式 2）

      .. code-block:: python

          mask = 512
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.select(
              z_local,
              y_local,
              x_local,
              p_local,
              sel_mode=asc.SelMode.VSEL_CMPMASK_SPR,
              mask=mask,
              repeat_times=1,
              repeat_params=params
          )

    - tensor 高维切分计算样例 —— 寄存器版本（无 selMask，模式 0 / 模式 2）

      .. code-block:: python

          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.select(
              z_local,
              x_local,
              p_local,
              repeat_times=1,
              repeat_params=params,
              sel_mode=asc.SelMode.VSEL_CMPMASK_SPR
          )
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_fix_pipe_pre_quant_flag_docstring():
    func_introduction = """
    DataCopy（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中标量量化参数。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetFixpipePreQuantFlag(uint64_t config)
    """

    param_list = """
    **参数说明**

    - config: 量化过程中使用到的标量量化参数，类型为uint64_t。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        deq_scalar = 11
        asc.set_fix_pipe_pre_quant_flag(deq_scalar)
    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def wait_pre_block_docstring():
    func_introduction = """
    多核同步接口，通过读取 Global Memory 中的标志位，等待上一个 AI Core 完成操作。

    此接口通常与 `notify_next_block` 配对使用，形成核间同步的“握手”机制。
    在使用前，需要确保已通过 `init_determine_compute_workspace` 接口初始化了核间同步所需的共享内存。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void WaitPreBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)
    """

    param_list = """
    **参数说明**

    - gm_workspace (asc.GlobalTensor): Global Memory 上的临时工作空间。
        - 用于核间通信的共享内存区域。通过读取此空间中的值，来判断上一个核是否已完成操作。
        - 类型必须为 `GlobalTensor<int32_t>`。
    - ub_workspace (asc.LocalTensor): UB 上的临时工作空间。
        - 用于在 AI Core 内部操作 `gm_workspace` 的暂存区。
        - 类型必须为 `LocalTensor<int32_t>`。
    """

    return_value = """
    **返回值**

    无。
    """

    constraint_list = """
    **约束说明**

    - 需要保证每个核调用该接口的次数相同。
    - gm_workspace申请的空间最少要求为：blockNum * 32Bytes；ub_workspace申请的空间最少要求为：blockNum * 32 + 32Bytes；其中blockNum为调用的核数，可调用get_block_num获取。
    - 分离模式下，使用该接口进行多核同步时，仅对AIV核生效，wait_pre_block和notify_next_block之间仅支持插入矢量计算相关指令，对矩阵计算相关指令不生效。
    - 使用该接口进行多核控制时，算子调用时指定的逻辑blockNum必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.wait_pre_block(gm_workspace, ub_workspace)

    """

    return [func_introduction, cpp_signature, param_list, return_value, constraint_list, py_example]


def notify_next_block_docstring():
    func_introduction = """
    多核同步接口，通过写入 Global Memory 中的标志位，通知下一个 AI Core 当前核的操作已完成。

    此接口通常与 `wait_pre_block` 配对使用。当前核调用此函数后，等待此核的下一个核将能够通过 `wait_pre_block` 检测到状态变化，从而继续执行。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline void NotifyNextBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)
    """

    param_list = """
    **参数说明**

    - gm_workspace (asc.GlobalTensor): Global Memory 上的临时工作空间。
        - 用于核间通信的共享内存区域。通过向此空间写入一个特定的标志位，来通知下一个核。
        - 类型必须为 `GlobalTensor<int32_t>`。
    - ub_workspace (asc.LocalTensor): UB 上的临时工作空间。
        - 用于在 AI Core 内部操作 `gm_workspace` 的暂存区。
        - 类型必须为 `LocalTensor<int32_t>`。
    """

    return_value = """
    **返回值**

    无。
    """

    constraint_list = """
    **约束说明**

    - 需要保证每个核调用该接口的次数相同。
    - gm_workspace申请的空间最少要求为：blockNum * 32Bytes；ub_workspace申请的空间最少要求为：blockNum * 32 + 32Bytes；其中blockNum为调用的核数，可调用get_block_num获取。
    - 分离模式下，使用该接口进行多核同步时，仅对AIV核生效，wait_pre_block和notify_next_block之间仅支持插入矢量计算相关指令，对矩阵计算相关指令不生效。
    - 使用该接口进行多核控制时，算子调用时指定的逻辑blockNum必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。
    """

    py_example = """
    **调用示例**

        .. code-block:: python

            asc.notify_next_block(gm_workspace, ub_workspace)

    """

    return [func_introduction, cpp_signature, param_list, return_value, constraint_list, py_example]


def fixpipe_docstring():
    func_introduction = """
    矩阵计算完成后，对结果进行处理，例如对计算结果进行量化操作，并把数据从CO1搬迁到Global Memory中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    通路CO1->GM，不使能tensor量化功能：

    .. code-block:: c++

        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
        void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsV220& intriParams)

    通路CO1->GM，使能tensor量化功能：

    .. code-block:: c++

        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR, typename S = uint64_t,
                  typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
        __aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
                                        const LocalTensor<S>& cbufWorkspace, const FixpipeParamsV220& intriParams)
    """

    param_list = """
    **参数说明**

    - dst: 目的操作数，类型为GlobalTensor。数据格式为NZ或ND格式。经过Fixpipe处理，在量化操作之后，会将矩阵计算中多申请的数据删除。
    - src: 源操作数，支持的TPosition为CO1，为Mmad接口计算的结果，类型为LocalTensor。支持的数据类型为float/int32_t，数据格式为NZ格式。起始地址需要满足64B对齐。
    - cbufWorkspace: 量化参数，类型为LocalTensor<uint64_t>，支持的TPosition为A1。仅当quantPre为VDEQF16/VQF322B8_PRE/VREQ8时支持。
    - intriParams: Fixpipe搬运参数，类型为FixpipeParamsV220，包含以下参数：
      - nSize: 源NZ矩阵在N方向上的大小。
        - 不使能NZ2ND功能：取值范围：nSize∈[1, 4095]。若使能channelSplit功能，nSize必须为8的倍数，若不使能channelSplit功能，nSize必须为16的倍数。
        - 使能NZ2ND功能：取值范围：nSize∈[1, 4095]。
      - mSize: 源NZ矩阵在M方向上的大小。
        - 不使能NZ2ND功能：取值范围：mSize∈[1, 65535]。
        - 使能NZ2ND功能：取值范围：mSize∈[1, 8192]。
      - srcStride: 源NZ矩阵中相邻Z排布的起始地址偏移，取值范围：srcStride∈[0, 65535]，单位：C0_Size(16*sizeof(T)，T为src的数据类型)。
      - dstStride: 目的矩阵的地址步长。
        - 不使能NZ2ND功能：目的NZ矩阵中相邻Z排布的起始地址偏移，取值不为0，单位：datablock(32Bytes)。
        - 使能NZ2ND功能：目的ND矩阵每一行中的元素个数，取值不为0，单位：element。
      - quantPre: 量化模式，类型为QuantMode_t，默认值为QuantMode_t::NoQuant（不使能量化功能）。支持以下取值：
        - NoQuant：不使能量化功能
        - F322F16：float量化成half，量化结果支持INF_NAN模式
        - F322BF16：float量化成bfloat16_t，量化结果支持INF_NAN模式
        - DEQF16：int32_t量化成half，scalar量化，量化结果不支持INF_NAN模式
        - VDEQF16：int32_t量化成half，tensor量化，量化结果不支持INF_NAN模式
        - QF322B8_PRE：float量化成uint8_t/int8_t，scalar量化
        - VQF322B8_PRE：float量化成uint8_t/int8_t，tensor量化
        - REQ8：int32_t量化成uint8_t/int8_t，scalar量化
        - VREQ8：int32_t量化成uint8_t/int8_t，tensor量化
      - deqScalar: scalar量化参数，表示单个scale值，quantPre量化模式为scalar量化时需要设置该参数。支持的数据类型为uint64_t。
      - ndNum: 源NZ矩阵的数目，也就是传输ND矩阵的数目，取值范围：ndNum∈[1, 65535]。
      - srcNdStride: 不同NZ矩阵起始地址之间的间隔，取值范围：srcNdStride∈[1, 512]，单位：1024B。当ndNum配置为1时，srcNdStride配置为0即可，不生效。
      - dstNdStride: 目的相邻ND矩阵起始地址之间的偏移，取值范围：dstNdstride∈[1, 65535]，单位：element。当ndNum配置为1时，dstNdStride配置为0即可，不生效。
      - reluEn: 是否使能relu的开关，false：不使能relu功能；true：使能relu功能。
      - unitFlag: Mmad指令和Fixpipe指令细粒度的并行控制。
        - 0：保留值
        - 2：使能unitFlag，硬件执行完指令之后，不会设置寄存器
        - 3：使能unitFlag，硬件执行完指令之后，会将unitFlag关闭
      - isChannelSplit: 是否使能通道拆分的功能。默认为false，不使能该功能。仅在src和dst都为float时才能使能通道拆分，且不能同时使能ChannelSplit和NZ2ND功能。
    """

    constraint_list = """
    **约束说明**

    - ndNum=0 表示不执行，此指令将不被执行并报warning。
    - 对于量化输入为float32数据类型的说明：标准的IEEE 754 float32格式为：1bit符号位，8bits指数位，23bits尾数位；当前AI处理器支持的float32格式为：1bit符号位，8bits指数位，10bits尾数位。如果用户提供的是标准的IEEE 754 float32输入，API内部会处理成处理器支持的float32格式进行计算，此时如果golden数据生成过程中使用的是标准的IEEE 754 float32数据，则可能引入精度不匹配问题，需要修正golden数据的生成，将量化参数的23bits尾数位的低13bits数据位清零再参与量化计算。
    """

    py_example = """
    **调用示例**

    - 通路CO1->GM，不使能tensor量化功能，使能F322F16量化

      .. code-block:: python

          dst_gm = asc.GlobalTensor()
          dst_gm.set_global_buffer(x)
          src_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          fixpipe_params = asc.FixpipeParamsV220(
            n_size=16, m_size=16, src_stride=32, dst_stride=32,
            quant_pre=asc.QuantModes.NoQuant, deq_scalar=0,
            nd_num=1, src_nd_stride=0, dst_nd_stride=0,
            relu_en=False, unit_flag=0, is_channel_split=False
          )
          asc.fixpipe(dst_gm, src_local, fixpipe_params)

    - 通路CO1->GM，使能tensor量化功能（VDEQF16）

      .. code-block:: python

          dst_gm = asc.GlobalTensor()
          dst_gm.set_global_buffer(x)
          src_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          workspace_local = asc.LocalTensor(dtype=asc.uint64, pos=asc.TPosition.VECIN, addr=512, tile_size=1024)
          fixpipe_params = asc.FixpipeParamsV220(
            n_size=16, m_size=32, src_stride=32, dst_stride=16,
            quant_pre=asc.QuantModes.VDEQF16, deq_scalar=0,
            nd_num=1, src_nd_stride=2, dst_nd_stride=512,
            relu_en=False, unit_flag=0, is_channel_split=False
          )
          asc.fixpipe(c_gm, c_local, cbuf_workspace, fixpipe_params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def brcb_docstring():
    func_introduction = """
    给定一个输入张量，每一次取输入张量中的8个数填充到结果张量的8个datablock（32Bytes）中去，每个数对应一个datablock。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void Brcb(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                    const uint8_t repeatTime, const BrcbRepeatParams& repeatParams)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - src0：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型和dst保持一致。
    - repeat_time：指令迭代次数，每次迭代完成8个datablock的数据收集，数据范围：repeat_time∈[0,255]。
    - repeat_params：用于控制指令迭代的相关参数。
    """

    constraint_list = """
    **约束说明**

    - 不支持src0与dst为同一块内存地址。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        brcb_params = asc.BrcbRepeatParams(1, 8)
        asc.brcb(x_buf, y_buf, 2, brcb_params)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def axpy_docstring():
    func_introduction = """
    源操作数src中每个元素与标量求积后和目的操作数dst中的对应元素相加，计算公式如下：
    dst[i] = src[i] * scalar + dst[i]
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    - tensor前n个数据计算

      .. code-block:: c++

          template <typename T, typename U>
          __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, const int32_t& count)

    - tensor高维切分计算

      - mask逐bit模式

        .. code-block:: c++

            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue,
                                    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)

      - mask连续模式

        .. code-block:: c++

            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue,
                                    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
    """

    param_list = """
    **参数说明**

    - is_set_mask：是否在接口内部设置mask。

      - True，表示在接口内部设置mask。
      - False，表示在接口外部设置mask，开发者需要使用set_vector_mask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。

    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - scalar：源操作数，scalar标量。scalar的数据类型需要和src保持一致。
    - count：参与计算的元素个数。

    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]

    - repeat_time：重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，
      必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeat_time表示迭代的次数。
    - repeat_params：控制操作数地址步长的参数。UnaryRepeatParams类型，包含操作数相邻迭代间相同data_block的地址步长，操作数同一迭代内不同data_block的地址步长等参数。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    使用tensor高维切分计算接口时，src和scalar的数据类型为half、dst的数据类型为float的情况下，
    一个迭代处理的源操作数元素个数需要和目的操作数保持一致，所以每次迭代选取前4个data_block参与计算。
    设置repeat_stride参数和mask参数以及地址重叠时，需要考虑该限制。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.axpy(dst, src, 2.0, mask=128, repeat_time=4, repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.axpy(dst, src, 2.0, mask=mask, repeat_time=4, repeat_params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.axpy(dst, src, 2.0, count=512)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def cast_deq_docstring():
    func_introduction = """
    对输入做量化并进行精度转换。不同的数据类型，转换公式不同。

    - 在输入类型为int16_t的情况下，对int16_t类型的输入做量化并进行精度转换，得到int8_t/uint8_t类型的数据。
      使用该接口前需要调用set_deq_scale设置scale、offset、sign_mode等量化参数。

      通过模板参数is_vec_deq控制是否选择向量量化模式。

      - 当is_vec_deq=false时，根据set_deq_scale设置的scale、offset、sign_mode，对输入做量化并进行精度转换。计算公式如下：
        dst[i] = (src[i] * scale) + offset
      - 当is_vec_deq=true时，根据set_deq_scale设置的一段128B的UB上的16组量化参数scale[0]-scale[15]、offset[0]-offset[15]、sign_mode[0]-sign_mode[15]，
        以循环的方式对输入做量化并进行精度转换。计算公式如下：dst[i] = (src[i] * scale[j]) + offset[j], 0<=j<=15

    - 在输入类型为int32_t的情况下，对int32_t类型的输入做量化并进行精度转换，得到half类型的数据。使用该接口前需要调用set_deq_scale设置scale参数。
      dst[i] = src[i] * scale
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    - tensor前n个数据计算

      .. code-block:: c++

          template <typename T, typename U, bool isVecDeq = true, bool halfBlock = true>
          __aicore__ inline void CastDeq(const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint32_t count)

    - tensor高维切分计算

      - mask逐bit模式

        .. code-block:: c++

            template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
            __aicore__ inline void CastDeq(const LocalTensor<T>& dst, const LocalTensor<U>& src,
                                    const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)

      - mask连续模式

        .. code-block:: c++

            template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
            __aicore__ inline void CastDeq(const LocalTensor<T>& dst, const LocalTensor<U>& src,
                                    const int32_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
    """

    param_list = """
    **参数说明**

    - is_set_mask：是否在接口内部设置mask。

      - True，表示在接口内部设置mask。
      - False，表示在接口外部设置mask，开发者需要使用set_vector_mask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。

    - is_vec_deq：控制是否选择向量量化模式。和set_deq_scale接口配合使用，当set_deq_scale接口传入Tensor时，is_vec_deq必须为true。
    - half_block：对int16_t类型的输入做量化并进行精度转换得到int8_t/uint8_t类型的数据时，half_block参数用于指示输出元素存放在上半还是下半Block。half_block=True时，结果存放在下半Block；half_block=False时，结果存放在上半Block。
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - count：参与计算的元素个数。

    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]

    - repeat_time：重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，
      为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeat_time表示迭代的次数。
    - repeat_params：控制操作数地址步长的参数。UnaryRepeatParams类型，包含操作数相邻迭代间相同data_block的地址步长，操作数同一迭代内不同data_block的地址步长等参数。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.cast_deq(dst, src, mask=128, repeat_time=4, repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.cast_deq(dst, src, mask=mask, repeat_time=4, repeat_params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.cast_deq(dst, src, count=512)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def gather_docstring():
    func_introduction = """
    给定输入的张量和一个地址偏移张量，本接口根据偏移地址将输入张量按元素收集到结果张量中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    - tensor前n个数据计算

      .. code-block:: c++

          template <typename T>
          __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src,
            const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr, const uint32_t count)

    - tensor高维切分计算

      - mask逐bit模式

        .. code-block:: c++

            template <typename T>
            __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr,
                                const uint64_t mask[], const uint8_t repeatTime, const uint16_t dstRepStride)

      - mask连续模式

        .. code-block:: c++

            template <typename T>
            __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr,
                                const uint64_t mask, const uint8_t repeatTime, const uint16_t dstRepStride)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - src_offset：每个元素在src中对应的地址偏移。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
      该偏移量相对于src的起始基地址而言。单位为Bytes。取值要求如下：
      - 取值应保证src元素类型位宽对齐。
      - 偏移地址后不能超出UB大小数据的范围。
      - 地址偏移的取值范围：不能超出uint32_t的范围。
    - src_base：src的起始基地址，用于指定Gather操作中源操作数的起始位置，单位为Bytes。取值应保证src元素类型位宽对齐，否则会导致非预期行为。
    - count：执行处理的数据个数。

    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]

    - repeat_time：指令迭代次数，每次迭代完成8个datablock（32Bytes）的数据收集，数据范围：repeat_time∈[0,255]。
    - dst_rep_stride：相邻迭代间的地址步长，单位是datablock（32Bytes）。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        src_offset = asc.LocalTensor(dtype=asc.uint32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        asc.gather(z_local, x_local, src_offset, src_base=0, count=512)
        asc.gather(z_local, x_local, src_offset, src_base=0, mask=512, repeat_times=1, dst_rep_stride=8)
        uint64_max = 2**64 - 1
        mask_bits = [uint64_max, uint64_max]
        asc.gather(z_local, x_local, src_offset, src_base=0, mask=mask_bits, repeat_times=1, dst_rep_stride=8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def gatherb_docstring():
    func_introduction = """
    给定一个输入的张量和一个地址偏移张量，本接口根据偏移地址按照data_block的粒度将输入张量收集到结果张量中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void Gatherb(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
            const LocalTensor<uint32_t>& offset, const uint8_t repeatTime, const GatherRepeatParams& repeatParams)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - offset：每个datablock在源操作数中对应的地址偏移。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
      该偏移量是相对于src0的基地址而言的。每个元素值要大于等于0，单位为字节；且需要保证偏移后的地址满足32字节对齐。
    - repeat_time：重复迭代次数，每次迭代完成8个data_block的数据收集，数据范围：repeat_time∈（0,255]。
    - dst_rep_stride：目的操作数相邻迭代间的地址步长。以一个repeat_time归约后的长度为单位。每个repeat_time(8个data_block)归约后，得到8个元素，所以输入类型为half类型时，rep_stride单位为16Byte；输入类型为float类型时，rep_stride单位为32Byte。
    """

    constraint_list = """
    **约束说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        params = asc.GatherRepeatParams(1, 8)
        asc.gatherb(y_buf, x_buf, offset_buf, repeat_time=2, repeat_params=params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def block_reduce_sum_docstring():
    func_introduction = """
    对每个datablock内所有元素求和。源操作数相加采用二叉树方式，两两相加。归约指令的总体介绍请参考如何使用归约计算API。
    以128个half类型的数据求和为例，每个datablock可以计算16个half类型数据，分成8个datablock进行计算；每个datablock内，通过二叉树的方式，两两相加。
    需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。
    例如，源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，然后计算-30000+100=-29900，最后计算65504-29900=35604。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    - mask逐比特模式

      .. code-block:: c++

          template <typename T, bool isSetMask = true>
          __aicore__ inline void BlockReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime,
            const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)

    - mask连续模式

      .. code-block:: c++

          template <typename T, bool isSetMask = true>
          __aicore__ inline void BlockReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,const int32_t repeatTime,
            const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
    """

    param_list = """
    **参数说明**

    - is_set_mask: 是否在接口内部设置mask。
      - True，表示在接口内部设置mask。
      - False，表示在接口外部设置mask，开发者需要使用set_vector_mask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要保证16字节对齐（针对half数据类型），32字节对齐（针对float数据类型）。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - repeat_time：迭代次数。取值范围为[0, 255]。
    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]
    - dst_rep_stride：目的操作数相邻迭代间的地址步长。以一个repeat_time归约后的长度为单位。每个repeat_time(8个datablock)归约后，得到8个元素，所以输入类型为half类型时，rep_stride单位为16Byte；输入类型为float类型时，rep_stride单位为32Byte。
    - src_blk_stride：单次迭代内datablock的地址步长。
    - src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 为了节省地址空间，您可以定义一个Tensor，供源操作数与目的操作数同时使用（即地址重叠），需要注意计算后的目的操作数数据不能覆盖未参与计算的源操作数，需要谨慎使用。
    """

    py_example = """
    **调用示例**

    - mask连续模式

      .. code-block:: python

          asc.block_reduce_sum(z_local, x_local, repeat=1, mask=128, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)

    - mask逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.block_reduce_sum(z_local, x_local, repeat=1, mask=mask, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def block_reduce_max_docstring():
    func_introduction = """
    对每个datablock内所有元素求最大值。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    - mask逐比特模式

      .. code-block:: c++

          template <typename T, bool isSetMask = true>
          __aicore__ inline void BlockReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime,
            const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)

    - mask连续模式

      .. code-block:: c++

          template <typename T, bool isSetMask = true>
          __aicore__ inline void BlockReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,const int32_t repeatTime,
            const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
    """

    param_list = """
    **参数说明**

    - is_set_mask: 是否在接口内部设置mask。
      - True，表示在接口内部设置mask。
      - False，表示在接口外部设置mask，开发者需要使用set_vector_mask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要保证16字节对齐（针对half数据类型），32字节对齐（针对float数据类型）。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - repeat_time：迭代次数。取值范围为[0, 255]。
    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]
    - dst_rep_stride：目的操作数相邻迭代间的地址步长。以一个repeat_time归约后的长度为单位。每个repeat_time(8个datablock)归约后，得到8个元素，所以输入类型为half类型时，RepStride单位为16Byte；输入类型为float类型时，RepStride单位为32Byte。
    - src_blk_stride：单次迭代内datablock的地址步长。
    - src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 为了节省地址空间，您可以定义一个Tensor，供源操作数与目的操作数同时使用（即地址重叠），需要注意计算后的目的操作数数据不能覆盖未参与计算的源操作数，需要谨慎使用。
    - 针对不同场景合理使用归约指令可以带来性能提升, 相关介绍请参考选择低延迟指令，优化归约操作性能。
    """

    py_example = """
    **调用示例**

    - mask连续模式

      .. code-block:: python

          asc.block_reduce_max(z_local, x_local, repeat=1, mask=128, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)

    - mask逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.block_reduce_max(z_local, x_local, repeat=1, mask=mask, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def block_reduce_min_docstring():
    func_introduction = """
    对每个datablock内所有元素求最小值。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    - mask逐比特模式

      .. code-block:: c++

          template <typename T, bool isSetMask = true>
          __aicore__ inline void BlockReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime,
            const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)

    - mask连续模式

      .. code-block:: c++

          template <typename T, bool isSetMask = true>
          __aicore__ inline void BlockReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src,const int32_t repeatTime,
            const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
    """

    param_list = """
    **参数说明**

    - is_set_mask: 是否在接口内部设置mask。
      - True，表示在接口内部设置mask。
      - False，表示在接口外部设置mask，开发者需要使用set_vector_mask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要保证16字节对齐（针对half数据类型），32字节对齐（针对float数据类型）。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - repeat_time：迭代次数。取值范围为[0, 255]。
    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]
    - dst_rep_stride：目的操作数相邻迭代间的地址步长。以一个repeat_time归约后的长度为单位。每个repeat_time(8个datablock)归约后，得到8个元素，所以输入类型为half类型时，RepStride单位为16Byte；输入类型为float类型时，RepStride单位为32Byte。
    - src_blk_stride：单次迭代内datablock的地址步长。
    - src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 为了节省地址空间，您可以定义一个Tensor，供源操作数与目的操作数同时使用（即地址重叠），需要注意计算后的目的操作数数据不能覆盖未参与计算的源操作数，需要谨慎使用。
    - 针对不同场景合理使用归约指令可以带来性能提升, 相关介绍请参考选择低延迟指令，优化归约操作性能。
    """

    py_example = """
    **调用示例**

    - mask连续模式

      .. code-block:: python

          asc.block_reduce_min(z_local, x_local, repeat=1, mask=128, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)

    - mask逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.block_reduce_min(z_local, x_local, repeat=1, mask=mask, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def reduce_min_docstring():
    func_introduction = """
    在所有的输入数据中找出最小值及最小值对应的索引位置。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    - tensor前n个数据计算

      .. code-block:: c++

          template <typename T>
          __aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src,
            const LocalTensor<T>& sharedTmpBuffer, const int32_t count, bool calIndex = 0)

    - tensor高维切分计算

      - mask逐比特模式

        .. code-block:: c++

          template <typename T>
          __aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                            const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[],
                                            const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)

      - mask连续模式

        .. code-block:: c++

          template <typename T>
          __aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                            const LocalTensor<T>& sharedTmpBuffer, const int32_t mask,
                                            const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要保证4字节对齐（针对half数据类型），8字节对齐（针对float数据类型）。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。源操作数的数据类型需要与目的操作数保持一致。
    - shared_tmp_buffer：API执行期间，部分硬件型号需要一块空间用于存储中间结果，空间大小需要满足最小所需空间的要求。
      类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型需要与目的操作数保持一致。
    - count：参与计算的元素个数。
      参数取值范围和操作数的数据类型有关，数据类型不同，能够处理的元素个数最大值不同，最大处理的数据量不能超过UB大小限制。
    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]
    - repeat_time：迭代次数。与通用参数说明中不同的是，支持更大的取值范围，保证不超过int32_t最大值的范围即可。
    - src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。
    - cal_index：指定是否获取最小值的索引，bool类型，默认值为false，取值：
      - True：同时获取最小值和最小值索引。
      - False：不获取索引，只获取最小值。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。需要使用shared_tmp_buffer的情况下，支持dst与shared_tmp_buffer地址重叠（通常情况下dst比shared_tmp_buffer所需的空间要小），此时shared_tmp_buffer必须满足最小所需空间要求，否则不支持地址重叠。
    - dst结果存储顺序为最小值，最小值索引，若不需要索引，只会存储最小值。返回结果中索引index数据是按照dst的数据类型进行存储的，比如dst使用half类型时，index按照half类型进行存储，如果按照half格式进行读取，index的值是不对的，因此index的读取需要使用reinterpret_cast方法转换到整数类型。
    - 返回最小值索引时，如果存在多个最小值，返回第一个最小值的索引。
    - 当输入类型是half的时候，只支持获取最大不超过65535（uint16_t能表示的最大值）的索引值。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          asc.reduce_min(dst, src, shared_tmp_buffer=shared_tmp, mask=128, repeat_time=128, src_rep_stride=65, cal_index=True)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.reduce_min(dst, src, shared_tmp_buffer=shared_tmp, mask=mask, repeat_time=65, src_rep_stride=8, cal_index=True)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.reduce_min(dst, src, shared_tmp_buffer=shared_tmp, count=2048, cal_index=True)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def reduce_max_docstring():
    func_introduction = """
    在所有的输入数据中找出最小值及最小值对应的索引位置。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    - tensor前n个数据计算

      .. code-block:: c++

          template <typename T>
          __aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
            const LocalTensor<T>& sharedTmpBuffer, const int32_t count, bool calIndex = 0)

    - tensor高维切分计算

      - mask逐比特模式

        .. code-block:: c++

          template <typename T>
          __aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                            const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[],
                                            const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)

      - mask连续模式

        .. code-block:: c++

          template <typename T>
          __aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                            const LocalTensor<T>& sharedTmpBuffer, const int32_t mask,
                                            const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要保证4字节对齐（针对half数据类型），8字节对齐（针对float数据类型）。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。源操作数的数据类型需要与目的操作数保持一致。
    - shared_tmp_buffer：API执行期间，部分硬件型号需要一块空间用于存储中间结果，空间大小需要满足最小所需空间的要求。
      类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型需要与目的操作数保持一致。
    - count：参与计算的元素个数。
      参数取值范围和操作数的数据类型有关，数据类型不同，能够处理的元素个数最大值不同，最大处理的数据量不能超过UB大小限制。
    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]
    - repeat_time：迭代次数。与通用参数说明中不同的是，支持更大的取值范围，保证不超过int32_t最大值的范围即可。
    - src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。
    - cal_index：指定是否获取最小值的索引，bool类型，默认值为false，取值：
      - True：同时获取最小值和最小值索引。
      - False：不获取索引，只获取最小值。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。需要使用shared_tmp_buffer的情况下，支持dst与shared_tmp_buffer地址重叠（通常情况下dst比shared_tmp_buffer所需的空间要小），此时shared_tmp_buffer必须满足最小所需空间要求，否则不支持地址重叠。
    - dst结果存储顺序为最大值，最大值索引，若不需要索引，只会存储最大值。返回结果中索引index数据是按照dst的数据类型进行存储的，比如dst使用half类型时，index按照half类型进行存储，如果按照half格式进行读取，index的值是不对的，因此index的读取需要使用reinterpret_cast方法转换到整数类型。
    - 返回最大值索引时，如果存在多个最大值，返回第一个最大值的索引。
    - 当输入类型是half的时候，只支持获取最大不超过65535（uint16_t能表示的最大值）的索引值。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          asc.reduce_max(dst, src, shared_tmp_buffer=shared_tmp, mask=128, repeat_time=128, src_rep_stride=65, cal_index=True)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.reduce_max(dst, src, shared_tmp_buffer=shared_tmp, mask=mask, repeat_time=65, src_rep_stride=8, cal_index=True)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.reduce_max(dst, src, shared_tmp_buffer=shared_tmp, count=2048, cal_index=True)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def reduce_sum_docstring():
    func_introduction = """
    对所有的输入数据求和。ReduceSum的相加方式分为两种：
    - 方式一：同一repeat内先按照二叉树累加、不同repeat的结果也按照二叉树累加。
    - 方式二：同一repeat内采用二叉树累加，不同repeat的结果按顺序累加。
    tensor前n个数据计算接口采用方式二，tensor高维切分计算接口采用方式一。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    - tensor前n个数据计算

      .. code-block:: c++

          template <typename T, bool isSetMask = true>
          __aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
            const LocalTensor<T>& sharedTmpBuffer, const int32_t count, bool calIndex = 0)

    - tensor高维切分计算

      - mask逐比特模式

        .. code-block:: c++

          template <typename T, bool isSetMask = true>
          __aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                            const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[],
                                            const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)

      - mask连续模式

        .. code-block:: c++

          template <typename T, bool isSetMask = true>
          __aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                            const LocalTensor<T>& sharedTmpBuffer, const int32_t mask,
                                            const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)
    """

    param_list = """
    **参数说明**

    - is_set_mask：预留参数，为后续的功能做保留。保持默认值即可。
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要保证2字节对齐（针对half数据类型），4字节对齐（针对float数据类型）。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。源操作数的数据类型需要与目的操作数保持一致。
    - shared_tmp_buffer：API执行期间，部分硬件型号需要一块空间用于存储中间结果，空间大小需要满足最小所需空间的要求。
      类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型需要与目的操作数保持一致。
    - count：参与计算的元素个数。
      参数取值范围和操作数的数据类型有关，数据类型不同，能够处理的元素个数最大值不同，最大处理的数据量不能超过UB大小限制。
    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]
    - repeat_time：迭代次数。与通用参数说明中不同的是，支持更大的取值范围，保证不超过int32_t最大值的范围即可。
    - src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 操作数地址重叠约束请参考 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址重叠约束”。需要使用shared_tmp_buffer的情况下，支持dst与shared_tmp_buffer地址重叠（通常情况下dst比shared_tmp_buffer所需的空间要小），此时shared_tmp_buffer必须满足最小所需空间要求，否则不支持地址重叠。
    - 该接口内部通过软件仿真来实现reduce_sum功能，某些场景下，性能可能不及直接使用硬件指令实现的block_reduce_sum和whole_reduce_sum接口。针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考选择低延迟指令，优化归约操作性能。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          asc.reduce_sum(dst, src, shared_tmp_buffer=shared_tmp, mask=128, repeat_time=128, src_rep_stride=65, cal_index=True)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.reduce_sum(dst, src, shared_tmp_buffer=shared_tmp, mask=mask, repeat_time=65, src_rep_stride=8, cal_index=True)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.reduce_sum(dst, src, shared_tmp_buffer=shared_tmp, count=2048, cal_index=True)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def init_soc_state_docstring():
    func_introduction = """
    在由于AI Core上存在一些全局状态，如原子累加状态、Mask模式等，在实际运行中，这些值可以被前序执行的算子修改而导致计算出现不符合预期的行为，在静态Tensor编程的场景中用户必须在Kernel入口处调用此函数来初始化AI Core状态 。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void InitSocState()

    """

    param_list = """
    **参数说明**

    无。
    """

    constraint_list = """
    **约束说明**

    不调用该接口，在部分场景下可能导致计算结果出现精度错误或者卡死等问题。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.init_soc_state()

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_store_atomic_config_docstring():
    func_introduction = """
    设置原子操作使能位与原子操作类型。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <AtomicDtype type, AtomicOp op>
        __aicore__ inline void SetStoreAtomicConfig()


    """

    param_list = """
    **参数说明**

    - type：原子操作使能位，AtomicDtype枚举类的定义如下：

      .. code-block:: python

          class AtomicDtype(IntEnum):
              ATOMIC_NONE = 0   // 无原子操作
              ATOMIC_F32 = 1    // 使能原子操作，进行原子操作的数据类型为float
              ATOMIC_F16 = 2    // 使能原子操作，进行原子操作的数据类型为half
              ATOMIC_S16 = 3    // 使能原子操作，进行原子操作的数据类型为int16_t
              ATOMIC_S32 = 4    // 使能原子操作，进行原子操作的数据类型为int32_t
              ATOMIC_S8 = 5     // 使能原子操作，进行原子操作的数据类型为int8_t
              ATOMIC_BF16 = 6   // 使能原子操作，进行原子操作的数据类型为bfloat16_t

    - op：原子操作类型，仅当使能原子操作时有效（即“type”为非“ATOMIC_NONE”的场景），当前仅支持求和操作。

      .. code-block:: python

          class AtomicOp(IntEnum):
              ATOMIC_SUM = 0    // 求和操作

    """

    constraint_list = """
    **约束说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.set_store_atomic_config(asc.AtomicDtype.ATOMIC_F16, asc.AtomicOp.ATOMIC_SUM)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_store_atomic_config_docstring():
    func_introduction = """
    获取原子操作使能位与原子操作类型的值。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void GetStoreAtomicConfig(uint16_t& atomicType, uint16_t& atomicOp)

    """

    param_list = """
    **参数说明**

    无。

    """

    return_list = """
    **返回值说明**

    - atomic_type（第一个返回值）：原子操作使能位。

      - 0：无原子操作
      - 1：使能原子操作，进行原子操作的数据类型为float
      - 2：使能原子操作，进行原子操作的数据类型为half
      - 3：使能原子操作，进行原子操作的数据类型为int16_t
      - 4：使能原子操作，进行原子操作的数据类型为int32_t
      - 5：使能原子操作，进行原子操作的数据类型为int8_t
      - 6：使能原子操作，进行原子操作的数据类型为bfloat16_t

    - atomic_op（第二个返回值）：原子操作类型。

      - 0：求和操作
    """

    constraint_list = """
    **约束说明**

    此接口需要与set_store_atomic_config(ISASI)配合使用，用以获取原子操作使能位与原子操作类型的值。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.set_store_atomic_config(asc.AtomicDtype.ATOMIC_F16, asc.AtomicOp.ATOMIC_SUM)
        atomic_type, atomic_op = asc.get_store_atomic_config()

    """

    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def check_local_memory_ia_docstring():
    func_introduction = """
    Check设定范围内的UB读写行为，如果有设定范围的读写行为则会出现EXCEPTION报错，无设定范围的读写行为则不会报错。

    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void CheckLocalMemoryIA(const CheckLocalMemoryIAParam& checkParams)

    """

    param_list = """
    **参数说明**

    - check_params：用于配置对UB访问的检查行为，类型为CheckLocalMemoryIAParam。
      - enable_bit：配置的异常寄存器，取值范围：enable_bit∈[0,3]，默认为0。
        - 0：异常寄存器0。
        - 1：异常寄存器1。
        - 2：异常寄存器2。
        - 3：异常寄存器3。
      - start_addr：Check的起始地址，32B对齐，取值范围：start_addr∈[0, 65535]，默认值为0。比如，可通过LocalTensor.get_phy_addr()/32来获取start_addr。
      - end_addr：Check的结束地址，32B对齐，取值范围：end_addr∈[0, 65535]。默认值为0。
      - is_scalar_read：Check标量读访问。
        - false：不开启，默认为false。
        - true：开启。
      - is_scalar_write： Check标量写访问。
        - false：不开启，默认为false。
        - true：开启。
      - is_vector_read： Check矢量读访问。
        - false：不开启，默认为false。
        - true：开启。
      - is_vector_write： Check矢量写访问。
        - false：不开启，默认为false。
        - true：开启。
      - is_mte_read： Check MTE读访问。
        - false：不开启，默认为false。
        - true：开启。
      - is_mte_write： Check MTE写访问。
        - false：不开启，默认为false。
        - true：开启。
      - is_enable： 是否使能enable_bit参数配置的异常寄存器。
        - false：不使能，默认为false。
        - true：使能。

    """
    constraint_list = """
    **约束说明**

    - start_addr/end_addr的单位是32B，check的范围不包含start_addr，包含end_addr，即(start_addr, end_addr]。
    - 每次调用完该接口需要进行复位（配置is_enable为False进行复位）。

    """
    py_example = """
    **调用示例**

    .. code-block:: python

        params = asc.CheckLocalMemoryIAParam()
        asc.check_local_memory_ia(params)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


DOC_HANDLES = {
    "axpy": axpy_docstring,
    "block_reduce_sum": block_reduce_sum_docstring,
    "block_reduce_max": block_reduce_max_docstring,
    "block_reduce_min": block_reduce_min_docstring,
    "brcb": brcb_docstring,
    "cast": cast_docstring,
    "cast_deq": cast_deq_docstring,
    "check_local_memory_ia": check_local_memory_ia_docstring,
    "compare": compare_docstring,
    "compare_scalar": compare_scalar_docstring,
    "copy": copy_docstring,
    "count_bits_cnt_same_as_sign_bit": count_bits_cnt_same_as_sign_bit_docstring,
    "cross_core_set_flag": cross_core_set_flag_docstring,
    "cross_core_wait_flag": cross_core_wait_flag_docstring,
    "data_cache_clean_and_invalid": data_cache_clean_and_invalid_docstring,
    "data_cache_preload": data_cache_preload_docstring,
    "data_copy": data_copy_docstring,
    "data_copy_pad": data_copy_pad_docstring,
    "data_sync_barrier": data_sync_barrier_docstring,
    "dump_acc_chk_point": dump_acc_chk_point_docstring,
    "dump_tensor": dump_tensor_docstring,
    "duplicate": duplicate_docstring,
    "fixpipe": fixpipe_docstring,
    "gather": gather_docstring,
    "gatherb": gatherb_docstring,
    "gather_mask": gather_mask_docstring,
    "get_block_idx": get_block_idx_docstring,
    "get_block_num": get_block_num_docstring,
    "get_cmp_mask": get_cmp_mask_docstring,
    "get_data_block_size_in_bytes": get_data_block_size_in_bytes_docstring,
    "get_hccl_context": get_hccl_context_docstring,
    "get_icache_preload_status": get_icache_preload_status_docstring,
    "get_mrg_sort_result": get_mrg_sort_result_docstring,
    "get_program_counter": get_program_counter_docstring,
    "get_store_atomic_config": get_store_atomic_config_docstring,
    "get_sort_len": get_sort_len_docstring,
    "get_sort_offset": get_sort_offset_docstring,
    "get_sub_block_idx": get_sub_block_idx_docstring,
    "get_sys_workspace": get_sys_workspace_docstring,
    "get_system_cycle": get_system_cycle_docstring,
    "get_task_ratio": get_task_ratio_docstring,
    "ib_set": ib_set_docstring,
    "ib_wait": ib_wait_docstring,
    "icache_preload": icache_preload_docstring,
    "init_const_value": init_const_value_docstring,
    "init_soc_state": init_soc_state_docstring,
    "load_data": load_data_docstring,
    "load_data_with_sparse": load_data_with_sparse_docstring,
    "load_data_with_transpose": load_data_with_transpose_docstring,
    "load_image_to_local": load_image_to_local_docstring,
    "metrics_prof_start": metrics_prof_start_docstring,
    "metrics_prof_stop": metrics_prof_stop_docstring,
    "mmad": mmad_docstring,
    "mmad_with_sparse": mmad_with_sparse_docstring,
    "mrg_sort": mrg_sort_docstring,
    "mrg_sort4": mrg_sort4_docstring,
    "notify_next_block": notify_next_block_docstring,
    "pair_reduce_sum": pair_reduce_sum_docstring,
    "pipe_barrier": pipe_barrier_docstring,
    "print_time_stamp": print_time_stamp_docstring,
    "printf": printf_docstring,
    "proposal_concat": proposal_concat_docstring,
    "proposal_extract": proposal_extract_docstring,
    "reduce_sum": reduce_sum_docstring,
    "reduce_max": reduce_max_docstring,
    "reduce_min": reduce_min_docstring,
    "repeat_reduce_sum": repeat_reduce_sum_docstring,
    "reset_mask": reset_mask_docstring,
    "rp_sort16": rp_sort16_docstring,
    "scalar_cast": scalar_cast_docstring,
    "scalar_count_leading_zero": scalar_count_leading_zero_docstring,
    "scalar_get_count_of_value": scalar_get_count_of_value_docstring,
    "scalar_get_sff_value": scalar_get_sff_value_docstring,
    "scatter": scatter_docstring,
    "select": select_docstring,
    "set_aipp_functions": set_aipp_functions_docstring,
    "set_atomic_add": set_atomic_add_docstring,
    "set_store_atomic_config": set_store_atomic_config_docstring,
    "set_atomic_max": set_atomic_max_docstring,
    "set_atomic_min": set_atomic_min_docstring,
    "set_atomic_none": set_atomic_none_docstring,
    "set_atomic_type": set_atomic_type_docstring,
    "set_cmp_mask": set_cmp_mask_docstring,
    "set_deq_scale": set_deq_scale_docstring,
    "set_flag": set_wait_flag_docstring,
    "set_fix_pipe_pre_quant_flag": set_fix_pipe_pre_quant_flag_docstring,
    "set_hccl_context": set_hccl_context_docstring,
    "set_hf32_mode": set_hf32_mode_docstring,
    "set_hf32_trans_mode": set_hf32_trans_mode_docstring,
    "set_load_data_boundary": set_load_data_boundary_docstring,
    "set_load_data_padding_value": set_load_data_padding_value_docstring,
    "set_load_data_repeat": set_load_data_repeat_docstring,
    "set_mask_count": set_mask_count_docstring,
    "set_mask_norm": set_mask_norm_docstring,
    "set_mm_layout_transform": set_mm_layout_transform_docstring,
    "set_pad_value": set_pad_value_docstring,
    "set_vector_mask": set_vector_mask_docstring,
    "sort": sort_docstring,
    "sort32": sort32_docstring,
    "sync_all": sync_all_docstring,
    "trans_data_to_5hd": trans_data_to_5hd_docstring,
    "transpose": transpose_docstring,
    "trap": trap_docstring,
    "wait_flag": set_wait_flag_docstring,
    "wait_pre_block": wait_pre_block_docstring,
    "whole_reduce_max": whole_reduce_max_docstring,
    "whole_reduce_min": whole_reduce_min_docstring,
    "whole_reduce_sum": whole_reduce_sum_docstring,
}


def set_common_docstring(api_name: Optional[str] = None) -> Callable[[T], T]:
    func_introduction = ""
    cpp_signature = ""
    param_list = ""
    return_list = ""
    constraint_list = ""
    py_example = ""

    if DOC_HANDLES.get(api_name) is None:
        raise RuntimeError(f"Invalid api name {api_name}")

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


class TensorDescDocstring:

    def __init__(self) -> None:
        ...

    @staticmethod
    def set_shape_addr_docstring():
        func_introduction = """
        配置用于储存shape信息的地址。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            void SetShapeAddr(uint64_t* shapePtr)
        """

        param_list = """
        **参数说明**

        - shape_ptr：用于储存shape信息的地址。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            tensor_desc = asc.TensorDesc()
            tensor_desc.set_shape_addr(0)
        """

        return [func_introduction, cpp_signature, param_list, "", "", py_example]

    @staticmethod
    def get_dim_docstring():
        func_introduction = """
        获取Tensor的维度。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            uint64_t GetDim()
        """

        return_list = """
        **返回值说明**

        返回Tensor的维度。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            tensor_desc = asc.TensorDesc()
            dim = tensor_desc.get_dim()
        """

        return [func_introduction, cpp_signature, "", return_list, "", py_example]

    @staticmethod
    def get_index_docstring():
        func_introduction = """
        获取TensorDesc在ListTensorDesc中对应的索引值。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            uint64_t GetIndex()
        """

        return_list = """
        **返回值说明**

        返回TensorDesc在ListTensorDesc中对应的索引值。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            tensor_desc = asc.TensorDesc()
            index = tensor_desc.get_index()
        """

        return [func_introduction, cpp_signature, "", return_list, "", py_example]

    @staticmethod
    def get_shape_docstring():
        func_introduction = """
        获取对应维度的shape信息。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            uint64_t GetShape(uint32_t offset)
        """

        param_list = """
        **参数说明**

        - offset：输入索引值。
        """

        return_list = """
        **返回值说明**

        返回对应维度的shape信息。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            tensor_desc = asc.TensorDesc()
            offset = 0
            shape = tensor_desc.get_shape
        """

        return [func_introduction, cpp_signature, param_list, return_list, "", py_example]

    @staticmethod
    def get_data_ptr_docstring():
        func_introduction = """
        获取储存Tensor数据地址。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            T* GetDataPtr()
        """

        return_list = """
        **返回值说明**

        返回储存Tensor数据地址。T数据类型。。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            tensor_desc = asc.TensorDesc()
            data_ptr = tensor_desc.get_data_ptr()
        """

        return [func_introduction, cpp_signature, "", return_list, "", py_example]

    @staticmethod
    def get_data_obj_docstring():
        func_introduction = """
        将数据指针置于GlobalTensor中并返回该GlobalTensor。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            GlobalTensor<T> GetDataObj()
        """

        return_list = """
        **返回值说明**

        返回设置了数据指针的GlobalTensor。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            tensor_desc = asc.TensorDesc()
            data_obj = tensor_desc.get_data_obj()
        """

        return [func_introduction, cpp_signature, "", return_list, "", py_example]


TENSOR_DOC_HANDLERS = {
    "TensorDesc": {
        "set_shape_addr": TensorDescDocstring.set_shape_addr_docstring,
        "get_dim": TensorDescDocstring.get_dim_docstring,
        "get_index": TensorDescDocstring.get_index_docstring,
        "get_shape": TensorDescDocstring.get_shape_docstring,
        "get_data_ptr": TensorDescDocstring.get_data_ptr_docstring,
        "get_data_obj": TensorDescDocstring.get_data_obj_docstring,
    },
    "ListTensorDesc": {
        "init": ListTensorDescDocstring.init_docstring,
        "get_desc": ListTensorDescDocstring.get_desc_docstring,
        "get_data_ptr": ListTensorDescDocstring.get_data_ptr_docstring,
        "get_size": ListTensorDescDocstring.get_size_docstring,
    },
}


def set_tensor_docstring(tensor_name: Optional[str] = None, api_name: Optional[str] = None) -> Callable[[T], T]:
    func_introduction = ""
    cpp_signature = ""
    param_list = ""
    return_list = ""
    constraint_list = ""
    py_example = ""
    if TENSOR_DOC_HANDLERS.get(tensor_name) is None:
        raise RuntimeError(f"Invalid tensor name {tensor_name}")
    if TENSOR_DOC_HANDLERS.get(tensor_name, {}).get(api_name) is None:
        raise RuntimeError(f"Unsupported API [{api_name}] for tensor type [{tensor_name}]")
    handler = TENSOR_DOC_HANDLERS.get(tensor_name, {}).get(api_name)
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
