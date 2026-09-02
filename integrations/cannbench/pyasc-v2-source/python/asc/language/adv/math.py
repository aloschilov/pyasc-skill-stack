# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional, Tuple, Union, overload

from ..core.dtype import KnownTypes
from ..core.ir_value import RuntimeInt, RuntimeNumeric, \
                            materialize_ir_value as _mat, RuntimeBool
from ..core.tensor import LocalTensor
from ..core.utils import check_type, require_jit, global_builder
from .utils import set_math_docstring


def math_op_impl(tensors: Tuple[LocalTensor], count: Optional[RuntimeInt], temp_buffer: Optional[LocalTensor], \
                 is_reuse_source: RuntimeBool, build_method: str) -> None:
    if count is not None:
        check_type("count", count, RuntimeInt)
        count = _mat(count, KnownTypes.int32).to_ir()
    if temp_buffer is not None:
        check_type("temp_buffer", temp_buffer, LocalTensor)
        temp_buffer = temp_buffer.to_ir()
    is_reuse_source = _mat(is_reuse_source, KnownTypes.bit).to_ir()
    getattr(global_builder.get_ir_builder(), build_method)(*(t.to_ir() for t in tensors), sharedTmpBuffer=temp_buffer,
                                                           calCount=count, isReuseSource=is_reuse_source)


@overload
def acos(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
         is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Acos", append_text="按元素做反余弦函数计算。")
def acos(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_AcosOp")


@overload
def acosh(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
          is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Acosh", append_text="按元素做双曲反余弦函数计算。")
def acosh(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
          temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_AcoshOp")


@overload
def asin(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
         is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Asin", append_text="按元素做反正弦函数计算。")
def asin(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_AsinOp")


@overload
def asinh(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
          is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Asinh", append_text="按元素做反双曲正弦函数计算。")
def asinh(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
          temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_AsinhOp")


@overload
def atan(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
         is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Atan", append_text="按元素做三角函数反正切运算。")
def atan(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_AtanOp")


@overload
def atanh(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
          is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Atanh", append_text="按元素做反双曲正切余弦函数计算。")
def atanh(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
          temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_AtanhOp")


@overload
def ceil(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
         is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Ceil", append_text="获取大于或等于x的最小的整数值，即向正无穷取整操作。")
def ceil(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_CeilOp")


@overload
def cos(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
        is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Cos", append_text="按元素做三角函数余弦运算。")
def cos(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
        temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_CosOp")


@overload
def cosh(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
         is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Cosh", append_text="按元素做双曲余弦函数计算。")
def cosh(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_CoshOp")


@overload
def digamma(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
            is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Digamma", append_text="按元素计算x的gamma函数的对数导数。")
def digamma(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
            temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_DigammaOp")


@overload
def erf(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
        is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Erf", append_text="按元素做误差函数计算（也称为高斯误差函数，error function or Gauss error function）。")
def erf(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
        temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_ErfOp")


@overload
def erfc(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
         is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Erfc", append_text="返回输入x的互补误差函数结果，积分区间为x到无穷大。")
def erfc(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_ErfcOp")


@overload
def exp(dst: LocalTensor, src: LocalTensor, count: int, taylor_expand_level: int,
        temp_buffer: Optional[LocalTensor] = None, is_reuse_source: bool = False) -> None:
    ...


@require_jit
def exp(dst: LocalTensor, src: LocalTensor, count: RuntimeInt, taylor_expand_level: RuntimeInt, \
        temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    """
    按元素取自然指数，用户可以选择是否使用泰勒展开公式进行计算。

    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, uint8_t taylorExpandLevel, bool isReuseSource = false>
        __aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, 
                                    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)

        template <typename T, uint8_t taylorExpandLevel, bool isReuseSource = false>
        __aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, 
                                    const uint32_t calCount)

    **参数说明**

    - taylor_expand_level：泰勒展开项数，项数为0表示不使用泰勒公式进行计算。项数太少时，精度会有一定误差。项数越多，精度相对而言更高，但是性能会更差。
    - is_reuse_source：是否允许修改源操作数，默认值为false。该参数仅在输入的数据类型为float时生效。
    - dst：目的操作数。
    - src：源操作数。
    - temp_buffer：临时缓存。
    - count：参与计算的元素个数。

    **约束说明**

    - 不支持源操作数与目的操作数地址重叠。
    - 不支持temp_buffer与源操作数和目的操作数地址重叠。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。

    **调用示例**

    .. code-block:: python

        pipe = asc.Tpipe()
        tmp_que = asc.TQue(asc.TPosition.VECCALC, 1)
        pipe.init_buffer(que=tmp_que, num=1, len=buffer_size)   # buffer_size 通过Host侧tiling参数获取
        shared_tmp_buffer = tmp_que.alloc_tensor(asc.uint8)
        # 输入tensor长度为1024，算子输入的数据类型为half，实际计算个数为512
        asc.adv.exp(dst, src, count=512, taylor_expand_level=0, temp_buffer=shared_tmp_buffer)
    """

    count = _mat(count, KnownTypes.uint32).to_ir()
    is_reuse_source = _mat(is_reuse_source, KnownTypes.bit).to_ir()
    taylor_expand_level = _mat(taylor_expand_level, KnownTypes.uint8).to_ir()
    if temp_buffer is not None:
        check_type("temp_buffer", temp_buffer, LocalTensor)
        temp_buffer = temp_buffer.to_ir()
    builder = global_builder.get_ir_builder()
    builder.create_asc_ExpOp(dst.to_ir(), src.to_ir(), count, taylor_expand_level, temp_buffer, is_reuse_source)


@overload
def floor(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
          is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Floor", append_text="获取小于或等于x的最小的整数值，即向负无穷取整操作。")
def floor(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
          temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_FloorOp")


@overload
def frac(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
         is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="frac", append_text="按元素做取小数计算。")
def frac(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_FracOp")


@overload
def lgamma(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
           is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Lgamma", append_text="按元素计算x的gamma函数的绝对值并求自然对数。")
def lgamma(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
           temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_LgammaOp")


@overload
def log(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
        is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Log", append_text="按元素以e为底做对数运算。")
def log(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
        temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_LogOp")


@overload
def round(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
          is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Round", append_text="将输入的元素四舍五入到最接近的整数。")
def round(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
          temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_RoundOp")


@overload
def sign(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
         is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Sign", append_text="按元素执行Sign操作，Sign是指返回输入数据的符号，如果为0则返回0，如果为正数则返回1，如果为负数则返回-1。")
def sign(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_SignOp")


@overload
def sin(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
        is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Sin", append_text="按元素做正弦函数计算。")
def sin(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
        temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_SinOp")


@overload
def sinh(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
         is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Sinh", append_text="按元素做双曲正弦函数计算。")
def sinh(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_SinhOp")


@overload
def tan(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
        is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Tan", append_text="按元素做正切函数计算。")
def tan(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
        temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_TanOp")


@overload
def tanh(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
         is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Tanh", append_text="按元素做逻辑回归Tanh。")
def tanh(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_TanhOp")


@overload
def trunc(dst: LocalTensor, src: LocalTensor, count: Optional[int] = None, temp_buffer: Optional[LocalTensor] = None,
          is_reuse_source: bool = False) -> None:
    ...


@require_jit
@set_math_docstring(api_name="Trunc", append_text="按元素做浮点数截断，即向零取整操作。")
def trunc(dst: LocalTensor, src: LocalTensor, count: Optional[RuntimeInt] = None,
          temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    math_op_impl((dst, src), count, temp_buffer, is_reuse_source, "create_asc_TruncOp")


@overload
def power(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: Optional[int] = None,
          temp_buffer: Optional[LocalTensor] = None, is_reuse_source: bool = False) -> None:
    ...


@require_jit
def power(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: Optional[RuntimeInt] = None,
          temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    """
    实现按元素做幂运算功能。

    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, bool isReuseSource = false>
        __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, 
                const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)

        template <typename T, bool isReuseSource = false>
        __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, 
                const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)

        template <typename T, bool isReuseSource = false>
        __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, 
                const LocalTensor<T>& src1Tensor, uint32_t calCount)

        template <typename T, bool isReuseSource = false>
        __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, 
                const LocalTensor<T>& src1Tensor)

    **参数说明**

    - is_reuse_source：是否允许修改源操作数，默认值为false。
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。
    - src1：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。
    - temp_buffer：临时内存空间。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - count：参与计算的元素个数。

    **约束说明**

    - 不支持源操作数与目的操作数地址重叠。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。

    **调用示例**

    .. code-block:: python

        asc.adv.power(dst, src0, src1)

    """

    math_op_impl((dst, src0, src1), count, temp_buffer, is_reuse_source, "create_asc_PowerOp")


@overload
def xor(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: Optional[int] = None,
        temp_buffer: Optional[LocalTensor] = None, is_reuse_source: bool = False) -> None:
    ...


@require_jit
def xor(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: Optional[RuntimeInt] = None,
        temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    """
    按元素执行Xor运算。

    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, bool isReuseSource = false>
        __aicore__ inline void Xor(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, 
            const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)

        template <typename T, bool isReuseSource = false>
         __aicore__ inline void Xor(const LocalTensor<T>& dstTensor, const LocalTensor<T> &src0Tensor, 
            const LocalTensor<T> &src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)

        template <typename T, bool isReuseSource = false>
        __aicore__ inline void Xor(const LocalTensor<T> &dstTensor, const LocalTensor<T> &src0Tensor, 
            const LocalTensor<T> &src1Tensor, const uint32_t calCount)

        template <typename T, bool isReuseSource = false>
        __aicore__ inline void Xor(const LocalTensor<T> &dstTensor, const LocalTensor<T> &src0Tensor, 
            const LocalTensor<T> &src1Tensor)

    **参数说明**

    - is_reuse_source：是否允许修改源操作数，默认值为false。
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。
    - src1：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。
    - temp_buffer：临时内存空间。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - count：参与计算的元素个数。

    **约束说明**

    - 不支持源操作数与目的操作数地址重叠。
    - 当前仅支持ND格式的输入，不支持其他格式。
    - count需要保证小于或等于src0Tensor和src1Tensor和dstTensor存储的元素范围。
    - 对于不带count参数的接口，需要保证src0Tensor和src1Tensor的shape大小相等。
    - 不支持temp_buffer与源操作数和目的操作数地址重叠。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。

    **调用示例**

    ..code-block:: python

        asc.adv.xor(z_local, x_local, y_local)

    """

    math_op_impl((dst, src0, src1), count, temp_buffer, is_reuse_source, "create_asc_XorOp")


@overload
def axpy(dst: LocalTensor, src: LocalTensor, scalar: Union[float, int], count: Optional[int] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: bool = False) -> None:
    ...


@require_jit
def axpy(dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, count: Optional[RuntimeInt] = None,
         temp_buffer: Optional[LocalTensor] = None, is_reuse_source: RuntimeBool = False) -> None:
    """
    源操作数(srcTensor)中每个元素与标量求积后和目的操作数(dstTensor)中的对应元素相加。
    该接口功能同基础API Axpy，区别在于此接口指令是通过Muls和Add组合计算，从而提供更优的精度。

    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U, bool isReuseSource = false>
        __aicore__ inline void Axpy(const LocalTensor<T>& dstTensor, const LocalTensor<U>& srcTensor, 
            const U scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)

    **参数说明**

    - is_reuse_source：是否允许修改源操作数，默认值为false。
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - scalar：scalar标量。支持的数据类型为：half/float。scalar操作数的类型需要和srcTensor保持一致。
    - temp_buffer：临时缓存。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - count：参与计算的元素个数。

    **约束说明**

    - 不支持源操作数与目的操作数地址重叠。
    - 不支持temp_buffer与源操作数和目的操作数地址重叠。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_ 中的“通用说明和约束-通用地址对齐约束”。
    - 该接口支持的精度组合如下：

      - half精度组合：src_local数据类型=half；scalar数据类型=half；dst_local数据类型=half；PAR=128
      - float精度组合：src_local数据类型=float；scalar数据类型=float；dst_local数据类型=float；PAR=64
      - mix精度组合：src_local数据类型=half；scalar数据类型=half；dst_local数据类型=float；PAR=64

    **调用示例**

    .. code-block:: python

        pipe = asc.Tpipe()
        tmp_que = asc.TQue(asc.TPosition.VECCALC, 1)
        pipe.init_buffer(que=tmp_que, num=1, len=buffer_size)   # buffer_size 通过Host侧tiling参数获取
        shared_tmp_buffer = tmp_que.alloc_tensor(asc.uint8)
        # 输入tensor长度为1024，算子输入的数据类型为half，实际计算个数为512
        asc.adv.axpy(dst, src, 3.0, count=512, temp_buffer=shared_tmp_buffer)
    """

    scalar = _mat(scalar, src.dtype)
    math_op_impl((dst, src, scalar), count, temp_buffer, is_reuse_source, "create_asc_AxpyOp")
