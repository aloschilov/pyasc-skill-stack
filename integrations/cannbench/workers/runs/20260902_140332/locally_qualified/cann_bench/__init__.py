"""pyasc asc2 submission package for CANN Bench.

Each module exposes one public callable whose name and signature match the
operator's ``proto.yaml`` schema. All numerical work happens in pyasc asc2
kernels launched directly on torch_npu-owned device buffers (zero-copy via
``Tensor.data_ptr()``).
"""

from .sigmoid import sigmoid

__all__ = ["sigmoid"]
