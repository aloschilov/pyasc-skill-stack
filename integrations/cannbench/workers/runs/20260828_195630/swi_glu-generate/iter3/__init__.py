"""pyasc asc2 submission package for CANN Bench.

Each module exposes one public callable whose name and signature match the
operator's ``proto.yaml`` schema. All numerical work happens in pyasc asc2
kernels launched directly on torch_npu-owned device buffers (zero-copy via
``Tensor.data_ptr()``).
"""

from .exp import exp
from .gelu import gelu
from .mish import mish
from .sigmoid import sigmoid
from .swi_glu import swi_glu

__all__ = ["exp", "gelu", "mish", "sigmoid", "swi_glu"]
