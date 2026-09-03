"""pyasc asc2 submission package for CANN Bench.

Each module exposes one public callable whose name and signature match the
operator's ``proto.yaml`` schema. All numerical work happens in pyasc asc2
kernels launched directly on torch_npu-owned device buffers (zero-copy via
``Tensor.data_ptr()``).
"""

from .exp import exp
from .foreach_addcdiv_scalar import foreach_addcdiv_scalar
from .foreach_norm import foreach_norm
from .gelu import gelu
from .masked_scale import masked_scale
from .mish import mish
from .rms_norm import rms_norm
from .sigmoid import sigmoid
from .swi_glu import swi_glu

__all__ = ["exp", "foreach_addcdiv_scalar", "foreach_norm", "gelu", "masked_scale", "mish", "rms_norm", "sigmoid", "swi_glu"]
