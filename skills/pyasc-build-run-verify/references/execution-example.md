# Execution example (simulator first, no hardware)

A complete, runnable vector add. `Backend.Model` is the cycle-accurate Ascend
**simulator**; `Backend.NPU` runs on a real device. `kernel[block_num](...)`
launches `block_num` AI cores.

This listing uses **numpy, not torch**: `pyasc-dev` declares only
`pybind11`, `numpy<2` and `typing_extensions`, so a plain
`pip install --pre pyasc-dev` has no torch. Install torch separately if you want
a torch reference — do not assume it is present.

```python
import os
from pathlib import Path

os.environ["PYASC_DUMP_PATH"] = str(Path.cwd() / "dump")
os.environ["PYASC_CACHE_DIR"] = str(Path.cwd() / "cache")

import numpy as np

# The published pyasc-dev wheel exposes asctile as asc.experimental.asctile;
# older source builds expose it as a top-level module.
try:
    from asc.experimental import asctile
except ImportError:  # pragma: no cover - legacy source build
    import asctile


@asctile.jit(always_compile=True)
def add_kernel(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress,
               out_ptr: asctile.GlobalAddress, size: int,
               tile_size: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    o_gm = asctile.global_tensor(out_ptr, [size])

    per_core = asctile.ceildiv(asctile.ceildiv(size, asctile.block_num()), tile_size)
    base = asctile.block_idx() * tile_size * per_core

    for i in asctile.range(per_core):
        off = base + i * tile_size
        x = asctile.copy_in(x_gm, [off], [tile_size])
        y = asctile.copy_in(y_gm, [off], [tile_size])
        asctile.copy_out(x + y, o_gm, [off])


if __name__ == "__main__":
    asctile.set_platform(asctile.Backend.Model, asctile.Platform.Ascend950PR_9599)

    rng = np.random.default_rng(0)
    size, blocks, tile = 8192, 16, 128
    x = rng.standard_normal(size, dtype=np.float32)
    y = rng.standard_normal(size, dtype=np.float32)
    out = np.zeros_like(x)

    add_kernel[blocks](x, y, out, size, tile)

    expected = x + y
    err = np.abs(out - expected).max()
    assert err == 0.0, f"max abs err {err}"
    print("PASSED max_abs_err", err)
```

## Running it

The simulator libraries must be on `LD_LIBRARY_PATH` before the first
`asc` import, or `set_platform` raises `RuntimeError: Runtime library is not
available!`.

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
python vector_add.py
```

The CANN root is installation-specific. Locate it rather than assuming: this
host has `/usr/local/Ascend/cann-9.0.0/set_env.sh`, and `$HOME/Ascend/` does not
exist. Confirm `$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/` is present
before blaming the kernel.

Verified output on `Ascend950PR_9599`, CANN 9.0.0, `pyasc-dev
1.1.1.dev20261021205441`, 16 blocks:

```
PASSED max_abs_err 0.0
[INFO] Total tick: 4842
```

The tick count is a simulator figure. It is not a hardware latency measurement
and must never be reported as one — see [hardware.md](hardware.md).

## Tails

This example divides evenly. When the size is not a multiple of the tile,
allocate the full physical tile via `shape`, pass the logical extent through
`real_shape`, and pad with an operation-neutral `pad_value`:

```python
valid = min(size - off, tile_size)
a = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[valid], pad_value=0)
asctile.copy_out(a + b, z_gm, [off], real_shape=[valid])
```
