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

from asc.experimental import asctile


@asctile.jit(always_compile=True)
def add_kernel(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress,
               out_ptr: asctile.GlobalAddress, size: int,
               tile_size: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    o_gm = asctile.global_tensor(out_ptr, [size])

    num_tiles = asctile.ceildiv(size, tile_size)
    for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num()):
        off = t * tile_size
        valid = min(size - off, tile_size)
        x = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[valid], pad_value=0)
        y = asctile.copy_in(y_gm, [off], [tile_size], real_shape=[valid], pad_value=0)
        asctile.copy_out(x + y, o_gm, [off], real_shape=[valid])


if __name__ == "__main__":
    asctile.set_platform(asctile.Backend.Model, asctile.Platform.Ascend950PR_9599)

    rng = np.random.default_rng(0)
    size, blocks, tile = 8192, 16, 128
    x = rng.standard_normal(size, dtype=np.float32)
    y = rng.standard_normal(size, dtype=np.float32)
    out = np.zeros_like(x)

    if size:
        add_kernel[blocks](x, y, out, size, tile)

    expected = x + y
    err = np.abs(out - expected).max() if size else 0.0
    assert err == 0.0, f"max abs err {err}"
    print("PASSED max_abs_err", err)
```

## Running it

The simulator libraries must be on `LD_LIBRARY_PATH` before the first
`asc` import, or `set_platform` raises `RuntimeError: Runtime library is not
available!`.

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
python vector_add.py
```

The CANN root is installation-specific; adjust the example to its actual path.
Confirm `$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/` is present before
blaming the kernel. Retain the selected toolkit version during qualification.

Historical output from the earlier evenly divided listing on `Ascend950PR_9599`, CANN 9.0.0, `pyasc-dev
1.1.1.dev20261021205441`, 16 blocks:

```
PASSED max_abs_err 0.0
[INFO] Total tick: 4842
```

This old observation does not qualify the newer setup above. The tick count is
a simulator figure. It is not a hardware latency measurement
and must never be reported as one — see [hardware.md](hardware.md).

## Tails

The listing visits only live tiles, including when there are more blocks than
tiles. Each copy has `0 < valid <= tile_size`; a block with no tile performs no
copy. For an empty input the host skips the launch.

A fixed `ceil(num_tiles / blocks)` iteration count for every block can leave
extra iterations in the last wave. Clamping their extent to zero still issues
a copy at the end of the allocation; an unclamped extent can become negative.
Exclude those iterations before loading or storing. If contiguous per-block
ownership is needed, guard the whole body with `off < size` instead of changing
the ownership scheme just to handle the tail.

Allocate the full physical tile via `shape`, bound both loads and the store with
`real_shape`, and choose operation-neutral padding. Test a partial tile and more
blocks than live tiles, with real numerical checks and a rejecting
suppressed-launch control.
