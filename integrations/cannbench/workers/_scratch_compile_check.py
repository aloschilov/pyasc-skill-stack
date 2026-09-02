"""Local JIT compile-check for the sigmoid kernel UB budget (scratch file)."""
import os

_CANN_ENV = "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
if os.path.exists(_CANN_ENV):
    import subprocess
    out = subprocess.check_output(["bash", "-c", f"source {_CANN_ENV} && env"]).decode()
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

# Point pyasc at the bisheng AscendC compiler so MLIR->AscendC->binary completes.
for _b in ("/usr/local/Ascend/cann-9.0.0/aarch64-linux/bin/bisheng",
           "/usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin/bisheng"):
    if os.path.exists(_b):
        os.environ["PYASC_COMPILER"] = _b
        break

import asc
import asc2
from asc import DataType
from asc.runtime.jit import MockTensor, MockValue
import asc.runtime.config as config


def set_platform():
    # Model (simulator) backend avoids the NPU runtime lib (libruntime.so),
    # which is absent on this host. check=False skips the lib-availability gate.
    plat = config.Platform("Ascend950PR_9599")
    config.set_platform(config.Backend.Model, plat, check=False)
    print(f"[env] platform set (Model/sim): {plat}", flush=True)


@asc2.jit
def _sigmoid_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                    size: int, num_tiles: int,
                    tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        y = asc2.div(1.0, asc2.exp(-xf) + 1.0)
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def compile_test(tile, dtype_str, cores=72):
    size = 72 * tile * 8
    num_tiles = (size + tile - 1) // tile
    dt = DataType(dtype_str)
    try:
        _sigmoid_kernel[cores](
            MockTensor(dt), MockTensor(dt),
            size, num_tiles, tile,
        )
        return "LAUNCH-ERR (compile OK, no UB error)"
    except RuntimeError as e:
        msg = str(e)
        low = msg.lower()
        if "ub overflow" in low or ("overflow" in low and "byte" in low):
            return f"UB-OVERFLOW: {msg[:300]}"
        return f"OTHER-RUNTIME: {msg[:500]}"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:500]}"


if __name__ == "__main__":
    set_platform()
    for tile in (2048, 4096, 8192):
        for dt in ("float16", "bfloat16", "float32"):
            r = compile_test(tile, dt)
            print(f"TILE={tile:5d} dtype={dt:9s} -> {r}", flush=True)
