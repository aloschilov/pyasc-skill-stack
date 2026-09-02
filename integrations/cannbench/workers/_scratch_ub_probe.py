"""Definitive UB-budget probe: reads memory_consumed['UB'] from the MLIR module
after the in-process passes, BEFORE the host link step (which fails locally).
No NPU/runtime needed. Capacity for 950PR (C310) UB = 253952 bytes."""
import os, sys, inspect

_CANN_ENV = "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
if os.path.exists(_CANN_ENV):
    import subprocess
    out = subprocess.check_output(["bash", "-c", f"source {_CANN_ENV} && env"]).decode()
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)
for _b in ("/usr/local/Ascend/cann-9.0.0/aarch64-linux/bin/bisheng",
           "/usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin/bisheng"):
    if os.path.exists(_b):
        os.environ["PYASC_COMPILER"] = _b
        break

import asc
import asc2
from asc import DataType
from asc._C import ir
from asc.runtime.jit import MockTensor, JITFunction
from asc.codegen.specialization import Specialization
from asc.codegen.function_visitor import CodegenOptions
from asc.runtime.compiler import CompileOptions, Compiler
import asc.runtime.config as config

UB_CAP = 253952


def set_platform():
    plat = config.Platform("Ascend950PR_9599")
    config.set_platform(config.Backend.Model, plat, check=False)


def make_kernel(expr_kind="naive"):
    """Build a fresh @asc2.jit sigmoid kernel with the chosen math expression."""
    if expr_kind == "naive":
        src = (
            "def _k(x_ptr, out_ptr, size, num_tiles, tile_size):\n"
            "    x_gm = asc2.global_tensor(x_ptr, [size])\n"
            "    out_gm = asc2.global_tensor(out_ptr, [size])\n"
            "    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2):\n"
            "        off = t * tile_size\n"
            "        n = tile_size if off + tile_size <= size else size - off\n"
            "        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])\n"
            "        xf = x.to(asc.float32)\n"
            "        y = asc2.div(1.0, asc2.exp(-xf) + 1.0)\n"
            "        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])\n"
        )
    elif expr_kind == "tanh":
        src = (
            "def _k(x_ptr, out_ptr, size, num_tiles, tile_size):\n"
            "    x_gm = asc2.global_tensor(x_ptr, [size])\n"
            "    out_gm = asc2.global_tensor(out_ptr, [size])\n"
            "    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2):\n"
            "        off = t * tile_size\n"
            "        n = tile_size if off + tile_size <= size else size - off\n"
            "        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])\n"
            "        xf = x.to(asc.float32)\n"
            "        y = asc2.tanh(xf * 0.5) * 0.5 + 0.5\n"
            "        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])\n"
        )
    elif expr_kind == "stable":
        src = (
            "def _k(x_ptr, out_ptr, size, num_tiles, tile_size):\n"
            "    x_gm = asc2.global_tensor(x_ptr, [size])\n"
            "    out_gm = asc2.global_tensor(out_ptr, [size])\n"
            "    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2):\n"
            "        off = t * tile_size\n"
            "        n = tile_size if off + tile_size <= size else size - off\n"
            "        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])\n"
            "        xf = x.to(asc.float32)\n"
            "        s = xf\n"
            "        y = asc2.exp(asc2.minimum(s, 0.0)) / (asc2.exp(-asc2.abs(s)) + 1.0)\n"
            "        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])\n"
        )
    ns = {"asc": asc, "asc2": asc2}
    exec(src, ns)
    fn = ns["_k"]
    # add annotations the JIT needs
    fn.__annotations__ = {
        "x_ptr": asc.GlobalAddress, "out_ptr": asc.GlobalAddress,
        "size": int, "num_tiles": int,
        "tile_size": asc.ConstExpr[int],
    }
    return asc2.jit(fn)


def probe_ub(kernel, tile, dtype_str, compile_kwargs=None):
    compile_kwargs = compile_kwargs or {}
    size = 72 * tile * 8
    num_tiles = (size + tile - 1) // tile
    dt = DataType(dtype_str)
    # mimic __call__ prereqs
    sig = inspect.signature(kernel.fn)
    bound = sig.bind(MockTensor(dt), MockTensor(dt), size, num_tiles, tile)
    annotations = {"x_ptr": asc.GlobalAddress, "out_ptr": asc.GlobalAddress,
                   "size": int, "num_tiles": int, "tile_size": asc.ConstExpr[int]}
    runtime_args, constexprs = kernel.split_args(bound.arguments, annotations)
    arg_types = {n: kernel.get_arg_type(v) for n, v in runtime_args.items()}
    codegen_options = CodegenOptions()
    compile_options = CompileOptions(**compile_kwargs)
    spec = Specialization(arg_types, constexprs)
    mod = kernel._run_codegen(spec, codegen_options)
    compiler = Compiler(compile_options)
    # set attrs exactly like Compiler.run does, pre-passes
    builder = ir.Builder(mod.op)
    mod.set_attr(ir.attr.compilation_arch, builder.get_str_attr(compiler.arch.value))
    mod.set_attr(ir.attr.soc_version, builder.get_str_attr(compiler.soc_version.value))
    if compile_options.static_alloc is not None:
        mod.set_attr(ir.attr.static_alloc, builder.get_bool_attr(compile_options.static_alloc))
    if compile_options.vf_vec_len is not None:
        mod.set_attr(ir.attr.vf_vec_len, builder.get_i32_attr(compile_options.vf_vec_len))
    try:
        compiler.run_passes(mod)
    except Exception as e:
        return f"run_passes ERR: {e}"
    try:
        source = compiler.run_translation(mod)
    except Exception as e:
        return f"run_translation ERR: {str(e)[:200]}"
    mc = mod.op.get_dict_of_int_attr(ir.attr.memory_consumed)
    ub = mc.get("UB", -1)
    flag = "FIT" if ub <= UB_CAP else "OVERFLOW"
    return f"UB={ub:>7d}  (cap {UB_CAP})  [{flag}]  L1={mc.get('L1',0)}  full={mc}"


if __name__ == "__main__":
    set_platform()
    print(f"[env] arch target: 950PR/C310, UB cap={UB_CAP}", flush=True)
    print("=== naive expr, default options, sweep tile ===", flush=True)
    k = make_kernel("naive")
    for tile in (2048, 3072, 4096, 6144, 8192):
        for dt in ("float16", "bfloat16", "float32"):
            print(f"  naive TILE={tile:5d} {dt:9s}: {probe_ub(k, tile, dt)}", flush=True)
    print("=== compile-option variants at TILE=4096 (naive, f32) ===", flush=True)
    k2 = make_kernel("naive")
    for kw in [{}, {"reuse_alloc": 1}, {"reuse_alloc": 2},
               {"static_alloc": True}, {"static_alloc": False},
               {"vf_fusion": True}, {"vf_vec_len": 256}, {"vf_vec_len": 128}]:
        try:
            r = probe_ub(k2, 4096, "float32", kw)
        except Exception as e:
            r = f"OPT-ERR: {type(e).__name__}: {str(e)[:120]}"
        print(f"  4096 f32 {str(kw):28s}: {r}", flush=True)
    print("=== expr variants at TILE=4096 (f32) ===", flush=True)
    for ek in ("naive", "tanh", "stable"):
        kk = make_kernel(ek)
        try:
            r = probe_ub(kk, 4096, "float32", {})
        except Exception as e:
            r = f"EXPR-ERR: {type(e).__name__}: {str(e)[:150]}"
        print(f"  {ek:7s} 4096 f32: {r}", flush=True)
