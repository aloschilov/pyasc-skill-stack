"""Private case 13 DMA and launch controls. The real and wide modes compute full SwiGLU."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import runpy
from shutil import copyfile
import sys
from unittest.mock import patch

from packed_math_cost import W81

CPP = "a9e1aa349ca757b9e2e9ebff6718ddc5e40e7e780935c88260e964f675d544e3"
MODES = {"empty": 0, "r1": 1, "r2": 2, "r3": 3, "full": 4, "upper": 4}
VOLUME_MODES = ("flat", "read", "write", "both")
ROWS, HALF, TILE, GRID = 143, 536, 4, 36


def device(original, mode):
    if sha256(original.encode()).hexdigest() != CPP:
        raise ValueError("expected the qualified case 13 specialization on CANN 9.2")
    head = original[:original.index(" {") + 2]
    if mode == "empty":
        return head + "\n  return;\n}\n"
    # Vary contiguous input and output lengths independently. With a flat
    # write even the three-row terminal tile can emit exactly half its bytes.
    limit = TILE if mode in VOLUME_MODES else MODES[mode]
    read_words = "valid * H" if mode in ("read", "both") else "valid * 2 * H"
    cpp = head + f"""
  using namespace AscendC;
  constexpr int32_t H = {HALF}, T = {TILE}, R = {ROWS}, limit = {limit};
  const int32_t block = GetBlockIdx();
  const int32_t count = GetBlockNum();
  const int32_t cutoff = count - count / 2;
  const int32_t tile = block < cutoff ? block + count / 2 : block - cutoff;
  const int32_t row = T * tile;
  const int32_t valid = R - row < limit ? R - row : limit;
  TPipe pipe;
  LocalTensor<float> slab{{TPosition::VECCALC, 8576, 2 * T * H}};
  GlobalTensor<float> src, dst;
  src.SetGlobalBuffer(v1_x);
  src.SetL2CacheHint<CacheRwMode::RW>(CacheMode::CACHE_MODE_DISABLE);
  dst.SetGlobalBuffer(v2_y);
  DataCopyExtParams read{{1, static_cast<uint32_t>({read_words} * sizeof(float)), 0, 0, 0}};
  DataCopyPadExtParams<float> pad{{true, 0, 0, 0.0f}};
  DataCopyPad(slab, src[row * 2 * H], read, pad);
  SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
  DataCopyExtParams write{{static_cast<uint16_t>(valid), H * sizeof(float), 0, 0, 0}};
  DataCopyPad(dst[tile * limit * H], slab[{'valid * H' if mode == 'upper' else '0'}], write);
}}
"""
    if mode in VOLUME_MODES:
        divisor = 2 if mode in ("write", "both") else 1
        cpp = cpp[:cpp.index("  DataCopyExtParams write")] + f"""
  DataCopyExtParams write{{1, static_cast<uint32_t>(valid * H * sizeof(float) / {divisor}), 0, 0, 0}};
  DataCopyPad(dst[tile * T * H / {divisor}], slab[0], write);
}}
"""
    return cpp


def output_elements(mode):
    if mode in VOLUME_MODES:
        return ROWS * HALF // (2 if mode in ("write", "both") else 1)
    limit = MODES[mode]
    return (35 * limit + min(3, limit)) * HALF


def stage(package, task, output, volume=False):
    source = (package / "swi_glu.py").read_text()
    if sha256(source.encode()).hexdigest() != W81:
        raise ValueError("expected W81")
    anchor = "                packed_prefix_rows[grid](x, out, rows, half, height)\n"
    assert source.count(anchor + "                return out\n") == 1
    modes = ("full", *VOLUME_MODES) if volume else MODES
    for mode in modes:
        limit = MODES.get(mode, TILE)
        root = output / mode
        root.mkdir()
        copyfile(package / "_pyasc_runtime.py", root / "_pyasc_runtime.py")
        (root / "__init__.py").touch()
        n = output_elements(mode)
        if mode == "empty":
            # An untouched, contiguous input view gives the private timing
            # harness a defined result. The separate sentinel test inspects y.
            ret = "return x.reshape(-1)[:out.numel()].view_as(out)"
        elif n < ROWS * HALF:
            ret = f"return out.reshape(-1)[:{n}]"
        else:
            ret = "return out"
        (root / "swi_glu.py").write_text(source.replace(anchor + "                return out\n", anchor + "                " + ret + "\n"))
        fx = output / "input" / mode / "tasks/level1/swi_glu"
        fx.mkdir(parents=True)
        for name in ("cases.csv", "cases.yaml", "proto.yaml"):
            copyfile(task / name, fx / name)
        golden = """import torch

def swi_glu(input, dim=-1):
    rows = input.reshape(143, 1072)
"""
        if mode == "empty":
            golden += "    return rows.flatten()[:143*536].reshape(11,13,8,67).clone()\n"
        else:
            golden += f"""    parts = []
    for start in range(0, 143, 4):
        data = rows[start:start+min({limit}, 143-start)].flatten()
        n = data.numel() // {4 if mode in ("write", "both") else 2}
        parts.append(data[{'n:' if mode == 'upper' else ':n'}])
    result = torch.cat(parts)
"""
            golden += "    return result.reshape(11,13,8,67)\n" if n == ROWS * HALF else "    return result\n"
        (fx / "golden.py").write_text(golden)

    if volume:
        # The VF already stores at row*H+col. Express that dense result as
        # rank one so the public emitter can issue one full-span write.
        old = (
            '""", [T, H], asctile.float32, [slab])\n'
            '    gy = asctile.global_tensor(y, [R, H])\n'
            '    asctile.copy_out(result, gy, [r, 0], real_shape=[valid, H])'
        )
        new = (
            '""", [T * H], asctile.float32, [slab])\n'
            '    gy = asctile.global_tensor(y, [R * H])\n'
            '    asctile.copy_out(result, gy, [r * H], real_shape=[valid * H])'
        )
        if source.count(old) != 1:
            raise ValueError("expected one contiguous packed register result")
        root = output / "wide"
        root.mkdir()
        (root / "__init__.py").touch()
        copyfile(package / "_pyasc_runtime.py", root / "_pyasc_runtime.py")
        (root / "swi_glu.py").write_text(source.replace(old, new))


@contextmanager
def compiled(mode):
    from asc.runtime.compiler import Compiler
    original = Compiler.run_translation
    def translate(module):
        return device(original(module), mode)
    with patch.object(Compiler, "run_translation", staticmethod(translate)):
        yield


def profile(root, mode, task, worker, dest):
    package = root / ("base" if mode == "real" else mode)
    fx = task if mode in ("real", "wide") else root / "input" / mode / "tasks/level1/swi_glu"
    sys.argv = [str(worker), str(package), str(fx), "13", str(dest)]
    if mode in ("real", "wide"):
        runpy.run_path(str(worker), run_name="__main__")
    else:
        with compiled(mode):
            runpy.run_path(str(worker), run_name="__main__")


def tagged(root, mode):
    import torch
    import torch_npu
    sys.path.insert(0, str(root))
    module = import_module(f"{mode}.swi_glu")
    module.ensure_npu_platform()
    x = torch.arange(ROWS * 2 * HALF, dtype=torch.float32).reshape(11,13,16,67)
    on_device = x.npu()
    guard = 128
    y = torch.full((ROWS * HALF + 2 * guard,), -1.0, device=on_device.device)
    output = y[guard:-guard].view(11,13,8,67)
    grid = module._npu_stream_grid(on_device, GRID)
    with compiled(mode):
        module.packed_prefix_rows[grid](on_device, output, ROWS, HALF, TILE)
    actual = y.cpu()
    n = output_elements(mode)
    filled = actual[guard:guard+n]
    assert (actual[:guard] == -1).all() and (actual[guard+n:] == -1).all()
    task = root / "input" / mode / "tasks/level1/swi_glu/golden.py"
    if n:
        oracle = runpy.run_path(str(task))["swi_glu"](x, dim=2).flatten()
        assert torch.equal(filled, oracle)
    assert torch.equal(on_device.cpu(), x)
    input_bytes = n * 8
    if mode in VOLUME_MODES:
        input_bytes = ROWS * HALF * (4 if mode in ("read", "both") else 8)
    print(json.dumps(dict(mode=mode, blocks=GRID, input_bytes=input_bytes, output_bytes=n*4,
                          checked_elements=n, guards_intact=True, input_intact=True,
                          words_sha256=sha256(filled.numpy().tobytes()).hexdigest())))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="cmd", required=True)
    s = cmd.add_parser("stage")
    for a in ("package", "task", "output"):s.add_argument(a, type=Path)
    s.add_argument("--volume", action="store_true", help="stage the independent read/write volume controls")
    for name in ("profile", "tag"):
        sub = cmd.add_parser(name)
        sub.add_argument("root", type=Path)
        sub.add_argument("mode", choices=[*MODES, *VOLUME_MODES, "real", "wide"] if name == "profile" else [*MODES, *VOLUME_MODES])
        if name == "profile":
            for a in ("task", "worker", "dest"):sub.add_argument(a, type=Path)
    a = p.parse_args()
    if a.cmd == "stage":stage(a.package, a.task, a.output, a.volume)
    elif a.cmd == "profile":profile(a.root, a.mode, a.task, a.worker, a.dest)
    else:tagged(a.root, a.mode)
