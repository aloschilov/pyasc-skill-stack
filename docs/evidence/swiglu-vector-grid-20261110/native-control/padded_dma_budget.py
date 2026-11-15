"""Measure both strided inputs, one bounded output and fused FP32 arithmetic."""

from argparse import ArgumentParser
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

ROWS, HALF, TILE, PITCH = 273, 509, 6, 512
GRID = (ROWS + TILE - 1) // TILE
MODES = ("gate", "value", "fused")


def device(original, mode):
    signature = "void row_single_wave_fp32(__gm__ float* v1_x, __gm__ float* v2_y) {"
    if signature not in original or "c509_i32 = 509" not in original or "c273_i32 = 273" not in original:
        raise ValueError("expected W81 padded FP32 case 11")
    header = original[:original.index(signature) + len(signature)]
    source = header + """
  using namespace AscendC;
  constexpr int R = 273, H = 509, T = 6, W = 512, N = 46;
  const int b = GetBlockIdx();
  const int slot = b % (N / 2);
  const int tile = ((N / 2 - 1 - slot) % 2) ? ((b + N / 2) % N) : b;
  const int row = tile * T;
  const int valid = (R - row < T) ? R - row : T;
  TPipe pipe;
  LocalTensor<float> gate{TPosition::VECCALC, T * W * sizeof(float), T * W};
  LocalTensor<float> value{TPosition::VECCALC, 0, T * W};
  GlobalTensor<float> src, dst;
  src.SetGlobalBuffer(v1_x);
  src.SetL2CacheHint<CacheRwMode::RW>(CacheMode::CACHE_MODE_DISABLE);
  dst.SetGlobalBuffer(v2_y);
  DataCopyExtParams input{static_cast<uint16_t>(valid), H * sizeof(float), H * sizeof(float), 0, 0};
  DataCopyPadExtParams<float> pad{true, 0, W-H, 0.0f};
  DataCopyPad(gate, src[row * 2 * H], input, pad);
  DataCopyPad(value, src[row * 2 * H + H], input, pad);
"""
    if mode == "fused":
        # Only valid padded rows were transferred. Restrict vector iteration
        # to those rows, including their three locally zero-padded lanes.
        source += """
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  const uint16_t repeats = valid * (W / 64);
  __ubuf__ float* g = reinterpret_cast<__ubuf__ float*>(gate.GetPhyAddr());
  __ubuf__ float* v = reinterpret_cast<__ubuf__ float*>(value.GetPhyAddr());
  __VEC_SCOPE__ {
    using namespace AscendC::Reg;
    auto mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> a, bval, z;
    for (uint16_t i=0; i<repeats; ++i) {
      LoadAlign(a, g + i*64);
      LoadAlign(bval, v + i*64);
      Muls(z, a, -1.0f, mask);
      Exp(z, z, mask);
      Adds(z, z, 1.0f, mask);
      Div(z, a, z, mask);
      Mul(z, z, bval, mask);
      StoreAlign(g + i*64, z, mask);
    }
  }
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
"""
    elif mode in ("gate", "value"):
        source += """
  SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
"""
    else:
        raise ValueError(mode)
    output = "value" if mode == "value" else "gate"
    return source + f"""
  DataCopyExtParams write{{static_cast<uint16_t>(valid), H*sizeof(float), 0, 0, 0}};
  DataCopyPad(dst[row*H], {output}, write);
}}
"""


@contextmanager
def compiled(mode):
    from asc.runtime.compiler import Compiler
    original = Compiler.run_translation
    with patch.object(Compiler, "run_translation", staticmethod(lambda mod: device(original(mod), mode))):
        yield


def stage(package, task, root):
    if sha256((package/"swi_glu.py").read_bytes()).hexdigest() != W81:
        raise ValueError("expected the independently generated W81 module")
    for mode in ("gate", "value"):
        fx = root/"input"/mode/"tasks/level1/swi_glu"
        fx.mkdir(parents=True)
        for name in ("cases.csv", "cases.yaml", "proto.yaml"):
            copyfile(task/name, fx/name)
        half = 0 if mode == "gate" else 1
        (fx/"golden.py").write_text(f"def swi_glu(input, dim=-1):\n    return input.chunk(2, dim=dim)[{half}].contiguous()\n")


def tagged(package, mode):
    import torch
    import torch_npu
    sys.path.insert(0, str(package.parent))
    mod = import_module(f"{package.name}.swi_glu")
    mod.ensure_npu_platform()
    x = torch.arange(ROWS*2*HALF, dtype=torch.float32).view(3,7,13,2*HALF)
    on_device = x.npu()
    y = torch.full((ROWS*HALF+256,), -1.0, device=on_device.device)
    grid = mod._npu_stream_grid(on_device, GRID)
    with compiled(mode):
        mod.row_single_wave_fp32[grid](on_device, y[128:-128], ROWS, HALF, TILE, PITCH)
    actual = y.cpu()
    half = 1 if mode == "value" else 0
    expected = x.chunk(2, dim=-1)[half].contiguous().reshape(-1)
    assert torch.equal(actual[128:-128], expected)
    assert (actual[:128] == -1).all() and (actual[-128:] == -1).all()
    assert torch.equal(x, on_device.cpu())
    print(json.dumps(dict(mode=mode, blocks=GRID, words=expected.numel(),
                          sha256=sha256(actual[128:-128].numpy().tobytes()).hexdigest())))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="cmd", required=True)
    s = cmd.add_parser("stage")
    for arg in ("package", "task", "root"):
        s.add_argument(arg, type=Path)
    t = cmd.add_parser("tag")
    t.add_argument("package", type=Path)
    t.add_argument("mode", choices=("gate","value"))
    p = cmd.add_parser("profile")
    p.add_argument("mode", choices=(*MODES, "real"))
    for arg in ("package","task","worker","root","dest"):
        p.add_argument(arg, type=Path)
    args = parser.parse_args()
    if args.cmd == "stage":
        stage(args.package, args.task, args.root)
    elif args.cmd == "tag":
        tagged(args.package, args.mode)
    else:
        fx = args.task if args.mode in ("fused","real") else args.root/"input"/args.mode/"tasks/level1/swi_glu"
        sys.argv = [str(args.worker), str(args.package), str(fx), "11", str(args.dest)]
        if args.mode == "real":
            runpy.run_path(str(args.worker), run_name="__main__")
        else:
            with compiled(args.mode):
                runpy.run_path(str(args.worker), run_name="__main__")
