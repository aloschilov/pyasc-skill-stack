"""Keep both strided GM reads and use the exact two planar UB extents in one VF."""
import argparse
import ast
import hashlib
from pathlib import Path

from row_math_cost import W62
from wave_rotation import prepare, run


def vector_code(count, pairs, *, chains=None, mask_overlap=False):
    # All loads/stores start on 32-byte boundaries. A short final vector
    # reuses lanes from the preceding vector, entirely inside this UB tile.
    if count < 64 or count % 8:
        raise ValueError("expected an aligned FP32 tile of at least one register")
    n = count // 64 - int(count % 64 != 0)
    funcs = []
    def batch(offsets, masks=None):
        if masks is None:
            masks = ["m"] * len(offsets)
        lines = []
        for i, off in enumerate(offsets):
            lines += [f"RegTensor<float> g{i}, v{i}, t{i};",
                      f"LoadAlign(g{i}, g + {off});", f"LoadAlign(v{i}, v + {off});"]
        for op in ("Muls(t{i}, g{i}, -1.0f, {mask});", "Exp(t{i}, t{i}, {mask});",
                   "Adds(t{i}, t{i}, 1.0f, {mask});", "Div(t{i}, g{i}, t{i}, {mask});",
                   "Mul(t{i}, t{i}, v{i}, {mask});"):
            lines += [op.format(i=i, mask=mask) for i, mask in enumerate(masks)]
        lines += [f"StoreAlign(y + {off}, t{i}, {masks[i]});" for i, off in enumerate(offsets)]
        return "\n".join(lines)
    step = chains if chains is not None else (2 if pairs else 1)
    if step not in (1, 2, 4):
        raise ValueError("expected 1, 2 or 4 independent register chains")
    funcs += [f"for(uint16_t i=0; i<{n//step}; ++i) {{",
              batch([f"{step*64}*i + {64*j}" for j in range(step)]), "}"]
    if n % step:
        funcs += ["{", batch([str(64*j) for j in range(n - n % step, n)]), "}"]
    if count % 64:
        # The second register ends exactly at the allocation boundary.
        # Its first lane is (count % 64) lanes beyond the first register.
        masks = None
        if mask_overlap:
            funcs += [f"uint32_t live = {count % 64};", "auto edge = UpdateMask<float>(live);"]
            masks = ["edge", "m"]
        funcs += ["{", batch([str(n*64), str(count-64)], masks), "}"]
    return """using namespace AscendC::Reg;
auto* g = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
auto* v = reinterpret_cast<__ubuf__ float*>($2.GetPhyAddr());
auto* y = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
auto m = CreateMask<uint8_t>();
""" + "\n".join(funcs)


def render(source, mode, count):
    if hashlib.sha256(source.encode()).hexdigest() != W62:
        raise ValueError("expected W62")
    if mode == "base":
        return source
    if mode not in ("single", "pair"):
        raise ValueError(mode)
    node = next(n for n in ast.parse(source).body if getattr(n, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    start = next(n.lineno - 1 for n in node.body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", "") == "gate_f")
    code = vector_code(count, mode == "pair")
    end = node.end_lineno - 1
    fallback = lines[start:end]
    lines[start:end] = [f"    if T * W == {count}:\n",
                        f"        out = asctile.inline_vf({code!r}, [T, W], asctile.float32, [gate, value])\n",
                        "    else:\n"] + ["    " + line for line in fallback]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "task", "worker"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--case", type=int, choices=(11, 13), required=True)
    args = parser.parse_args()
    shape, count = {11: ([273, 1, 1, 1018], 4096), 13: ([143, 1, 16, 67], 2144)}[args.case]
    fixtures = {"public": None, "finite": shape}
    modes = ("base", "single", "pair")
    prepare(args.base.resolve(), args.task.resolve(), fixtures, modes,
            renderer=lambda src, mode: render(src, mode, count), case=args.case)
    raise SystemExit(run(args.worker.resolve(), fixtures, modes, case=args.case))
