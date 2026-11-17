"""Build a short-row FP32 control that computes gather offsets inside the VF.

The two selected public shapes have halves longer than one FP32 register.
A 64-output group can therefore cross at most one gate/value boundary.
Input and output transfers remain dense across rows, including the final tile.
"""
from pathlib import Path
from shutil import copyfile


def source(half, rows, height):
    lanes = height * half
    if half < 64 or lanes % 8:
        raise ValueError('expected a half at least 64 lanes wide and DMA-aligned tile')
    groups = (lanes + 63) // 64
    vf = f"""using namespace AscendC::Reg;
auto* src = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
auto* dst = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
RegTensor<int32_t> lane, shifted, index;
RegTensor<float> gate, value, acc;
auto full = CreateMask<uint8_t>();
Arange(lane, (int32_t)0);
Adds(shifted, lane, (int32_t){half}, full);
for (uint16_t i = 0; i < {groups}; ++i) {{
    uint32_t start = (uint32_t)i * 64;
    uint32_t row = start / {half};
    uint32_t col = start % {half};
    uint32_t left = {half} - col;
    uint32_t valid = {lanes} - start;
    auto first = UpdateMask<float>(left);
    auto active = UpdateMask<float>(valid);
    Select(index, lane, shifted, first);
    auto* ptr = src + row * {2*half} + col;
    Gather(gate, ptr, (RegTensor<uint32_t>&)index, active);
    Gather(value, ptr + {half}, (RegTensor<uint32_t>&)index, active);
    Muls(acc, gate, -1.0f, active);
    Exp(acc, acc, active);
    Adds(acc, acc, 1.0f, active);
    Div(gate, gate, acc, active);
    Mul(gate, gate, value, active);
    StoreAlign(dst + start, gate, active);
}}"""
    return f"""from asc.experimental import asctile
from ._pyasc_runtime import ensure_npu_platform

@asctile.jit(reuse_alloc=2)
def dense_gather(x: asctile.GlobalAddress, y: asctile.GlobalAddress):
    source = asctile.global_tensor(x, [{2*rows*half}], l2_cache=False)
    output = asctile.global_tensor(y, [{rows*half}])
    offset = asctile.block_idx() * {lanes}
    count = min({lanes}, {rows*half} - offset)
    slab = asctile.copy_in(source, [2*offset], [{2*lanes}], real_shape=[2*count], pad_value=0.0)
    result = asctile.inline_vf({vf!r}, [{lanes}], asctile.float32, [slab])
    asctile.copy_out(result, output, [offset], real_shape=[count])

def swi_glu(input, dim=-1):
    ensure_npu_platform()
    shape = list(input.shape)
    shape[dim] //= 2
    y = input.new_empty(shape)
    dense_gather[{(rows+height-1)//height}](input, y)
    return y
"""


if __name__ == '__main__':
    for case, half, rows, height in ((11, 509, 273, 8), (13, 536, 143, 4)):
        # Independently check every per-register index against the physical rows.
        for start in range(0, height * half, 64):
            row, col = divmod(start, half)
            for lane in range(min(64, height * half - start)):
                r, c = divmod(start + lane, half)
                offset = row * 2 * half + col + lane + (half if col + lane >= half else 0)
                assert offset == r * 2 * half + c
                assert offset + half < 2 * height * half
        dest = Path(f'gather{case}')
        dest.mkdir()
        module = source(half, rows, height)
        compile(module, str(dest / 'swi_glu.py'), 'exec')
        (dest / 'swi_glu.py').write_text(module)
        copyfile('base/_pyasc_runtime.py', dest / '_pyasc_runtime.py')
