"""Control for direct SIMT GM reads and stores on a single FP32 row wave.

The only specialized dispatcher branch is the existing single-wave FP32 route.
Keep the source as a diagnostic until the compile and finite golden qualify.
"""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile


def kernel(threads):
    definition = f'''#include "simt_api/asc_simt.h"
__simt_vf__ LAUNCH_BOUND({threads}) inline void RowSwigluDirect(
    __gm__ float* x, __gm__ float* y, uint32_t rows, uint32_t half,
    uint32_t height, uint32_t owner) {{
    uint32_t start = owner * height;
    uint32_t end = (start + height < rows ? start + height : rows);
    uint32_t count = (end - start) * half;
    for (uint32_t i = AscendC::Simt::GetThreadIdx<0>(); i < count; i += {threads}) {{
        uint32_t row = start + i / half;
        uint32_t col = i % half;
        uint32_t address = (2 * row) * half + col;
        float gate = x[address];
        float value = x[address + half];
        y[start * half + i] = (gate / (1.0f + AscendC::Simt::Exp(-gate))) * value;
    }}
}}'''
    call = f'''AscendC::Simt::VF_CALL<RowSwigluDirect>(AscendC::Simt::Dim3{{{threads},1,1}},
    reinterpret_cast<__gm__ float*>($0), reinterpret_cast<__gm__ float*>($1),
    static_cast<uint32_t>($2), static_cast<uint32_t>($3), static_cast<uint32_t>($4), static_cast<uint32_t>($5));
AscendC::PipeBarrier<PIPE_ALL>();'''
    return f'''    b = asctile.block_idx()
    blocks = asctile.block_num()
    owner = (b + blocks // 2) % blocks if W == H else b
    asctile.inline({definition!r}, before_function=True)
    asctile.inline({call!r}, [x, y, R, H, T, owner])
'''


if __name__ == '__main__':
    text = Path('base/swi_glu.py').read_text()
    if sha256(text.encode()).hexdigest() != '30bc199acb49eecb1f262ca4afb5e16e3ce871f5f365954eb167107a61db412a':
        raise ValueError('expected the qualified 5ac7788 source')
    start = text.index('    gx =', text.index('def single_wave_fp32_rows('))
    end = text.index('\n\n@asctile.jit', start)
    for threads in (128, 256):
        dest = Path(f'simt{threads}')
        code = text[:start] + kernel(threads) + text[end:]
        compile(code, str(dest), 'exec')
        dest.mkdir()
        (dest/'swi_glu.py').write_text(code)
        copyfile('base/_pyasc_runtime.py', dest/'_pyasc_runtime.py')
