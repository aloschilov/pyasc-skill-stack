// Diagnostic for a packed FP32 input. Both variants present identical UB
// addresses to the VF and have disjoint, bounded output ownership.
#include "kernel_operator.h"

template<unsigned Height, unsigned Half, unsigned Pitch>
__simd_vf__ inline void window_swiglu(__ubuf__ float* input, __ubuf__ float* out) {
    using namespace AscendC::Reg;
    auto mask = CreateMask<uint8_t>();
    for (uint16_t r = 0; r < Height; ++r) {
        auto* g = input + 2 * r * Half;
        auto* v = g + Half;
        UnalignRegForLoad g_align, v_align;
        LoadUnAlignPre(g_align, g);
        LoadUnAlignPre(v_align, v);
        for (uint16_t c = 0; c < Pitch; c += 64) {
            RegTensor<float> gate, value, z;
            LoadUnAlign(gate, g_align, g, 64);
            LoadUnAlign(value, v_align, v, 64);
            Muls(z, gate, -1.0f, mask);
            Exp(z, z, mask);
            Adds(z, z, 1.0f, mask);
            Div(z, gate, z, mask);
            Mul(z, z, value, mask);
            StoreAlign(out + r * Pitch + c, z, mask);
        }
    }
}

extern "C" __global__ __vector__ void staged_rows(__gm__ float* x, __gm__ float* y) {
    using namespace AscendC;
    constexpr unsigned rows = 273, half = 509, height = 8, pitch = 512;
    constexpr unsigned quantum = 64; // element count for a 256-byte window
    constexpr unsigned input_size = 2 * rows * half;
    // A shifted start and a final unaligned register load may each use a vector.
    constexpr unsigned ub_size = 2 * height * pitch + 3 * quantum;
    InitSocState();
    unsigned first_row = GetBlockIdx() * height;
    unsigned valid = rows - first_row < height ? rows - first_row : height;
    unsigned first = 2 * first_row * half, end = first + 2 * valid * half;
    unsigned aligned_begin = (first / quantum) * quantum;
    unsigned rounded_end = ((end + quantum - 1) / quantum) * quantum;
    unsigned aligned_end = rounded_end < input_size ? rounded_end : input_size;
    unsigned lead = first - aligned_begin;
    GlobalTensor<float> input, output;
    input.SetGlobalBuffer(x);
    output.SetGlobalBuffer(y);
    LocalTensor<float> src(TPosition::VECCALC, 0, ub_size);
    LocalTensor<float> dst(TPosition::VECCALC, ub_size * sizeof(float), height * pitch);
    unsigned start = WIDEN_WINDOW ? aligned_begin : first;
    unsigned stop = WIDEN_WINDOW ? aligned_end : end;
    unsigned offset = WIDEN_WINDOW ? 0 : lead;
    DataCopyExtParams read{1, static_cast<uint32_t>((stop - start) * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(src[offset], input[start], read, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    auto* ptr = reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()) + lead;
    auto* res = reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr());
    if (valid == height)
        asc_vf_call<window_swiglu<height, half, pitch>>(ptr, res);
    else
        asc_vf_call<window_swiglu<rows % height, half, pitch>>(ptr, res);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams write{static_cast<uint16_t>(valid), half * sizeof(float), 0, 0, 0};
    DataCopyPad(output[first_row * half], dst, write);
    PipeBarrier<PIPE_ALL>();
}
