// Compare packing the UB result with overwriting the consumed gate rows.
#include "kernel_operator.h"

template<unsigned Count, unsigned Half, unsigned Pitch, unsigned OutputPitch>
__simd_vf__ inline void activate_rows(__ubuf__ float* input, __ubuf__ float* output) {
    using namespace AscendC::Reg;
    for (uint16_t row = 0; row < Count; ++row) {
        for (uint16_t col = 0; col < Half; col += 64) {
            uint32_t active = Half - col;
            auto mask = UpdateMask<float>(active);
            RegTensor<float> gate, value, tmp;
            LoadAlign(gate, input + 2 * row * Pitch + col);
            LoadAlign(value, input + (2 * row + 1) * Pitch + col);
            Muls(tmp, gate, -1.0f, mask);
            Exp(tmp, tmp, mask);
            Adds(tmp, tmp, 1.0f, mask);
            Div(tmp, gate, tmp, mask);
            Mul(tmp, tmp, value, mask);
            StoreAlign(output + row * OutputPitch + col, tmp, mask);
        }
    }
}

extern "C" __global__ __vector__ void staged_rows(__gm__ float* x, __gm__ float* y) {
    using namespace AscendC;
    constexpr unsigned rows = TEST_ROWS, half = TEST_HALF, height = TEST_HEIGHT;
    constexpr unsigned pitch = (half + 7) / 8 * 8;
    constexpr unsigned input_size = 2 * height * pitch + 64;
    constexpr unsigned output_pitch = IN_GATE ? 2 * pitch : pitch;
    unsigned first = GetBlockIdx() * height;
    unsigned valid = rows - first < height ? rows - first : height;
    InitSocState();
    GlobalTensor<float> gx, gy;
    gx.SetGlobalBuffer(x);
    gx.SetL2CacheHint<CacheRwMode::RW>(CacheMode::CACHE_MODE_DISABLE);
    gy.SetGlobalBuffer(y);
    LocalTensor<float> slab(TPosition::VECCALC, 0, input_size);
    LocalTensor<float> result(TPosition::VECCALC, IN_GATE ? 0 : input_size * sizeof(float),
                              height * output_pitch);
    DataCopyExtParams rd{static_cast<uint16_t>(2 * valid), half * sizeof(float), 0, 0, 0};
    DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(pitch - half), 0};
    DataCopyPad(slab, gx[2 * first * half], rd, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    auto* src = reinterpret_cast<__ubuf__ float*>(slab.GetPhyAddr());
    auto* dst = reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr());
    if (valid == height)
        asc_vf_call<activate_rows<height, half, pitch, output_pitch>>(src, dst);
    else
        asc_vf_call<activate_rows<rows % height, half, pitch, output_pitch>>(src, dst);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams wr{static_cast<uint16_t>(valid), half * sizeof(float),
                         (output_pitch - pitch) * sizeof(float) / 32, 0, 0};
    DataCopyPad(gy[first * half], result, wr);
    PipeBarrier<PIPE_ALL>();
}
