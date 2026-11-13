// Single packed GM read, one register calculation and one bounded output DMA.
#pragma once
#include "kernel_operator.h"

template<unsigned Height, unsigned Pitch>
__simd_vf__ inline void eval_rows(__ubuf__ float* x, __ubuf__ float* y) {
    using namespace AscendC::Reg;
    const auto mask = CreateMask<uint8_t>();
    for (uint16_t row = 0; row < Height; ++row) {
        for (uint16_t part = 0; part < (Pitch + 63) / 64; ++part) {
            // Overlap the last vector to keep its address aligned and in the row.
            const uint16_t col = part * 64 + 64 <= Pitch ? part * 64 : Pitch - 64;
            RegTensor<float> gate, value, result;
            LoadAlign(gate, x + 2 * row * Pitch + col);
            LoadAlign(value, x + (2 * row + 1) * Pitch + col);
            Muls(result, gate, -1.0f, mask);
            Exp(result, result, mask);
            Adds(result, result, 1.0f, mask);
            Div(result, gate, result, mask);
            Mul(result, result, value, mask);
            StoreAlign(y + row * Pitch + col, result, mask);
        }
    }
}

template<unsigned Rows, unsigned Half, unsigned Height>
__global__ __vector__ void vector_rows(__gm__ float* x, __gm__ float* y) {
    using namespace AscendC;
    constexpr unsigned pitch = (Half + 7) / 8 * 8;
    constexpr unsigned lanes = Height * pitch;
    InitSocState();
    const unsigned row = GetBlockIdx() * Height;
    const uint16_t valid = Rows - row < Height ? Rows - row : Height;
    GlobalTensor<float> input, output;
    input.SetGlobalBuffer(x);
    output.SetGlobalBuffer(y);
    LocalTensor<float> xy(TPosition::VECCALC, 0, 2 * lanes);
    LocalTensor<float> out(TPosition::VECCALC, 2 * lanes * sizeof(float), lanes);
    DataCopyExtParams read{static_cast<uint16_t>(2 * valid), Half * sizeof(float), 0, 0, 0};
    DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(pitch - Half), 0};
    DataCopyPad<float>(xy, input[2 * row * Half], read, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    asc_vf_call<eval_rows<Height, pitch>>((__ubuf__ float*)xy.GetPhyAddr(), (__ubuf__ float*)out.GetPhyAddr());
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams write{valid, Half * sizeof(float), 0, 0, 0};
    DataCopyPad<float>(output[row * Half], out, write);
    PipeBarrier<PIPE_ALL>();
}
