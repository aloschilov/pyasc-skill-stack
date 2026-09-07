#include "kernel_operator.h"
extern "C"  __global__ __aicore__ void _gelu_register(__gm__ bfloat16_t* v1_x_ptr, __gm__ bfloat16_t* v2_y_ptr, int32_t v3_size) {
  constexpr int64_t c13824_i64 = 13824;
  constexpr int32_t c27648_i32 = 27648;
  constexpr int32_t c32_i32 = 32;
  constexpr int32_t c2_i32 = 2;
  constexpr int32_t c1_i32 = 1;
  constexpr int32_t c0_i32 = 0;
  constexpr int32_t c13823_i32 = 13823;
  bfloat16_t c0_bf16 = 0.0e+00;
  constexpr int32_t c13824_i32 = 13824;
  AscendC::TPipe v4;
  int32_t v5 = AscendC::GetBlockIdx();
  int32_t v6 = v3_size + c13823_i32;
  int32_t v7 = v6 / c13824_i32;
  int32_t v8 = AscendC::GetBlockNum();
  AscendC::LocalTensor<bfloat16_t> v9{AscendC::TPosition::VECCALC, 0, 13824};
  AscendC::LocalTensor<bfloat16_t> v10{AscendC::TPosition::VECCALC, 27648, 13824};
  for (int32_t v11 = v5; v11 < v7; v11 += v8) {
    AscendC::GlobalTensor<bfloat16_t> v12;
    v12.SetGlobalBuffer(v1_x_ptr);
    AscendC::GlobalTensor<bfloat16_t> v13;
    v13.SetGlobalBuffer(v2_y_ptr);
    int32_t v14 = v11 * c13824_i32;
    int32_t v15 = v14 + c13824_i32;
    bool v16 = v15 <= v3_size;
    int32_t v17;
    if (v16) {
      v17 = c13824_i32;
    } else {
      int32_t v18 = v3_size - v14;
      v17 = v18;
    }
    AscendC::GlobalTensor<bfloat16_t> v19 = v12[v14];
    int32_t v20 = v3_size - v14;
    bool v21 = v20 < c0_i32;
    int32_t v22 = v21 ? c0_i32 : v20;
    int32_t v23 = ((v17 < v22) ? (v17) : (v22));
    int32_t v24 = v23 * c2_i32;
    int32_t v25 = v3_size - v23;
    int32_t v26 = v24 % c32_i32;
    bool v27 = v26 == c0_i32;
    int32_t v28 = c32_i32 - v26;
    int32_t v29 = v27 ? c0_i32 : v28;
    int32_t v30 = v24 + v29;
    bool v31 = v30 < c27648_i32;
    if (v31) {
      get_buf(PIPE_V, 0, 0);
      AscendC::Duplicate(v10, c0_bf16, c13824_i64);
      rls_buf(PIPE_V, 0, 0);
    }
    int32_t v32 = v25 * c2_i32;
    int32_t v33 = v29 / c2_i32;
    int32_t v34 = c27648_i32 - v30;
    int32_t v35 = v34 / c32_i32;
    AscendC::DataCopyExtParams v36{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v24), static_cast<uint32_t>(v32), static_cast<uint32_t>(v35), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<bfloat16_t> v37{c1_i32, c0_i32, static_cast<uint8_t>(v33), c0_bf16};
    get_buf(PIPE_MTE2, 0, 0);
    AscendC::DataCopyPad(v10, v19, v36, v37);
    rls_buf(PIPE_MTE2, 0, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 0, 0);
    {
      __VEC_SCOPE__
      {

        static constexpr AscendC::Reg::CastTrait up = {AscendC::Reg::RegLayout::ZERO,
            AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::Reg::CastTrait down = {AscendC::Reg::RegLayout::ZERO,
            AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        AscendC::Reg::RegTensor<bfloat16_t> low;

        auto* in = reinterpret_cast<__ubuf__ bfloat16_t*>(v10.GetPhyAddr());
        auto* out = reinterpret_cast<__ubuf__ bfloat16_t*>(v9.GetPhyAddr());
        AscendC::Reg::RegTensor<float> x, s, y;
        uint32_t count = v10.GetSize();
        uint16_t loops = (count + 63) / 64;
        for (uint16_t i=0; i<loops; ++i) {
            auto mask = AscendC::Reg::UpdateMask<float>(count);

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(low, in+i*64);
        AscendC::Reg::Cast<float, bfloat16_t, up>(x, low, mask);

            AscendC::Reg::Mul(s, x, x, mask);
            AscendC::Reg::Muls(s, s, -0.0713548163282308f, mask);
            AscendC::Reg::Adds(s, s, -1.5957691216057308f, mask);
            AscendC::Reg::Mul(s, s, x, mask);
            AscendC::Reg::Exp(s, s, mask);
            AscendC::Reg::Adds(s, s, 1.0f, mask);
            AscendC::Reg::Div(y, x, s, mask);

        AscendC::Reg::Cast<bfloat16_t, float, down>(low, y, mask);
        AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out+i*64, low, mask);

        }
        ;
      }
    }
    rls_buf(PIPE_V, 0, 0);
    rls_buf(PIPE_V, 1, 0);
    AscendC::GlobalTensor<bfloat16_t> v38 = v13[v14];
    int32_t v39 = c13824_i32 - v23;
    int32_t v40 = v39 * c2_i32;
    int32_t v41 = v40 / c32_i32;
    AscendC::DataCopyExtParams v42{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v24), static_cast<uint32_t>(v41), static_cast<uint32_t>(v32), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 1, 0);
    AscendC::DataCopyPad(v38, v9, v42);
    rls_buf(PIPE_MTE3, 1, 0);
  }
  return;
}
