#include "kernel_operator.h"
extern "C"  __global__ __aicore__ void gelu_kernel(__gm__ float* v1_input_ptr, __gm__ float* v2_output_ptr, int32_t v3_size) {
  constexpr uint32_t c248_idx = 248;
  constexpr uint32_t c64_idx = 64;
  constexpr uint32_t c15872_idx = 15872;
  constexpr int32_t c15872_i32 = 15872;
  constexpr int32_t c1_i32 = 1;
  constexpr int32_t c0_i32 = 0;
  constexpr float c0_f32 = (float)0.0e+00;
  constexpr int32_t c15871_i32 = 15871;
  constexpr float c0_707106769_f32 = (float)7.071067690e-01;
  constexpr float c1_f32 = (float)1.000000000e+00;
  constexpr float c0_5_f32 = (float)5.000000000e-01;
  constexpr int32_t c4_i32 = 4;
  constexpr int32_t c32_i32 = 32;
  constexpr int32_t c63488_i32 = 63488;
  constexpr int64_t c15872_i64 = 15872;
  AscendC::TPipe v4;
  AscendC::LocalTensor<float> v5{AscendC::TPosition::VECCALC, 0, 15872};
  AscendC::LocalTensor<float> v6{AscendC::TPosition::VECCALC, 63488, 15872};
  AscendC::LocalTensor<float> v7{AscendC::TPosition::VECCALC, 126976, 15872};
  int32_t v8 = v3_size + c15871_i32;
  int32_t v9 = v8 / c15872_i32;
  int32_t v10 = AscendC::GetBlockNum();
  int32_t v11 = v9 / v10;
  int32_t v12 = v9 % v10;
  int32_t v13 = AscendC::GetBlockIdx();
  int32_t v14 = v12 - v13;
  int32_t v15 = ((v14 > c0_i32) ? (v14) : (c0_i32));
  int32_t v16 = ((v15 < c1_i32) ? (v15) : (c1_i32));
  int32_t v17 = v11 + v16;
  int32_t v18 = v13 * v11;
  int32_t v19 = ((v13 < v12) ? (v13) : (v12));
  int32_t v20 = v18 + v19;
  for (int32_t v21 = c0_i32; v21 < v17; v21 += c1_i32) {
    AscendC::GlobalTensor<float> v22;
    v22.SetGlobalBuffer(v1_input_ptr);
    AscendC::GlobalTensor<float> v23;
    v23.SetGlobalBuffer(v2_output_ptr);
    int32_t v24 = v20 + v21;
    int32_t v25 = v24 * c15872_i32;
    int32_t v26 = v3_size - v25;
    int32_t v27 = ((v26 < c15872_i32) ? (v26) : (c15872_i32));
    int32_t v28 = ((v27 > c0_i32) ? (v27) : (c0_i32));
    AscendC::GlobalTensor<float> v29 = v22[v25];
    bool v30 = v26 < c0_i32;
    int32_t v31 = v30 ? c0_i32 : v26;
    int32_t v32 = ((v28 < v31) ? (v28) : (v31));
    int32_t v33 = v32 * c4_i32;
    int32_t v34 = v3_size - v32;
    int32_t v35 = v33 % c32_i32;
    bool v36 = v35 == c0_i32;
    int32_t v37 = c32_i32 - v35;
    int32_t v38 = v36 ? c0_i32 : v37;
    int32_t v39 = v33 + v38;
    bool v40 = v39 < c63488_i32;
    if (v40) {
      get_buf(PIPE_V, 0, 0);
      AscendC::Duplicate(v7, c0_f32, c15872_i64);
      rls_buf(PIPE_V, 0, 0);
    }
    int32_t v41 = v34 * c4_i32;
    int32_t v42 = v38 / c4_i32;
    int32_t v43 = c63488_i32 - v39;
    int32_t v44 = v43 / c32_i32;
    AscendC::DataCopyExtParams v45{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v33), static_cast<uint32_t>(v41), static_cast<uint32_t>(v44), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<float> v46{c1_i32, c0_i32, static_cast<uint8_t>(v42), c0_f32};
    get_buf(PIPE_MTE2, 0, 0);
    AscendC::DataCopyPad(v7, v29, v45, v46);
    rls_buf(PIPE_MTE2, 0, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 0, 0);
    AscendC::Muls<float, 0>(v5, v7, c0_707106769_f32, c15872_i64);
    rls_buf(PIPE_V, 0, 0);
    rls_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 1, 0);
    __VEC_SCOPE__
    {
      __VEC_SCOPE__
      {

                            {
                                static constexpr AscendC::ErfConfig config = {
                                    AscendC::ErfAlgo::SUBSECTION_POLYNOMIAL_APPROXIMATION};
                                AscendC::Erf<float, false, config>(v6, v5, v5.GetSize());
                            }
                        ;
        ;
      };
      ;
    };
    rls_buf(PIPE_V, 1, 0);
    rls_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 0, 0);
    __ubuf__ float* v47 = reinterpret_cast<__ubuf__ float*>(v6.GetPhyAddr());
    __ubuf__ float* v48 = reinterpret_cast<__ubuf__ float*>(v7.GetPhyAddr());
    __VEC_SCOPE__
    {
      AscendC::Reg::RegTensor<float> v49;
      AscendC::Reg::RegTensor<float> v50;
      AscendC::Reg::RegTensor<float> v51;
      AscendC::Reg::RegTensor<float> v52;
      AscendC::Reg::RegTensor<float> v53;
      AscendC::Reg::RegTensor<float> v54;
      AscendC::Reg::RegTensor<float> v55;
      AscendC::Reg::MaskReg v56 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
      uint32_t v57[1]{c15872_idx};
      for (uint16_t v58 = 0; v58 < static_cast<uint16_t>(c248_idx); v58 += 1) {
        uint32_t v59 = v58 * c64_idx;
        // The mask is updated on every iteration to match the total count
        AscendC::Reg::MaskReg v60 = AscendC::Reg::UpdateMask<float>(*v57);
        __ubuf__ float* v61 = v47 + v59;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v49, v61);
        AscendC::Reg::Duplicate(v50, c1_f32, v56);
        AscendC::Reg::Add(v51, v49, v50, v56);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(v61, v51, v60);
        __ubuf__ float* v62 = v48 + v59;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v52, v62);
        AscendC::Reg::Duplicate(v53, c0_5_f32, v56);
        AscendC::Reg::Mul(v54, v52, v53, v56);
        AscendC::Reg::Mul(v55, v51, v54, v56);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(v62, v55, v60);
      };
      ;
    };
    rls_buf(PIPE_V, 0, 0);
    rls_buf(PIPE_V, 2, 0);
    AscendC::GlobalTensor<float> v63 = v23[v25];
    int32_t v64 = c15872_i32 - v32;
    int32_t v65 = v64 * c4_i32;
    int32_t v66 = v65 / c32_i32;
    AscendC::DataCopyExtParams v67{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v33), static_cast<uint32_t>(v66), static_cast<uint32_t>(v41), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 0, 0);
    AscendC::DataCopyPad(v63, v7, v67);
    rls_buf(PIPE_MTE3, 0, 0);
  }
  return;
}

