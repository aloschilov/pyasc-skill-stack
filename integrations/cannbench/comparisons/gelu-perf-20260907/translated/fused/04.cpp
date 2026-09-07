#include "kernel_operator.h"
extern "C"  __global__ __aicore__ void _gelu_fused(__gm__ float* v1_x_ptr, __gm__ float* v2_y_ptr, int32_t v3_size) {
  constexpr uint32_t c16_idx = 16;
  constexpr uint32_t c1024_idx = 1024;
  constexpr uint32_t c64_idx = 64;
  constexpr int64_t c1024_i64 = 1024;
  constexpr int32_t c4096_i32 = 4096;
  constexpr int32_t c32_i32 = 32;
  constexpr int32_t c4_i32 = 4;
  constexpr int32_t c1_i32 = 1;
  constexpr int32_t c0_i32 = 0;
  constexpr float c1_f32 = (float)1.000000000e+00;
  constexpr float cm1_f32 = (float)-1.000000000e+00;
  constexpr float c1_59576917_f32 = (float)1.595769170e+00;
  constexpr float c0_0713548139_f32 = (float)7.135481390e-02;
  constexpr int32_t c1023_i32 = 1023;
  constexpr float c0_f32 = (float)0.0e+00;
  constexpr int32_t c1024_i32 = 1024;
  AscendC::TPipe v4;
  int32_t v5 = AscendC::GetBlockIdx();
  int32_t v6 = v3_size + c1023_i32;
  int32_t v7 = v6 / c1024_i32;
  int32_t v8 = AscendC::GetBlockNum();
  AscendC::LocalTensor<float> v9{AscendC::TPosition::VECCALC, 0, 1024};
  AscendC::LocalTensor<float> v10{AscendC::TPosition::VECCALC, 4096, 1024};
  AscendC::LocalTensor<float> v11{AscendC::TPosition::VECCALC, 8192, 1024};
  AscendC::LocalTensor<float> v12{AscendC::TPosition::VECCALC, 12288, 1024};
  AscendC::LocalTensor<float> v13{AscendC::TPosition::VECCALC, 16384, 1024};
  AscendC::LocalTensor<float> v14{AscendC::TPosition::VECCALC, 20480, 1024};
  AscendC::LocalTensor<float> v15{AscendC::TPosition::VECCALC, 24576, 1024};
  AscendC::LocalTensor<float> v16{AscendC::TPosition::VECCALC, 28672, 1024};
  AscendC::LocalTensor<float> v17{AscendC::TPosition::VECCALC, 32768, 1024};
  for (int32_t v18 = v5; v18 < v7; v18 += v8) {
    AscendC::GlobalTensor<float> v19;
    v19.SetGlobalBuffer(v1_x_ptr);
    AscendC::GlobalTensor<float> v20;
    v20.SetGlobalBuffer(v2_y_ptr);
    int32_t v21 = v18 * c1024_i32;
    int32_t v22 = v21 + c1024_i32;
    bool v23 = v22 <= v3_size;
    int32_t v24;
    if (v23) {
      v24 = c1024_i32;
    } else {
      int32_t v25 = v3_size - v21;
      v24 = v25;
    }
    AscendC::GlobalTensor<float> v26 = v19[v21];
    int32_t v27 = v3_size - v21;
    bool v28 = v27 < c0_i32;
    int32_t v29 = v28 ? c0_i32 : v27;
    int32_t v30 = ((v24 < v29) ? (v24) : (v29));
    int32_t v31 = v30 * c4_i32;
    int32_t v32 = v3_size - v30;
    int32_t v33 = v31 % c32_i32;
    bool v34 = v33 == c0_i32;
    int32_t v35 = c32_i32 - v33;
    int32_t v36 = v34 ? c0_i32 : v35;
    int32_t v37 = v31 + v36;
    bool v38 = v37 < c4096_i32;
    if (v38) {
      get_buf(PIPE_V, 0, 0);
      AscendC::Duplicate(v17, c0_f32, c1024_i64);
      rls_buf(PIPE_V, 0, 0);
    }
    int32_t v39 = v32 * c4_i32;
    int32_t v40 = v36 / c4_i32;
    int32_t v41 = c4096_i32 - v37;
    int32_t v42 = v41 / c32_i32;
    AscendC::DataCopyExtParams v43{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v31), static_cast<uint32_t>(v39), static_cast<uint32_t>(v42), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<float> v44{c1_i32, c0_i32, static_cast<uint8_t>(v40), c0_f32};
    get_buf(PIPE_MTE2, 0, 0);
    AscendC::DataCopyPad(v17, v26, v43, v44);
    rls_buf(PIPE_MTE2, 0, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 3, 0);
    get_buf(PIPE_V, 4, 0);
    get_buf(PIPE_V, 5, 0);
    get_buf(PIPE_V, 6, 0);
    get_buf(PIPE_V, 7, 0);
    get_buf(PIPE_V, 8, 0);
    get_buf(PIPE_V, 0, 0);
    {
      __ubuf__ float* v45 = reinterpret_cast<__ubuf__ float*>(v17.GetPhyAddr());
      __ubuf__ float* v46 = reinterpret_cast<__ubuf__ float*>(v16.GetPhyAddr());
      __ubuf__ float* v47 = reinterpret_cast<__ubuf__ float*>(v15.GetPhyAddr());
      __ubuf__ float* v48 = reinterpret_cast<__ubuf__ float*>(v14.GetPhyAddr());
      __ubuf__ float* v49 = reinterpret_cast<__ubuf__ float*>(v13.GetPhyAddr());
      __ubuf__ float* v50 = reinterpret_cast<__ubuf__ float*>(v12.GetPhyAddr());
      __ubuf__ float* v51 = reinterpret_cast<__ubuf__ float*>(v11.GetPhyAddr());
      __ubuf__ float* v52 = reinterpret_cast<__ubuf__ float*>(v10.GetPhyAddr());
      __ubuf__ float* v53 = reinterpret_cast<__ubuf__ float*>(v9.GetPhyAddr());
      __VEC_SCOPE__
      {
        AscendC::Reg::RegTensor<float> v54;
        AscendC::Reg::RegTensor<float> v55;
        AscendC::Reg::RegTensor<float> v56;
        AscendC::Reg::RegTensor<float> v57;
        AscendC::Reg::RegTensor<float> v58;
        AscendC::Reg::RegTensor<float> v59;
        AscendC::Reg::RegTensor<float> v60;
        AscendC::Reg::RegTensor<float> v61;
        AscendC::Reg::RegTensor<float> v62;
        AscendC::Reg::RegTensor<float> v63;
        AscendC::Reg::RegTensor<float> v64;
        AscendC::Reg::RegTensor<float> v65;
        AscendC::Reg::RegTensor<float> v66;
        AscendC::Reg::MaskReg v67 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t v68[1]{c1024_idx};
        AscendC::Reg::Duplicate(v56, c0_0713548139_f32, v67);
        AscendC::Reg::Duplicate(v58, c1_59576917_f32, v67);
        AscendC::Reg::Duplicate(v61, cm1_f32, v67);
        AscendC::Reg::Duplicate(v64, c1_f32, v67);
        for (uint16_t v69 = 0; v69 < static_cast<uint16_t>(c16_idx); v69 += 1) {
          uint32_t v70 = v69 * c64_idx;
          // The mask is updated on every iteration to match the total count
          AscendC::Reg::MaskReg v71 = AscendC::Reg::UpdateMask<float>(*v68);
          __ubuf__ float* v72 = v45 + v70;
          AscendC::Reg::DataCopy(v54, v72);
          AscendC::Reg::Mul(v55, v54, v54, v67);
          __ubuf__ float* v73 = v46 + v70;
          AscendC::Reg::DataCopy(v73, v55, v71);
          AscendC::Reg::Mul(v57, v55, v56, v67);
          __ubuf__ float* v74 = v47 + v70;
          AscendC::Reg::DataCopy(v74, v57, v71);
          AscendC::Reg::Add(v59, v57, v58, v67);
          __ubuf__ float* v75 = v48 + v70;
          AscendC::Reg::DataCopy(v75, v59, v71);
          AscendC::Reg::Mul(v60, v59, v54, v67);
          __ubuf__ float* v76 = v49 + v70;
          AscendC::Reg::DataCopy(v76, v60, v71);
          AscendC::Reg::Mul(v62, v60, v61, v67);
          __ubuf__ float* v77 = v50 + v70;
          AscendC::Reg::DataCopy(v77, v62, v71);
          AscendC::Reg::Exp(v63, v62, v67);
          __ubuf__ float* v78 = v51 + v70;
          AscendC::Reg::DataCopy(v78, v63, v71);
          AscendC::Reg::Add(v65, v63, v64, v67);
          __ubuf__ float* v79 = v52 + v70;
          AscendC::Reg::DataCopy(v79, v65, v71);
          AscendC::Reg::Div(v66, v54, v65, v67);
          __ubuf__ float* v80 = v53 + v70;
          AscendC::Reg::DataCopy(v80, v66, v71);
        }
      }
    }
    rls_buf(PIPE_V, 0, 0);
    rls_buf(PIPE_V, 8, 0);
    rls_buf(PIPE_V, 7, 0);
    rls_buf(PIPE_V, 6, 0);
    rls_buf(PIPE_V, 5, 0);
    rls_buf(PIPE_V, 4, 0);
    rls_buf(PIPE_V, 3, 0);
    rls_buf(PIPE_V, 2, 0);
    rls_buf(PIPE_V, 1, 0);
    AscendC::GlobalTensor<float> v81 = v20[v21];
    int32_t v82 = c1024_i32 - v30;
    int32_t v83 = v82 * c4_i32;
    int32_t v84 = v83 / c32_i32;
    AscendC::DataCopyExtParams v85{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v31), static_cast<uint32_t>(v84), static_cast<uint32_t>(v39), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 8, 0);
    AscendC::DataCopyPad(v81, v9, v85);
    rls_buf(PIPE_MTE3, 8, 0);
  }
  return;
}
