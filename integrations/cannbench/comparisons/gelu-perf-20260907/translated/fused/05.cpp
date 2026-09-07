#include "kernel_operator.h"
extern "C"  __global__ __aicore__ void _gelu_fused(__gm__ bfloat16_t* v1_x_ptr, __gm__ bfloat16_t* v2_y_ptr, int32_t v3_size) {
  constexpr uint32_t c16_idx = 16;
  constexpr uint32_t c1024_idx = 1024;
  constexpr uint32_t c64_idx = 64;
  constexpr int64_t c1024_i64 = 1024;
  constexpr int32_t c2048_i32 = 2048;
  constexpr int32_t c32_i32 = 32;
  constexpr int32_t c2_i32 = 2;
  constexpr int32_t c1_i32 = 1;
  constexpr int32_t c0_i32 = 0;
  constexpr float c1_f32 = (float)1.000000000e+00;
  constexpr float cm1_f32 = (float)-1.000000000e+00;
  constexpr float c1_59576917_f32 = (float)1.595769170e+00;
  constexpr float c0_0713548139_f32 = (float)7.135481390e-02;
  constexpr int32_t c1023_i32 = 1023;
  bfloat16_t c0_bf16 = 0.0e+00;
  constexpr int32_t c1024_i32 = 1024;
  AscendC::TPipe v4;
  int32_t v5 = AscendC::GetBlockIdx();
  int32_t v6 = v3_size + c1023_i32;
  int32_t v7 = v6 / c1024_i32;
  int32_t v8 = AscendC::GetBlockNum();
  AscendC::LocalTensor<bfloat16_t> v9{AscendC::TPosition::VECCALC, 0, 1024};
  AscendC::LocalTensor<float> v10{AscendC::TPosition::VECCALC, 2048, 1024};
  AscendC::LocalTensor<float> v11{AscendC::TPosition::VECCALC, 6144, 1024};
  AscendC::LocalTensor<float> v12{AscendC::TPosition::VECCALC, 10240, 1024};
  AscendC::LocalTensor<float> v13{AscendC::TPosition::VECCALC, 14336, 1024};
  AscendC::LocalTensor<float> v14{AscendC::TPosition::VECCALC, 18432, 1024};
  AscendC::LocalTensor<float> v15{AscendC::TPosition::VECCALC, 22528, 1024};
  AscendC::LocalTensor<float> v16{AscendC::TPosition::VECCALC, 26624, 1024};
  AscendC::LocalTensor<float> v17{AscendC::TPosition::VECCALC, 30720, 1024};
  AscendC::LocalTensor<float> v18{AscendC::TPosition::VECCALC, 34816, 1024};
  AscendC::LocalTensor<bfloat16_t> v19{AscendC::TPosition::VECCALC, 38912, 1024};
  for (int32_t v20 = v5; v20 < v7; v20 += v8) {
    AscendC::GlobalTensor<bfloat16_t> v21;
    v21.SetGlobalBuffer(v1_x_ptr);
    AscendC::GlobalTensor<bfloat16_t> v22;
    v22.SetGlobalBuffer(v2_y_ptr);
    int32_t v23 = v20 * c1024_i32;
    int32_t v24 = v23 + c1024_i32;
    bool v25 = v24 <= v3_size;
    int32_t v26;
    if (v25) {
      v26 = c1024_i32;
    } else {
      int32_t v27 = v3_size - v23;
      v26 = v27;
    }
    AscendC::GlobalTensor<bfloat16_t> v28 = v21[v23];
    int32_t v29 = v3_size - v23;
    bool v30 = v29 < c0_i32;
    int32_t v31 = v30 ? c0_i32 : v29;
    int32_t v32 = ((v26 < v31) ? (v26) : (v31));
    int32_t v33 = v32 * c2_i32;
    int32_t v34 = v3_size - v32;
    int32_t v35 = v33 % c32_i32;
    bool v36 = v35 == c0_i32;
    int32_t v37 = c32_i32 - v35;
    int32_t v38 = v36 ? c0_i32 : v37;
    int32_t v39 = v33 + v38;
    bool v40 = v39 < c2048_i32;
    if (v40) {
      get_buf(PIPE_V, 0, 0);
      AscendC::Duplicate(v19, c0_bf16, c1024_i64);
      rls_buf(PIPE_V, 0, 0);
    }
    int32_t v41 = v34 * c2_i32;
    int32_t v42 = v38 / c2_i32;
    int32_t v43 = c2048_i32 - v39;
    int32_t v44 = v43 / c32_i32;
    AscendC::DataCopyExtParams v45{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v33), static_cast<uint32_t>(v41), static_cast<uint32_t>(v44), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<bfloat16_t> v46{c1_i32, c0_i32, static_cast<uint8_t>(v42), c0_bf16};
    get_buf(PIPE_MTE2, 0, 0);
    AscendC::DataCopyPad(v19, v28, v45, v46);
    rls_buf(PIPE_MTE2, 0, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 0, 0);
    AscendC::Cast<float, bfloat16_t>(v18, v19, AscendC::RoundMode::CAST_NONE, c1024_i64);
    rls_buf(PIPE_V, 0, 0);
    rls_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 3, 0);
    get_buf(PIPE_V, 4, 0);
    get_buf(PIPE_V, 5, 0);
    get_buf(PIPE_V, 6, 0);
    get_buf(PIPE_V, 7, 0);
    get_buf(PIPE_V, 8, 0);
    get_buf(PIPE_V, 9, 0);
    get_buf(PIPE_V, 1, 0);
    {
      __ubuf__ float* v47 = reinterpret_cast<__ubuf__ float*>(v18.GetPhyAddr());
      __ubuf__ float* v48 = reinterpret_cast<__ubuf__ float*>(v17.GetPhyAddr());
      __ubuf__ float* v49 = reinterpret_cast<__ubuf__ float*>(v16.GetPhyAddr());
      __ubuf__ float* v50 = reinterpret_cast<__ubuf__ float*>(v15.GetPhyAddr());
      __ubuf__ float* v51 = reinterpret_cast<__ubuf__ float*>(v14.GetPhyAddr());
      __ubuf__ float* v52 = reinterpret_cast<__ubuf__ float*>(v13.GetPhyAddr());
      __ubuf__ float* v53 = reinterpret_cast<__ubuf__ float*>(v12.GetPhyAddr());
      __ubuf__ float* v54 = reinterpret_cast<__ubuf__ float*>(v11.GetPhyAddr());
      __ubuf__ float* v55 = reinterpret_cast<__ubuf__ float*>(v10.GetPhyAddr());
      __VEC_SCOPE__
      {
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
        AscendC::Reg::RegTensor<float> v67;
        AscendC::Reg::RegTensor<float> v68;
        AscendC::Reg::MaskReg v69 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t v70[1]{c1024_idx};
        AscendC::Reg::Duplicate(v58, c0_0713548139_f32, v69);
        AscendC::Reg::Duplicate(v60, c1_59576917_f32, v69);
        AscendC::Reg::Duplicate(v63, cm1_f32, v69);
        AscendC::Reg::Duplicate(v66, c1_f32, v69);
        for (uint16_t v71 = 0; v71 < static_cast<uint16_t>(c16_idx); v71 += 1) {
          uint32_t v72 = v71 * c64_idx;
          // The mask is updated on every iteration to match the total count
          AscendC::Reg::MaskReg v73 = AscendC::Reg::UpdateMask<float>(*v70);
          __ubuf__ float* v74 = v47 + v72;
          AscendC::Reg::DataCopy(v56, v74);
          AscendC::Reg::Mul(v57, v56, v56, v69);
          __ubuf__ float* v75 = v48 + v72;
          AscendC::Reg::DataCopy(v75, v57, v73);
          AscendC::Reg::Mul(v59, v57, v58, v69);
          __ubuf__ float* v76 = v49 + v72;
          AscendC::Reg::DataCopy(v76, v59, v73);
          AscendC::Reg::Add(v61, v59, v60, v69);
          __ubuf__ float* v77 = v50 + v72;
          AscendC::Reg::DataCopy(v77, v61, v73);
          AscendC::Reg::Mul(v62, v61, v56, v69);
          __ubuf__ float* v78 = v51 + v72;
          AscendC::Reg::DataCopy(v78, v62, v73);
          AscendC::Reg::Mul(v64, v62, v63, v69);
          __ubuf__ float* v79 = v52 + v72;
          AscendC::Reg::DataCopy(v79, v64, v73);
          AscendC::Reg::Exp(v65, v64, v69);
          __ubuf__ float* v80 = v53 + v72;
          AscendC::Reg::DataCopy(v80, v65, v73);
          AscendC::Reg::Add(v67, v65, v66, v69);
          __ubuf__ float* v81 = v54 + v72;
          AscendC::Reg::DataCopy(v81, v67, v73);
          AscendC::Reg::Div(v68, v56, v67, v69);
          __ubuf__ float* v82 = v55 + v72;
          AscendC::Reg::DataCopy(v82, v68, v73);
        }
      }
    }
    rls_buf(PIPE_V, 1, 0);
    rls_buf(PIPE_V, 9, 0);
    rls_buf(PIPE_V, 8, 0);
    rls_buf(PIPE_V, 7, 0);
    rls_buf(PIPE_V, 6, 0);
    rls_buf(PIPE_V, 5, 0);
    rls_buf(PIPE_V, 4, 0);
    rls_buf(PIPE_V, 3, 0);
    rls_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 10, 0);
    get_buf(PIPE_V, 9, 0);
    AscendC::Cast<bfloat16_t, float>(v9, v10, AscendC::RoundMode::CAST_RINT, c1024_i64);
    rls_buf(PIPE_V, 9, 0);
    rls_buf(PIPE_V, 10, 0);
    AscendC::GlobalTensor<bfloat16_t> v83 = v22[v23];
    int32_t v84 = c1024_i32 - v32;
    int32_t v85 = v84 * c2_i32;
    int32_t v86 = v85 / c32_i32;
    AscendC::DataCopyExtParams v87{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v33), static_cast<uint32_t>(v86), static_cast<uint32_t>(v41), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 10, 0);
    AscendC::DataCopyPad(v83, v9, v87);
    rls_buf(PIPE_MTE3, 10, 0);
  }
  return;
}
