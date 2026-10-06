#include "kernel_operator.h"
extern "C"  __global__ __aicore__ void gelu_kernel(__gm__ half* v1_input_ptr, __gm__ half* v2_output_ptr, int32_t v3_input_length) {
  constexpr uint32_t c84_idx = 84;
  constexpr uint32_t c5376_idx = 5376;
  constexpr uint32_t c64_idx = 64;
  constexpr int32_t c72_i32 = 72;
  constexpr int32_t c1_i32 = 1;
  constexpr int32_t c512_i32 = 512;
  constexpr int32_t c0_i32 = 0;
  constexpr int32_t c5376_i32 = 5376;
  constexpr int32_t c2_i32 = 2;
  half c0_f16 = 0.0e+00;
  constexpr int32_t c71_i32 = 71;
  constexpr int32_t c511_i32 = 511;
  constexpr int32_t c5375_i32 = 5375;
  constexpr float c0_0447149985_f32 = (float)4.471499850e-02;
  constexpr float cm1_59576917_f32 = (float)-1.595769170e+00;
  constexpr float c1_f32 = (float)1.000000000e+00;
  constexpr int32_t c32_i32 = 32;
  constexpr int32_t c10752_i32 = 10752;
  constexpr int64_t c5376_i64 = 5376;
  AscendC::TPipe v4;
  AscendC::LocalTensor<half> v5{AscendC::TPosition::VECCALC, 0, 5376};
  AscendC::LocalTensor<half> v6{AscendC::TPosition::VECCALC, 10752, 5376};
  AscendC::LocalTensor<float> v7{AscendC::TPosition::VECCALC, 21504, 5376};
  AscendC::LocalTensor<float> v8{AscendC::TPosition::VECCALC, 43008, 5376};
  int32_t v9 = v3_input_length + c71_i32;
  int32_t v10 = v9 / c72_i32;
  int32_t v11 = v10 + c511_i32;
  int32_t v12 = v11 / c512_i32;
  int32_t v13 = v12 * c512_i32;
  int32_t v14 = AscendC::GetBlockIdx();
  int32_t v15 = v14 * v13;
  int32_t v16 = v15 + v13;
  int32_t v17 = ((v16 < v3_input_length) ? (v16) : (v3_input_length));
  int32_t v18 = v17 - v15;
  int32_t v19 = ((v18 > c0_i32) ? (v18) : (c0_i32));
  int32_t v20 = v19 + c5375_i32;
  int32_t v21 = v20 / c5376_i32;
  for (int32_t v22 = c0_i32; v22 < v21; v22 += c2_i32) {
    bool v23 = v22 < v21;
    if (v23) {
      AscendC::GlobalTensor<half> v24;
      v24.SetGlobalBuffer(v1_input_ptr);
      AscendC::GlobalTensor<half> v25;
      v25.SetGlobalBuffer(v2_output_ptr);
      int32_t v26 = v22 * c5376_i32;
      int32_t v27 = v15 + v26;
      int32_t v28 = v17 - v27;
      int32_t v29 = ((v28 < c5376_i32) ? (v28) : (c5376_i32));
      int32_t v30 = ((v29 > c0_i32) ? (v29) : (c0_i32));
      AscendC::GlobalTensor<half> v31 = v24[v27];
      int32_t v32 = v3_input_length - v27;
      bool v33 = v32 < c0_i32;
      int32_t v34 = v33 ? c0_i32 : v32;
      int32_t v35 = ((v30 < v34) ? (v30) : (v34));
      int32_t v36 = v35 * c2_i32;
      int32_t v37 = v3_input_length - v35;
      int32_t v38 = v36 % c32_i32;
      bool v39 = v38 == c0_i32;
      int32_t v40 = c32_i32 - v38;
      int32_t v41 = v39 ? c0_i32 : v40;
      int32_t v42 = v36 + v41;
      bool v43 = v42 < c10752_i32;
      if (v43) {
        get_buf(PIPE_V, 0, 0);
        AscendC::Duplicate(v5, c0_f16, c5376_i64);
        rls_buf(PIPE_V, 0, 0);
      }
      int32_t v44 = v37 * c2_i32;
      int32_t v45 = v41 / c2_i32;
      int32_t v46 = c10752_i32 - v42;
      int32_t v47 = v46 / c32_i32;
      AscendC::DataCopyExtParams v48{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v36), static_cast<uint32_t>(v44), static_cast<uint32_t>(v47), static_cast<uint32_t>(c0_i32)};
      AscendC::DataCopyPadExtParams<half> v49{c1_i32, c0_i32, static_cast<uint8_t>(v45), c0_f16};
      get_buf(PIPE_MTE2, 0, 0);
      AscendC::DataCopyPad(v5, v31, v48, v49);
      rls_buf(PIPE_MTE2, 0, 0);
      get_buf(PIPE_V, 1, 0);
      get_buf(PIPE_V, 0, 0);
      AscendC::Cast<float, half>(v7, v5, AscendC::RoundMode::CAST_NONE, c5376_i64);
      rls_buf(PIPE_V, 0, 0);
      rls_buf(PIPE_V, 1, 0);
      get_buf(PIPE_V, 2, 0);
      get_buf(PIPE_V, 1, 0);
      {
        __ubuf__ float* v50 = reinterpret_cast<__ubuf__ float*>(v7.GetPhyAddr());
        __ubuf__ float* v51 = reinterpret_cast<__ubuf__ float*>(v8.GetPhyAddr());
        __VEC_SCOPE__
        {
          AscendC::Reg::RegTensor<float> v52;
          AscendC::Reg::RegTensor<float> v53;
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
          AscendC::Reg::MaskReg v64 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
          uint32_t v65[1]{c5376_idx};
          AscendC::Reg::Duplicate(v55, c0_0447149985_f32, v64);
          AscendC::Reg::Duplicate(v58, cm1_59576917_f32, v64);
          AscendC::Reg::Duplicate(v61, c1_f32, v64);
          for (uint16_t v66 = 0; v66 < static_cast<uint16_t>(c84_idx); v66 += 1) {
            uint32_t v67 = v66 * c64_idx;
            // The mask is updated on every iteration to match the total count
            AscendC::Reg::MaskReg v68 = AscendC::Reg::UpdateMask<float>(*v65);
            __ubuf__ float* v69 = v50 + v67;
            AscendC::Reg::DataCopy(v52, v69);
            AscendC::Reg::Mul(v53, v52, v52, v64);
            AscendC::Reg::Mul(v54, v53, v52, v64);
            AscendC::Reg::Mul(v56, v54, v55, v64);
            AscendC::Reg::Add(v57, v52, v56, v64);
            AscendC::Reg::Mul(v59, v57, v58, v64);
            AscendC::Reg::Exp(v60, v59, v64);
            AscendC::Reg::Add(v62, v60, v61, v64);
            AscendC::Reg::Div(v63, v52, v62, v64);
            __ubuf__ float* v70 = v51 + v67;
            AscendC::Reg::DataCopy(v70, v63, v68);
          }
        }
      }
      rls_buf(PIPE_V, 1, 0);
      rls_buf(PIPE_V, 2, 0);
      get_buf(PIPE_V, 0, 0);
      get_buf(PIPE_V, 2, 0);
      AscendC::Cast<half, float>(v5, v8, AscendC::RoundMode::CAST_RINT, c5376_i64);
      rls_buf(PIPE_V, 2, 0);
      rls_buf(PIPE_V, 0, 0);
      AscendC::GlobalTensor<half> v71 = v25[v27];
      int32_t v72 = c5376_i32 - v35;
      int32_t v73 = v72 * c2_i32;
      int32_t v74 = v73 / c32_i32;
      AscendC::DataCopyExtParams v75{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v36), static_cast<uint32_t>(v74), static_cast<uint32_t>(v44), static_cast<uint32_t>(c0_i32)};
      get_buf(PIPE_MTE3, 0, 0);
      AscendC::DataCopyPad(v71, v5, v75);
      rls_buf(PIPE_MTE3, 0, 0);
    }
    int32_t v76 = v22 + c1_i32;
    bool v77 = v76 < v21;
    if (v77) {
      AscendC::GlobalTensor<half> v78;
      v78.SetGlobalBuffer(v1_input_ptr);
      AscendC::GlobalTensor<half> v79;
      v79.SetGlobalBuffer(v2_output_ptr);
      int32_t v80 = v76 * c5376_i32;
      int32_t v81 = v15 + v80;
      int32_t v82 = v17 - v81;
      int32_t v83 = ((v82 < c5376_i32) ? (v82) : (c5376_i32));
      int32_t v84 = ((v83 > c0_i32) ? (v83) : (c0_i32));
      AscendC::GlobalTensor<half> v85 = v78[v81];
      int32_t v86 = v3_input_length - v81;
      bool v87 = v86 < c0_i32;
      int32_t v88 = v87 ? c0_i32 : v86;
      int32_t v89 = ((v84 < v88) ? (v84) : (v88));
      int32_t v90 = v89 * c2_i32;
      int32_t v91 = v3_input_length - v89;
      int32_t v92 = v90 % c32_i32;
      bool v93 = v92 == c0_i32;
      int32_t v94 = c32_i32 - v92;
      int32_t v95 = v93 ? c0_i32 : v94;
      int32_t v96 = v90 + v95;
      bool v97 = v96 < c10752_i32;
      if (v97) {
        get_buf(PIPE_V, 3, 0);
        AscendC::Duplicate(v6, c0_f16, c5376_i64);
        rls_buf(PIPE_V, 3, 0);
      }
      int32_t v98 = v91 * c2_i32;
      int32_t v99 = v95 / c2_i32;
      int32_t v100 = c10752_i32 - v96;
      int32_t v101 = v100 / c32_i32;
      AscendC::DataCopyExtParams v102{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v90), static_cast<uint32_t>(v98), static_cast<uint32_t>(v101), static_cast<uint32_t>(c0_i32)};
      AscendC::DataCopyPadExtParams<half> v103{c1_i32, c0_i32, static_cast<uint8_t>(v99), c0_f16};
      get_buf(PIPE_MTE2, 3, 0);
      AscendC::DataCopyPad(v6, v85, v102, v103);
      rls_buf(PIPE_MTE2, 3, 0);
      get_buf(PIPE_V, 1, 0);
      get_buf(PIPE_V, 3, 0);
      AscendC::Cast<float, half>(v7, v6, AscendC::RoundMode::CAST_NONE, c5376_i64);
      rls_buf(PIPE_V, 3, 0);
      rls_buf(PIPE_V, 1, 0);
      get_buf(PIPE_V, 2, 0);
      get_buf(PIPE_V, 1, 0);
      {
        __ubuf__ float* v104 = reinterpret_cast<__ubuf__ float*>(v7.GetPhyAddr());
        __ubuf__ float* v105 = reinterpret_cast<__ubuf__ float*>(v8.GetPhyAddr());
        __VEC_SCOPE__
        {
          AscendC::Reg::RegTensor<float> v106;
          AscendC::Reg::RegTensor<float> v107;
          AscendC::Reg::RegTensor<float> v108;
          AscendC::Reg::RegTensor<float> v109;
          AscendC::Reg::RegTensor<float> v110;
          AscendC::Reg::RegTensor<float> v111;
          AscendC::Reg::RegTensor<float> v112;
          AscendC::Reg::RegTensor<float> v113;
          AscendC::Reg::RegTensor<float> v114;
          AscendC::Reg::RegTensor<float> v115;
          AscendC::Reg::RegTensor<float> v116;
          AscendC::Reg::RegTensor<float> v117;
          AscendC::Reg::MaskReg v118 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
          uint32_t v119[1]{c5376_idx};
          AscendC::Reg::Duplicate(v109, c0_0447149985_f32, v118);
          AscendC::Reg::Duplicate(v112, cm1_59576917_f32, v118);
          AscendC::Reg::Duplicate(v115, c1_f32, v118);
          for (uint16_t v120 = 0; v120 < static_cast<uint16_t>(c84_idx); v120 += 1) {
            uint32_t v121 = v120 * c64_idx;
            // The mask is updated on every iteration to match the total count
            AscendC::Reg::MaskReg v122 = AscendC::Reg::UpdateMask<float>(*v119);
            __ubuf__ float* v123 = v104 + v121;
            AscendC::Reg::DataCopy(v106, v123);
            AscendC::Reg::Mul(v107, v106, v106, v118);
            AscendC::Reg::Mul(v108, v107, v106, v118);
            AscendC::Reg::Mul(v110, v108, v109, v118);
            AscendC::Reg::Add(v111, v106, v110, v118);
            AscendC::Reg::Mul(v113, v111, v112, v118);
            AscendC::Reg::Exp(v114, v113, v118);
            AscendC::Reg::Add(v116, v114, v115, v118);
            AscendC::Reg::Div(v117, v106, v116, v118);
            __ubuf__ float* v124 = v105 + v121;
            AscendC::Reg::DataCopy(v124, v117, v122);
          }
        }
      }
      rls_buf(PIPE_V, 1, 0);
      rls_buf(PIPE_V, 2, 0);
      get_buf(PIPE_V, 3, 0);
      get_buf(PIPE_V, 2, 0);
      AscendC::Cast<half, float>(v6, v8, AscendC::RoundMode::CAST_RINT, c5376_i64);
      rls_buf(PIPE_V, 2, 0);
      rls_buf(PIPE_V, 3, 0);
      AscendC::GlobalTensor<half> v125 = v79[v81];
      int32_t v126 = c5376_i32 - v89;
      int32_t v127 = v126 * c2_i32;
      int32_t v128 = v127 / c32_i32;
      AscendC::DataCopyExtParams v129{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v90), static_cast<uint32_t>(v128), static_cast<uint32_t>(v98), static_cast<uint32_t>(c0_i32)};
      get_buf(PIPE_MTE3, 3, 0);
      AscendC::DataCopyPad(v125, v6, v129);
      rls_buf(PIPE_MTE3, 3, 0);
    }
  }
  return;
}

