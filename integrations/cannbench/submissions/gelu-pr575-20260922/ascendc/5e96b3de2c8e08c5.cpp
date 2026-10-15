#include "kernel_operator.h"
extern "C"  __global__ __aicore__ void gelu_kernel(__gm__ float* v1_input_ptr, __gm__ float* v2_output_ptr, int32_t v3_input_length) {
  constexpr uint32_t c85_idx = 85;
  constexpr uint32_t c5440_idx = 5440;
  constexpr uint32_t c64_idx = 64;
  constexpr int32_t c72_i32 = 72;
  constexpr int32_t c1_i32 = 1;
  constexpr int32_t c512_i32 = 512;
  constexpr int32_t c0_i32 = 0;
  constexpr int32_t c5440_i32 = 5440;
  constexpr int32_t c2_i32 = 2;
  constexpr float c0_f32 = (float)0.0e+00;
  constexpr int32_t c71_i32 = 71;
  constexpr int32_t c511_i32 = 511;
  constexpr int32_t c5439_i32 = 5439;
  constexpr float cm0_707106769_f32 = (float)-7.071067690e-01;
  constexpr float c0_5_f32 = (float)5.000000000e-01;
  constexpr int32_t c4_i32 = 4;
  constexpr int32_t c32_i32 = 32;
  constexpr int32_t c21760_i32 = 21760;
  constexpr int64_t c5440_i64 = 5440;
  AscendC::TPipe v4;
  AscendC::LocalTensor<float> v5{AscendC::TPosition::VECCALC, 0, 5440};
  AscendC::LocalTensor<float> v6{AscendC::TPosition::VECCALC, 21760, 5440};
  AscendC::LocalTensor<float> v7{AscendC::TPosition::VECCALC, 43520, 5440};
  AscendC::LocalTensor<float> v8{AscendC::TPosition::VECCALC, 65280, 5440};
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
  int32_t v20 = v19 + c5439_i32;
  int32_t v21 = v20 / c5440_i32;
  for (int32_t v22 = c0_i32; v22 < v21; v22 += c2_i32) {
    bool v23 = v22 < v21;
    if (v23) {
      AscendC::GlobalTensor<float> v24;
      v24.SetGlobalBuffer(v1_input_ptr);
      AscendC::GlobalTensor<float> v25;
      v25.SetGlobalBuffer(v2_output_ptr);
      int32_t v26 = v22 * c5440_i32;
      int32_t v27 = v15 + v26;
      int32_t v28 = v17 - v27;
      int32_t v29 = ((v28 < c5440_i32) ? (v28) : (c5440_i32));
      int32_t v30 = ((v29 > c0_i32) ? (v29) : (c0_i32));
      AscendC::GlobalTensor<float> v31 = v24[v27];
      int32_t v32 = v3_input_length - v27;
      bool v33 = v32 < c0_i32;
      int32_t v34 = v33 ? c0_i32 : v32;
      int32_t v35 = ((v30 < v34) ? (v30) : (v34));
      int32_t v36 = v35 * c4_i32;
      int32_t v37 = v3_input_length - v35;
      int32_t v38 = v36 % c32_i32;
      bool v39 = v38 == c0_i32;
      int32_t v40 = c32_i32 - v38;
      int32_t v41 = v39 ? c0_i32 : v40;
      int32_t v42 = v36 + v41;
      bool v43 = v42 < c21760_i32;
      if (v43) {
        get_buf(PIPE_V, 0, 0);
        AscendC::Duplicate(v5, c0_f32, c5440_i64);
        rls_buf(PIPE_V, 0, 0);
      }
      int32_t v44 = v37 * c4_i32;
      int32_t v45 = v41 / c4_i32;
      int32_t v46 = c21760_i32 - v42;
      int32_t v47 = v46 / c32_i32;
      AscendC::DataCopyExtParams v48{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v36), static_cast<uint32_t>(v44), static_cast<uint32_t>(v47), static_cast<uint32_t>(c0_i32)};
      AscendC::DataCopyPadExtParams<float> v49{c1_i32, c0_i32, static_cast<uint8_t>(v45), c0_f32};
      get_buf(PIPE_MTE2, 0, 0);
      AscendC::DataCopyPad(v5, v31, v48, v49);
      rls_buf(PIPE_MTE2, 0, 0);
      get_buf(PIPE_V, 1, 0);
      get_buf(PIPE_V, 0, 0);
      AscendC::Muls<float, 0>(v7, v5, cm0_707106769_f32, c5440_i64);
      rls_buf(PIPE_V, 0, 0);
      rls_buf(PIPE_V, 1, 0);
      get_buf(PIPE_V, 2, 0);
      get_buf(PIPE_V, 1, 0);
      {
        __VEC_SCOPE__
        {
          __VEC_SCOPE__
          {

                                    {
                                        AscendC::Erfc<float, false>(v6, v7, v7.GetSize());
                                    }
                                ;
          }
        }
      }
      rls_buf(PIPE_V, 1, 0);
      rls_buf(PIPE_V, 2, 0);
      get_buf(PIPE_V, 1, 0);
      get_buf(PIPE_V, 0, 0);
      get_buf(PIPE_V, 2, 0);
      {
        __ubuf__ float* v50 = reinterpret_cast<__ubuf__ float*>(v5.GetPhyAddr());
        __ubuf__ float* v51 = reinterpret_cast<__ubuf__ float*>(v7.GetPhyAddr());
        __ubuf__ float* v52 = reinterpret_cast<__ubuf__ float*>(v6.GetPhyAddr());
        __VEC_SCOPE__
        {
          AscendC::Reg::RegTensor<float> v53;
          AscendC::Reg::RegTensor<float> v54;
          AscendC::Reg::RegTensor<float> v55;
          AscendC::Reg::RegTensor<float> v56;
          AscendC::Reg::RegTensor<float> v57;
          AscendC::Reg::MaskReg v58 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
          uint32_t v59[1]{c5440_idx};
          AscendC::Reg::Duplicate(v54, c0_5_f32, v58);
          for (uint16_t v60 = 0; v60 < static_cast<uint16_t>(c85_idx); v60 += 1) {
            uint32_t v61 = v60 * c64_idx;
            // The mask is updated on every iteration to match the total count
            AscendC::Reg::MaskReg v62 = AscendC::Reg::UpdateMask<float>(*v59);
            __ubuf__ float* v63 = v50 + v61;
            AscendC::Reg::DataCopy(v53, v63);
            AscendC::Reg::Mul(v55, v53, v54, v58);
            __ubuf__ float* v64 = v51 + v61;
            AscendC::Reg::DataCopy(v64, v55, v62);
            __ubuf__ float* v65 = v52 + v61;
            AscendC::Reg::DataCopy(v56, v65);
            AscendC::Reg::Mul(v57, v55, v56, v58);
            AscendC::Reg::DataCopy(v63, v57, v62);
          }
        }
      }
      rls_buf(PIPE_V, 2, 0);
      rls_buf(PIPE_V, 0, 0);
      rls_buf(PIPE_V, 1, 0);
      AscendC::GlobalTensor<float> v66 = v25[v27];
      int32_t v67 = c5440_i32 - v35;
      int32_t v68 = v67 * c4_i32;
      int32_t v69 = v68 / c32_i32;
      AscendC::DataCopyExtParams v70{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v36), static_cast<uint32_t>(v69), static_cast<uint32_t>(v44), static_cast<uint32_t>(c0_i32)};
      get_buf(PIPE_MTE3, 0, 0);
      AscendC::DataCopyPad(v66, v5, v70);
      rls_buf(PIPE_MTE3, 0, 0);
    }
    int32_t v71 = v22 + c1_i32;
    bool v72 = v71 < v21;
    if (v72) {
      AscendC::GlobalTensor<float> v73;
      v73.SetGlobalBuffer(v1_input_ptr);
      AscendC::GlobalTensor<float> v74;
      v74.SetGlobalBuffer(v2_output_ptr);
      int32_t v75 = v71 * c5440_i32;
      int32_t v76 = v15 + v75;
      int32_t v77 = v17 - v76;
      int32_t v78 = ((v77 < c5440_i32) ? (v77) : (c5440_i32));
      int32_t v79 = ((v78 > c0_i32) ? (v78) : (c0_i32));
      AscendC::GlobalTensor<float> v80 = v73[v76];
      int32_t v81 = v3_input_length - v76;
      bool v82 = v81 < c0_i32;
      int32_t v83 = v82 ? c0_i32 : v81;
      int32_t v84 = ((v79 < v83) ? (v79) : (v83));
      int32_t v85 = v84 * c4_i32;
      int32_t v86 = v3_input_length - v84;
      int32_t v87 = v85 % c32_i32;
      bool v88 = v87 == c0_i32;
      int32_t v89 = c32_i32 - v87;
      int32_t v90 = v88 ? c0_i32 : v89;
      int32_t v91 = v85 + v90;
      bool v92 = v91 < c21760_i32;
      if (v92) {
        get_buf(PIPE_V, 3, 0);
        AscendC::Duplicate(v8, c0_f32, c5440_i64);
        rls_buf(PIPE_V, 3, 0);
      }
      int32_t v93 = v86 * c4_i32;
      int32_t v94 = v90 / c4_i32;
      int32_t v95 = c21760_i32 - v91;
      int32_t v96 = v95 / c32_i32;
      AscendC::DataCopyExtParams v97{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v85), static_cast<uint32_t>(v93), static_cast<uint32_t>(v96), static_cast<uint32_t>(c0_i32)};
      AscendC::DataCopyPadExtParams<float> v98{c1_i32, c0_i32, static_cast<uint8_t>(v94), c0_f32};
      get_buf(PIPE_MTE2, 3, 0);
      AscendC::DataCopyPad(v8, v80, v97, v98);
      rls_buf(PIPE_MTE2, 3, 0);
      get_buf(PIPE_V, 2, 0);
      get_buf(PIPE_V, 3, 0);
      AscendC::Muls<float, 0>(v6, v8, cm0_707106769_f32, c5440_i64);
      rls_buf(PIPE_V, 3, 0);
      rls_buf(PIPE_V, 2, 0);
      get_buf(PIPE_V, 1, 0);
      get_buf(PIPE_V, 2, 0);
      {
        __VEC_SCOPE__
        {
          __VEC_SCOPE__
          {

                                    {
                                        AscendC::Erfc<float, false>(v7, v6, v6.GetSize());
                                    }
                                ;
          }
        }
      }
      rls_buf(PIPE_V, 2, 0);
      rls_buf(PIPE_V, 1, 0);
      get_buf(PIPE_V, 3, 0);
      get_buf(PIPE_V, 1, 0);
      {
        __ubuf__ float* v99 = reinterpret_cast<__ubuf__ float*>(v8.GetPhyAddr());
        __ubuf__ float* v100 = reinterpret_cast<__ubuf__ float*>(v7.GetPhyAddr());
        __VEC_SCOPE__
        {
          AscendC::Reg::RegTensor<float> v101;
          AscendC::Reg::RegTensor<float> v102;
          AscendC::Reg::RegTensor<float> v103;
          AscendC::Reg::RegTensor<float> v104;
          AscendC::Reg::RegTensor<float> v105;
          AscendC::Reg::MaskReg v106 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
          uint32_t v107[1]{c5440_idx};
          AscendC::Reg::Duplicate(v102, c0_5_f32, v106);
          for (uint16_t v108 = 0; v108 < static_cast<uint16_t>(c85_idx); v108 += 1) {
            uint32_t v109 = v108 * c64_idx;
            // The mask is updated on every iteration to match the total count
            AscendC::Reg::MaskReg v110 = AscendC::Reg::UpdateMask<float>(*v107);
            __ubuf__ float* v111 = v99 + v109;
            AscendC::Reg::DataCopy(v101, v111);
            AscendC::Reg::Mul(v103, v101, v102, v106);
            __ubuf__ float* v112 = v100 + v109;
            AscendC::Reg::DataCopy(v104, v112);
            AscendC::Reg::Mul(v105, v103, v104, v106);
            AscendC::Reg::DataCopy(v111, v105, v110);
          }
        }
      }
      rls_buf(PIPE_V, 1, 0);
      rls_buf(PIPE_V, 3, 0);
      AscendC::GlobalTensor<float> v113 = v74[v76];
      int32_t v114 = c5440_i32 - v84;
      int32_t v115 = v114 * c4_i32;
      int32_t v116 = v115 / c32_i32;
      AscendC::DataCopyExtParams v117{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v85), static_cast<uint32_t>(v116), static_cast<uint32_t>(v93), static_cast<uint32_t>(c0_i32)};
      get_buf(PIPE_MTE3, 3, 0);
      AscendC::DataCopyPad(v113, v8, v117);
      rls_buf(PIPE_MTE3, 3, 0);
    }
  }
  return;
}

