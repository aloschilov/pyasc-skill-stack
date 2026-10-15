#include "kernel_operator.h"
extern "C"  __global__ __aicore__ void gelu_reference_kernel(__gm__ float* v1_input_ptr, __gm__ float* v2_output_ptr, int32_t v3_input_length) {
  constexpr uint32_t c128_idx = 128;
  constexpr uint32_t c8192_idx = 8192;
  constexpr uint32_t c64_idx = 64;
  constexpr int32_t c72_i32 = 72;
  constexpr int32_t c1_i32 = 1;
  constexpr int32_t c512_i32 = 512;
  constexpr int32_t c0_i32 = 0;
  constexpr int32_t c8192_i32 = 8192;
  constexpr float c0_f32 = (float)0.0e+00;
  constexpr int32_t c71_i32 = 71;
  constexpr int32_t c511_i32 = 511;
  constexpr int32_t c8191_i32 = 8191;
  constexpr float c0_0447149985_f32 = (float)4.471499850e-02;
  constexpr float cm1_59576917_f32 = (float)-1.595769170e+00;
  constexpr float c1_f32 = (float)1.000000000e+00;
  constexpr int32_t c4_i32 = 4;
  constexpr int32_t c32_i32 = 32;
  constexpr int32_t c32768_i32 = 32768;
  constexpr int64_t c8192_i64 = 8192;
  constexpr int32_t c2_i32 = 2;
  AscendC::TPipe v4;
  AscendC::LocalTensor<float> v5{AscendC::TPosition::VECCALC, 0, 8192};
  AscendC::LocalTensor<float> v6{AscendC::TPosition::VECCALC, 32768, 8192};
  AscendC::LocalTensor<float> v7{AscendC::TPosition::VECCALC, 65536, 8192};
  AscendC::LocalTensor<float> v8{AscendC::TPosition::VECCALC, 98304, 8192};
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
  int32_t v20 = v19 + c8191_i32;
  int32_t v21 = v20 / c8192_i32;
  int32_t v22 = v21 % c2_i32;
  int32_t v23 = v21 - v22;
  for (int32_t v24 = c0_i32; v24 < v23; v24 += c2_i32) {
    AscendC::GlobalTensor<float> v25;
    v25.SetGlobalBuffer(v1_input_ptr);
    AscendC::GlobalTensor<float> v26;
    v26.SetGlobalBuffer(v2_output_ptr);
    int32_t v27 = v24 * c8192_i32;
    int32_t v28 = v15 + v27;
    int32_t v29 = v17 - v28;
    int32_t v30 = ((v29 < c8192_i32) ? (v29) : (c8192_i32));
    int32_t v31 = ((v30 > c0_i32) ? (v30) : (c0_i32));
    AscendC::GlobalTensor<float> v32 = v25[v28];
    int32_t v33 = v3_input_length - v28;
    bool v34 = v33 < c0_i32;
    int32_t v35 = v34 ? c0_i32 : v33;
    int32_t v36 = ((v31 < v35) ? (v31) : (v35));
    int32_t v37 = v36 * c4_i32;
    int32_t v38 = v3_input_length - v36;
    int32_t v39 = v37 % c32_i32;
    bool v40 = v39 == c0_i32;
    int32_t v41 = c32_i32 - v39;
    int32_t v42 = v40 ? c0_i32 : v41;
    int32_t v43 = v37 + v42;
    bool v44 = v43 < c32768_i32;
    if (v44) {
      get_buf(PIPE_V, 0, 0);
      AscendC::Duplicate(v5, c0_f32, c8192_i64);
      rls_buf(PIPE_V, 0, 0);
    }
    int32_t v45 = v38 * c4_i32;
    int32_t v46 = v42 / c4_i32;
    int32_t v47 = c32768_i32 - v43;
    int32_t v48 = v47 / c32_i32;
    AscendC::DataCopyExtParams v49{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v37), static_cast<uint32_t>(v45), static_cast<uint32_t>(v48), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<float> v50{c1_i32, c0_i32, static_cast<uint8_t>(v46), c0_f32};
    get_buf(PIPE_MTE2, 0, 0);
    AscendC::DataCopyPad(v5, v32, v49, v50);
    rls_buf(PIPE_MTE2, 0, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 0, 0);
    {
      __ubuf__ float* v51 = reinterpret_cast<__ubuf__ float*>(v5.GetPhyAddr());
      __ubuf__ float* v52 = reinterpret_cast<__ubuf__ float*>(v6.GetPhyAddr());
      __VEC_SCOPE__
      {
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
        AscendC::Reg::RegTensor<float> v64;
        AscendC::Reg::MaskReg v65 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t v66[1]{c8192_idx};
        AscendC::Reg::Duplicate(v56, c0_0447149985_f32, v65);
        AscendC::Reg::Duplicate(v59, cm1_59576917_f32, v65);
        AscendC::Reg::Duplicate(v62, c1_f32, v65);
        for (uint16_t v67 = 0; v67 < static_cast<uint16_t>(c128_idx); v67 += 1) {
          uint32_t v68 = v67 * c64_idx;
          // The mask is updated on every iteration to match the total count
          AscendC::Reg::MaskReg v69 = AscendC::Reg::UpdateMask<float>(*v66);
          __ubuf__ float* v70 = v51 + v68;
          AscendC::Reg::DataCopy(v53, v70);
          AscendC::Reg::Mul(v54, v53, v53, v65);
          AscendC::Reg::Mul(v55, v54, v53, v65);
          AscendC::Reg::Mul(v57, v55, v56, v65);
          AscendC::Reg::Add(v58, v53, v57, v65);
          AscendC::Reg::Mul(v60, v58, v59, v65);
          AscendC::Reg::Exp(v61, v60, v65);
          AscendC::Reg::Add(v63, v61, v62, v65);
          AscendC::Reg::Div(v64, v53, v63, v65);
          __ubuf__ float* v71 = v52 + v68;
          AscendC::Reg::DataCopy(v71, v64, v69);
        }
      }
    }
    rls_buf(PIPE_V, 0, 0);
    rls_buf(PIPE_V, 1, 0);
    AscendC::GlobalTensor<float> v72 = v26[v28];
    int32_t v73 = c8192_i32 - v36;
    int32_t v74 = v73 * c4_i32;
    int32_t v75 = v74 / c32_i32;
    AscendC::DataCopyExtParams v76{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v37), static_cast<uint32_t>(v75), static_cast<uint32_t>(v45), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 1, 0);
    AscendC::DataCopyPad(v72, v6, v76);
    rls_buf(PIPE_MTE3, 1, 0);
    int32_t v77 = v24 + c1_i32;
    AscendC::GlobalTensor<float> v78;
    v78.SetGlobalBuffer(v1_input_ptr);
    AscendC::GlobalTensor<float> v79;
    v79.SetGlobalBuffer(v2_output_ptr);
    int32_t v80 = v77 * c8192_i32;
    int32_t v81 = v15 + v80;
    int32_t v82 = v17 - v81;
    int32_t v83 = ((v82 < c8192_i32) ? (v82) : (c8192_i32));
    int32_t v84 = ((v83 > c0_i32) ? (v83) : (c0_i32));
    AscendC::GlobalTensor<float> v85 = v78[v81];
    int32_t v86 = v3_input_length - v81;
    bool v87 = v86 < c0_i32;
    int32_t v88 = v87 ? c0_i32 : v86;
    int32_t v89 = ((v84 < v88) ? (v84) : (v88));
    int32_t v90 = v89 * c4_i32;
    int32_t v91 = v3_input_length - v89;
    int32_t v92 = v90 % c32_i32;
    bool v93 = v92 == c0_i32;
    int32_t v94 = c32_i32 - v92;
    int32_t v95 = v93 ? c0_i32 : v94;
    int32_t v96 = v90 + v95;
    bool v97 = v96 < c32768_i32;
    if (v97) {
      get_buf(PIPE_V, 2, 0);
      AscendC::Duplicate(v7, c0_f32, c8192_i64);
      rls_buf(PIPE_V, 2, 0);
    }
    int32_t v98 = v91 * c4_i32;
    int32_t v99 = v95 / c4_i32;
    int32_t v100 = c32768_i32 - v96;
    int32_t v101 = v100 / c32_i32;
    AscendC::DataCopyExtParams v102{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v90), static_cast<uint32_t>(v98), static_cast<uint32_t>(v101), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<float> v103{c1_i32, c0_i32, static_cast<uint8_t>(v99), c0_f32};
    get_buf(PIPE_MTE2, 2, 0);
    AscendC::DataCopyPad(v7, v85, v102, v103);
    rls_buf(PIPE_MTE2, 2, 0);
    get_buf(PIPE_V, 3, 0);
    get_buf(PIPE_V, 2, 0);
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
        uint32_t v119[1]{c8192_idx};
        AscendC::Reg::Duplicate(v109, c0_0447149985_f32, v118);
        AscendC::Reg::Duplicate(v112, cm1_59576917_f32, v118);
        AscendC::Reg::Duplicate(v115, c1_f32, v118);
        for (uint16_t v120 = 0; v120 < static_cast<uint16_t>(c128_idx); v120 += 1) {
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
    rls_buf(PIPE_V, 2, 0);
    rls_buf(PIPE_V, 3, 0);
    AscendC::GlobalTensor<float> v125 = v79[v81];
    int32_t v126 = c8192_i32 - v89;
    int32_t v127 = v126 * c4_i32;
    int32_t v128 = v127 / c32_i32;
    AscendC::DataCopyExtParams v129{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v90), static_cast<uint32_t>(v128), static_cast<uint32_t>(v98), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 3, 0);
    AscendC::DataCopyPad(v125, v8, v129);
    rls_buf(PIPE_MTE3, 3, 0);
  }
  for (int32_t v130 = v23; v130 < v21; v130 += c1_i32) {
    AscendC::GlobalTensor<float> v131;
    v131.SetGlobalBuffer(v1_input_ptr);
    AscendC::GlobalTensor<float> v132;
    v132.SetGlobalBuffer(v2_output_ptr);
    int32_t v133 = v130 * c8192_i32;
    int32_t v134 = v15 + v133;
    int32_t v135 = v17 - v134;
    int32_t v136 = ((v135 < c8192_i32) ? (v135) : (c8192_i32));
    int32_t v137 = ((v136 > c0_i32) ? (v136) : (c0_i32));
    AscendC::GlobalTensor<float> v138 = v131[v134];
    int32_t v139 = v3_input_length - v134;
    bool v140 = v139 < c0_i32;
    int32_t v141 = v140 ? c0_i32 : v139;
    int32_t v142 = ((v137 < v141) ? (v137) : (v141));
    int32_t v143 = v142 * c4_i32;
    int32_t v144 = v3_input_length - v142;
    int32_t v145 = v143 % c32_i32;
    bool v146 = v145 == c0_i32;
    int32_t v147 = c32_i32 - v145;
    int32_t v148 = v146 ? c0_i32 : v147;
    int32_t v149 = v143 + v148;
    bool v150 = v149 < c32768_i32;
    if (v150) {
      get_buf(PIPE_V, 2, 0);
      AscendC::Duplicate(v7, c0_f32, c8192_i64);
      rls_buf(PIPE_V, 2, 0);
    }
    int32_t v151 = v144 * c4_i32;
    int32_t v152 = v148 / c4_i32;
    int32_t v153 = c32768_i32 - v149;
    int32_t v154 = v153 / c32_i32;
    AscendC::DataCopyExtParams v155{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v143), static_cast<uint32_t>(v151), static_cast<uint32_t>(v154), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<float> v156{c1_i32, c0_i32, static_cast<uint8_t>(v152), c0_f32};
    get_buf(PIPE_MTE2, 2, 0);
    AscendC::DataCopyPad(v7, v138, v155, v156);
    rls_buf(PIPE_MTE2, 2, 0);
    get_buf(PIPE_V, 3, 0);
    get_buf(PIPE_V, 2, 0);
    {
      __ubuf__ float* v157 = reinterpret_cast<__ubuf__ float*>(v7.GetPhyAddr());
      __ubuf__ float* v158 = reinterpret_cast<__ubuf__ float*>(v8.GetPhyAddr());
      __VEC_SCOPE__
      {
        AscendC::Reg::RegTensor<float> v159;
        AscendC::Reg::RegTensor<float> v160;
        AscendC::Reg::RegTensor<float> v161;
        AscendC::Reg::RegTensor<float> v162;
        AscendC::Reg::RegTensor<float> v163;
        AscendC::Reg::RegTensor<float> v164;
        AscendC::Reg::RegTensor<float> v165;
        AscendC::Reg::RegTensor<float> v166;
        AscendC::Reg::RegTensor<float> v167;
        AscendC::Reg::RegTensor<float> v168;
        AscendC::Reg::RegTensor<float> v169;
        AscendC::Reg::RegTensor<float> v170;
        AscendC::Reg::MaskReg v171 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t v172[1]{c8192_idx};
        AscendC::Reg::Duplicate(v162, c0_0447149985_f32, v171);
        AscendC::Reg::Duplicate(v165, cm1_59576917_f32, v171);
        AscendC::Reg::Duplicate(v168, c1_f32, v171);
        for (uint16_t v173 = 0; v173 < static_cast<uint16_t>(c128_idx); v173 += 1) {
          uint32_t v174 = v173 * c64_idx;
          // The mask is updated on every iteration to match the total count
          AscendC::Reg::MaskReg v175 = AscendC::Reg::UpdateMask<float>(*v172);
          __ubuf__ float* v176 = v157 + v174;
          AscendC::Reg::DataCopy(v159, v176);
          AscendC::Reg::Mul(v160, v159, v159, v171);
          AscendC::Reg::Mul(v161, v160, v159, v171);
          AscendC::Reg::Mul(v163, v161, v162, v171);
          AscendC::Reg::Add(v164, v159, v163, v171);
          AscendC::Reg::Mul(v166, v164, v165, v171);
          AscendC::Reg::Exp(v167, v166, v171);
          AscendC::Reg::Add(v169, v167, v168, v171);
          AscendC::Reg::Div(v170, v159, v169, v171);
          __ubuf__ float* v177 = v158 + v174;
          AscendC::Reg::DataCopy(v177, v170, v175);
        }
      }
    }
    rls_buf(PIPE_V, 2, 0);
    rls_buf(PIPE_V, 3, 0);
    AscendC::GlobalTensor<float> v178 = v132[v134];
    int32_t v179 = c8192_i32 - v142;
    int32_t v180 = v179 * c4_i32;
    int32_t v181 = v180 / c32_i32;
    AscendC::DataCopyExtParams v182{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v143), static_cast<uint32_t>(v181), static_cast<uint32_t>(v151), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 3, 0);
    AscendC::DataCopyPad(v178, v8, v182);
    rls_buf(PIPE_MTE3, 3, 0);
  }
  return;
}

