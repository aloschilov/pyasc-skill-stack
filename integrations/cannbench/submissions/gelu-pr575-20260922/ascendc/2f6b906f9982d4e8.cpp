#include "kernel_operator.h"
extern "C"  __global__ __aicore__ void gelu_reference_kernel(__gm__ half* v1_input_ptr, __gm__ half* v2_output_ptr, int32_t v3_input_length) {
  constexpr uint32_t c72_idx = 72;
  constexpr uint32_t c4608_idx = 4608;
  constexpr uint32_t c64_idx = 64;
  constexpr int32_t c72_i32 = 72;
  constexpr int32_t c1_i32 = 1;
  constexpr int32_t c512_i32 = 512;
  constexpr int32_t c0_i32 = 0;
  constexpr int32_t c4608_i32 = 4608;
  half c0_f16 = 0.0e+00;
  constexpr int32_t c71_i32 = 71;
  constexpr int32_t c511_i32 = 511;
  constexpr int32_t c4607_i32 = 4607;
  constexpr float c0_5_f32 = (float)5.000000000e-01;
  constexpr float c0_707106769_f32 = (float)7.071067690e-01;
  constexpr float c1_f32 = (float)1.000000000e+00;
  constexpr int32_t c2_i32 = 2;
  constexpr int32_t c32_i32 = 32;
  constexpr int32_t c9216_i32 = 9216;
  constexpr int64_t c4608_i64 = 4608;
  constexpr bool c0_i1 = false;
  AscendC::TPipe v4;
  AscendC::LocalTensor<half> v5{AscendC::TPosition::VECCALC, 0, 4608};
  AscendC::LocalTensor<half> v6{AscendC::TPosition::VECCALC, 9216, 4608};
  AscendC::LocalTensor<float> v7{AscendC::TPosition::VECCALC, 18432, 4608};
  AscendC::LocalTensor<float> v8{AscendC::TPosition::VECCALC, 36864, 4608};
  AscendC::LocalTensor<float> v9{AscendC::TPosition::VECCALC, 55296, 4608};
  int32_t v10 = v3_input_length + c71_i32;
  int32_t v11 = v10 / c72_i32;
  int32_t v12 = v11 + c511_i32;
  int32_t v13 = v12 / c512_i32;
  int32_t v14 = v13 * c512_i32;
  int32_t v15 = AscendC::GetBlockIdx();
  int32_t v16 = v15 * v14;
  int32_t v17 = v16 + v14;
  int32_t v18 = ((v17 < v3_input_length) ? (v17) : (v3_input_length));
  int32_t v19 = v18 - v16;
  int32_t v20 = ((v19 > c0_i32) ? (v19) : (c0_i32));
  int32_t v21 = v20 + c4607_i32;
  int32_t v22 = v21 / c4608_i32;
  int32_t v23 = v22 % c2_i32;
  int32_t v24 = v22 - v23;
  for (int32_t v25 = c0_i32; v25 < v24; v25 += c2_i32) {
    AscendC::GlobalTensor<half> v26;
    v26.SetGlobalBuffer(v1_input_ptr);
    AscendC::GlobalTensor<half> v27;
    v27.SetGlobalBuffer(v2_output_ptr);
    int32_t v28 = v25 * c4608_i32;
    int32_t v29 = v16 + v28;
    int32_t v30 = v18 - v29;
    int32_t v31 = ((v30 < c4608_i32) ? (v30) : (c4608_i32));
    int32_t v32 = ((v31 > c0_i32) ? (v31) : (c0_i32));
    AscendC::GlobalTensor<half> v33 = v26[v29];
    int32_t v34 = v3_input_length - v29;
    bool v35 = v34 < c0_i32;
    int32_t v36 = v35 ? c0_i32 : v34;
    int32_t v37 = ((v32 < v36) ? (v32) : (v36));
    int32_t v38 = v37 * c2_i32;
    int32_t v39 = v3_input_length - v37;
    int32_t v40 = v38 % c32_i32;
    bool v41 = v40 == c0_i32;
    int32_t v42 = c32_i32 - v40;
    int32_t v43 = v41 ? c0_i32 : v42;
    int32_t v44 = v38 + v43;
    bool v45 = v44 < c9216_i32;
    if (v45) {
      get_buf(PIPE_V, 0, 0);
      AscendC::Duplicate(v5, c0_f16, c4608_i64);
      rls_buf(PIPE_V, 0, 0);
    }
    int32_t v46 = v39 * c2_i32;
    int32_t v47 = v43 / c2_i32;
    int32_t v48 = c9216_i32 - v44;
    int32_t v49 = v48 / c32_i32;
    AscendC::DataCopyExtParams v50{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v38), static_cast<uint32_t>(v46), static_cast<uint32_t>(v49), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<half> v51{c1_i32, c0_i32, static_cast<uint8_t>(v47), c0_f16};
    get_buf(PIPE_MTE2, 0, 0);
    AscendC::DataCopyPad(v5, v33, v50, v51);
    rls_buf(PIPE_MTE2, 0, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 0, 0);
    AscendC::Cast<float, half>(v9, v5, AscendC::RoundMode::CAST_NONE, c4608_i64);
    rls_buf(PIPE_V, 0, 0);
    rls_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 3, 0);
    get_buf(PIPE_V, 1, 0);
    {
      __ubuf__ float* v52 = reinterpret_cast<__ubuf__ float*>(v9.GetPhyAddr());
      __ubuf__ float* v53 = reinterpret_cast<__ubuf__ float*>(v8.GetPhyAddr());
      __ubuf__ float* v54 = reinterpret_cast<__ubuf__ float*>(v7.GetPhyAddr());
      __VEC_SCOPE__
      {
        AscendC::Reg::RegTensor<float> v55;
        AscendC::Reg::RegTensor<float> v56;
        AscendC::Reg::RegTensor<float> v57;
        AscendC::Reg::RegTensor<float> v58;
        AscendC::Reg::RegTensor<float> v59;
        AscendC::Reg::MaskReg v60 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t v61[1]{c4608_idx};
        AscendC::Reg::Duplicate(v56, c0_5_f32, v60);
        AscendC::Reg::Duplicate(v58, c0_707106769_f32, v60);
        for (uint16_t v62 = 0; v62 < static_cast<uint16_t>(c72_idx); v62 += 1) {
          uint32_t v63 = v62 * c64_idx;
          // The mask is updated on every iteration to match the total count
          AscendC::Reg::MaskReg v64 = AscendC::Reg::UpdateMask<float>(*v61);
          __ubuf__ float* v65 = v52 + v63;
          AscendC::Reg::DataCopy(v55, v65);
          AscendC::Reg::Mul(v57, v55, v56, v60);
          __ubuf__ float* v66 = v53 + v63;
          AscendC::Reg::DataCopy(v66, v57, v64);
          AscendC::Reg::Mul(v59, v55, v58, v60);
          __ubuf__ float* v67 = v54 + v63;
          AscendC::Reg::DataCopy(v67, v59, v64);
        }
      }
    }
    rls_buf(PIPE_V, 1, 0);
    rls_buf(PIPE_V, 3, 0);
    rls_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 3, 0);
    AscendC::Erf<float, c0_i1>(v9, v7, c4608_i64);
    rls_buf(PIPE_V, 3, 0);
    rls_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 2, 0);
    {
      __ubuf__ float* v68 = reinterpret_cast<__ubuf__ float*>(v9.GetPhyAddr());
      __ubuf__ float* v69 = reinterpret_cast<__ubuf__ float*>(v8.GetPhyAddr());
      __VEC_SCOPE__
      {
        AscendC::Reg::RegTensor<float> v70;
        AscendC::Reg::RegTensor<float> v71;
        AscendC::Reg::RegTensor<float> v72;
        AscendC::Reg::RegTensor<float> v73;
        AscendC::Reg::RegTensor<float> v74;
        AscendC::Reg::MaskReg v75 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t v76[1]{c4608_idx};
        AscendC::Reg::Duplicate(v71, c1_f32, v75);
        for (uint16_t v77 = 0; v77 < static_cast<uint16_t>(c72_idx); v77 += 1) {
          uint32_t v78 = v77 * c64_idx;
          // The mask is updated on every iteration to match the total count
          AscendC::Reg::MaskReg v79 = AscendC::Reg::UpdateMask<float>(*v76);
          __ubuf__ float* v80 = v68 + v78;
          AscendC::Reg::DataCopy(v70, v80);
          AscendC::Reg::Add(v72, v70, v71, v75);
          __ubuf__ float* v81 = v69 + v78;
          AscendC::Reg::DataCopy(v73, v81);
          AscendC::Reg::Mul(v74, v73, v72, v75);
          AscendC::Reg::DataCopy(v80, v74, v79);
        }
      }
    }
    rls_buf(PIPE_V, 2, 0);
    rls_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 0, 0);
    get_buf(PIPE_V, 1, 0);
    AscendC::Cast<half, float>(v5, v9, AscendC::RoundMode::CAST_RINT, c4608_i64);
    rls_buf(PIPE_V, 1, 0);
    rls_buf(PIPE_V, 0, 0);
    AscendC::GlobalTensor<half> v82 = v27[v29];
    int32_t v83 = c4608_i32 - v37;
    int32_t v84 = v83 * c2_i32;
    int32_t v85 = v84 / c32_i32;
    AscendC::DataCopyExtParams v86{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v38), static_cast<uint32_t>(v85), static_cast<uint32_t>(v46), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 0, 0);
    AscendC::DataCopyPad(v82, v5, v86);
    rls_buf(PIPE_MTE3, 0, 0);
    int32_t v87 = v25 + c1_i32;
    AscendC::GlobalTensor<half> v88;
    v88.SetGlobalBuffer(v1_input_ptr);
    AscendC::GlobalTensor<half> v89;
    v89.SetGlobalBuffer(v2_output_ptr);
    int32_t v90 = v87 * c4608_i32;
    int32_t v91 = v16 + v90;
    int32_t v92 = v18 - v91;
    int32_t v93 = ((v92 < c4608_i32) ? (v92) : (c4608_i32));
    int32_t v94 = ((v93 > c0_i32) ? (v93) : (c0_i32));
    AscendC::GlobalTensor<half> v95 = v88[v91];
    int32_t v96 = v3_input_length - v91;
    bool v97 = v96 < c0_i32;
    int32_t v98 = v97 ? c0_i32 : v96;
    int32_t v99 = ((v94 < v98) ? (v94) : (v98));
    int32_t v100 = v99 * c2_i32;
    int32_t v101 = v3_input_length - v99;
    int32_t v102 = v100 % c32_i32;
    bool v103 = v102 == c0_i32;
    int32_t v104 = c32_i32 - v102;
    int32_t v105 = v103 ? c0_i32 : v104;
    int32_t v106 = v100 + v105;
    bool v107 = v106 < c9216_i32;
    if (v107) {
      get_buf(PIPE_V, 4, 0);
      AscendC::Duplicate(v6, c0_f16, c4608_i64);
      rls_buf(PIPE_V, 4, 0);
    }
    int32_t v108 = v101 * c2_i32;
    int32_t v109 = v105 / c2_i32;
    int32_t v110 = c9216_i32 - v106;
    int32_t v111 = v110 / c32_i32;
    AscendC::DataCopyExtParams v112{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v100), static_cast<uint32_t>(v108), static_cast<uint32_t>(v111), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<half> v113{c1_i32, c0_i32, static_cast<uint8_t>(v109), c0_f16};
    get_buf(PIPE_MTE2, 4, 0);
    AscendC::DataCopyPad(v6, v95, v112, v113);
    rls_buf(PIPE_MTE2, 4, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 4, 0);
    AscendC::Cast<float, half>(v9, v6, AscendC::RoundMode::CAST_NONE, c4608_i64);
    rls_buf(PIPE_V, 4, 0);
    rls_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 3, 0);
    get_buf(PIPE_V, 1, 0);
    {
      __ubuf__ float* v114 = reinterpret_cast<__ubuf__ float*>(v9.GetPhyAddr());
      __ubuf__ float* v115 = reinterpret_cast<__ubuf__ float*>(v8.GetPhyAddr());
      __ubuf__ float* v116 = reinterpret_cast<__ubuf__ float*>(v7.GetPhyAddr());
      __VEC_SCOPE__
      {
        AscendC::Reg::RegTensor<float> v117;
        AscendC::Reg::RegTensor<float> v118;
        AscendC::Reg::RegTensor<float> v119;
        AscendC::Reg::RegTensor<float> v120;
        AscendC::Reg::RegTensor<float> v121;
        AscendC::Reg::MaskReg v122 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t v123[1]{c4608_idx};
        AscendC::Reg::Duplicate(v118, c0_5_f32, v122);
        AscendC::Reg::Duplicate(v120, c0_707106769_f32, v122);
        for (uint16_t v124 = 0; v124 < static_cast<uint16_t>(c72_idx); v124 += 1) {
          uint32_t v125 = v124 * c64_idx;
          // The mask is updated on every iteration to match the total count
          AscendC::Reg::MaskReg v126 = AscendC::Reg::UpdateMask<float>(*v123);
          __ubuf__ float* v127 = v114 + v125;
          AscendC::Reg::DataCopy(v117, v127);
          AscendC::Reg::Mul(v119, v117, v118, v122);
          __ubuf__ float* v128 = v115 + v125;
          AscendC::Reg::DataCopy(v128, v119, v126);
          AscendC::Reg::Mul(v121, v117, v120, v122);
          __ubuf__ float* v129 = v116 + v125;
          AscendC::Reg::DataCopy(v129, v121, v126);
        }
      }
    }
    rls_buf(PIPE_V, 1, 0);
    rls_buf(PIPE_V, 3, 0);
    rls_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 3, 0);
    AscendC::Erf<float, c0_i1>(v9, v7, c4608_i64);
    rls_buf(PIPE_V, 3, 0);
    rls_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 2, 0);
    {
      __ubuf__ float* v130 = reinterpret_cast<__ubuf__ float*>(v9.GetPhyAddr());
      __ubuf__ float* v131 = reinterpret_cast<__ubuf__ float*>(v8.GetPhyAddr());
      __VEC_SCOPE__
      {
        AscendC::Reg::RegTensor<float> v132;
        AscendC::Reg::RegTensor<float> v133;
        AscendC::Reg::RegTensor<float> v134;
        AscendC::Reg::RegTensor<float> v135;
        AscendC::Reg::RegTensor<float> v136;
        AscendC::Reg::MaskReg v137 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t v138[1]{c4608_idx};
        AscendC::Reg::Duplicate(v133, c1_f32, v137);
        for (uint16_t v139 = 0; v139 < static_cast<uint16_t>(c72_idx); v139 += 1) {
          uint32_t v140 = v139 * c64_idx;
          // The mask is updated on every iteration to match the total count
          AscendC::Reg::MaskReg v141 = AscendC::Reg::UpdateMask<float>(*v138);
          __ubuf__ float* v142 = v130 + v140;
          AscendC::Reg::DataCopy(v132, v142);
          AscendC::Reg::Add(v134, v132, v133, v137);
          __ubuf__ float* v143 = v131 + v140;
          AscendC::Reg::DataCopy(v135, v143);
          AscendC::Reg::Mul(v136, v135, v134, v137);
          AscendC::Reg::DataCopy(v142, v136, v141);
        }
      }
    }
    rls_buf(PIPE_V, 2, 0);
    rls_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 4, 0);
    get_buf(PIPE_V, 1, 0);
    AscendC::Cast<half, float>(v6, v9, AscendC::RoundMode::CAST_RINT, c4608_i64);
    rls_buf(PIPE_V, 1, 0);
    rls_buf(PIPE_V, 4, 0);
    AscendC::GlobalTensor<half> v144 = v89[v91];
    int32_t v145 = c4608_i32 - v99;
    int32_t v146 = v145 * c2_i32;
    int32_t v147 = v146 / c32_i32;
    AscendC::DataCopyExtParams v148{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v100), static_cast<uint32_t>(v147), static_cast<uint32_t>(v108), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 4, 0);
    AscendC::DataCopyPad(v144, v6, v148);
    rls_buf(PIPE_MTE3, 4, 0);
  }
  AscendC::LocalTensor<half> v149 = v8.ReinterpretCast<half>();
  for (int32_t v150 = v24; v150 < v22; v150 += c1_i32) {
    AscendC::GlobalTensor<half> v151;
    v151.SetGlobalBuffer(v1_input_ptr);
    AscendC::GlobalTensor<half> v152;
    v152.SetGlobalBuffer(v2_output_ptr);
    int32_t v153 = v150 * c4608_i32;
    int32_t v154 = v16 + v153;
    int32_t v155 = v18 - v154;
    int32_t v156 = ((v155 < c4608_i32) ? (v155) : (c4608_i32));
    int32_t v157 = ((v156 > c0_i32) ? (v156) : (c0_i32));
    AscendC::GlobalTensor<half> v158 = v151[v154];
    int32_t v159 = v3_input_length - v154;
    bool v160 = v159 < c0_i32;
    int32_t v161 = v160 ? c0_i32 : v159;
    int32_t v162 = ((v157 < v161) ? (v157) : (v161));
    int32_t v163 = v162 * c2_i32;
    int32_t v164 = v3_input_length - v162;
    int32_t v165 = v163 % c32_i32;
    bool v166 = v165 == c0_i32;
    int32_t v167 = c32_i32 - v165;
    int32_t v168 = v166 ? c0_i32 : v167;
    int32_t v169 = v163 + v168;
    bool v170 = v169 < c9216_i32;
    if (v170) {
      get_buf(PIPE_V, 2, 0);
      AscendC::Duplicate(v149, c0_f16, c4608_i64);
      rls_buf(PIPE_V, 2, 0);
    }
    int32_t v171 = v164 * c2_i32;
    int32_t v172 = v168 / c2_i32;
    int32_t v173 = c9216_i32 - v169;
    int32_t v174 = v173 / c32_i32;
    AscendC::DataCopyExtParams v175{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v163), static_cast<uint32_t>(v171), static_cast<uint32_t>(v174), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<half> v176{c1_i32, c0_i32, static_cast<uint8_t>(v172), c0_f16};
    get_buf(PIPE_MTE2, 2, 0);
    AscendC::DataCopyPad(v149, v158, v175, v176);
    rls_buf(PIPE_MTE2, 2, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 2, 0);
    AscendC::Cast<float, half>(v9, v149, AscendC::RoundMode::CAST_NONE, c4608_i64);
    rls_buf(PIPE_V, 2, 0);
    rls_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 3, 0);
    get_buf(PIPE_V, 1, 0);
    {
      __ubuf__ float* v177 = reinterpret_cast<__ubuf__ float*>(v9.GetPhyAddr());
      __ubuf__ float* v178 = reinterpret_cast<__ubuf__ float*>(v8.GetPhyAddr());
      __ubuf__ float* v179 = reinterpret_cast<__ubuf__ float*>(v7.GetPhyAddr());
      __VEC_SCOPE__
      {
        AscendC::Reg::RegTensor<float> v180;
        AscendC::Reg::RegTensor<float> v181;
        AscendC::Reg::RegTensor<float> v182;
        AscendC::Reg::RegTensor<float> v183;
        AscendC::Reg::RegTensor<float> v184;
        AscendC::Reg::MaskReg v185 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t v186[1]{c4608_idx};
        AscendC::Reg::Duplicate(v181, c0_5_f32, v185);
        AscendC::Reg::Duplicate(v183, c0_707106769_f32, v185);
        for (uint16_t v187 = 0; v187 < static_cast<uint16_t>(c72_idx); v187 += 1) {
          uint32_t v188 = v187 * c64_idx;
          // The mask is updated on every iteration to match the total count
          AscendC::Reg::MaskReg v189 = AscendC::Reg::UpdateMask<float>(*v186);
          __ubuf__ float* v190 = v177 + v188;
          AscendC::Reg::DataCopy(v180, v190);
          AscendC::Reg::Mul(v182, v180, v181, v185);
          __ubuf__ float* v191 = v178 + v188;
          AscendC::Reg::DataCopy(v191, v182, v189);
          AscendC::Reg::Mul(v184, v180, v183, v185);
          __ubuf__ float* v192 = v179 + v188;
          AscendC::Reg::DataCopy(v192, v184, v189);
        }
      }
    }
    rls_buf(PIPE_V, 1, 0);
    rls_buf(PIPE_V, 3, 0);
    rls_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 3, 0);
    AscendC::Erf<float, c0_i1>(v9, v7, c4608_i64);
    rls_buf(PIPE_V, 3, 0);
    rls_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 2, 0);
    {
      __ubuf__ float* v193 = reinterpret_cast<__ubuf__ float*>(v9.GetPhyAddr());
      __ubuf__ float* v194 = reinterpret_cast<__ubuf__ float*>(v8.GetPhyAddr());
      __VEC_SCOPE__
      {
        AscendC::Reg::RegTensor<float> v195;
        AscendC::Reg::RegTensor<float> v196;
        AscendC::Reg::RegTensor<float> v197;
        AscendC::Reg::RegTensor<float> v198;
        AscendC::Reg::RegTensor<float> v199;
        AscendC::Reg::MaskReg v200 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t v201[1]{c4608_idx};
        AscendC::Reg::Duplicate(v196, c1_f32, v200);
        for (uint16_t v202 = 0; v202 < static_cast<uint16_t>(c72_idx); v202 += 1) {
          uint32_t v203 = v202 * c64_idx;
          // The mask is updated on every iteration to match the total count
          AscendC::Reg::MaskReg v204 = AscendC::Reg::UpdateMask<float>(*v201);
          __ubuf__ float* v205 = v193 + v203;
          AscendC::Reg::DataCopy(v195, v205);
          AscendC::Reg::Add(v197, v195, v196, v200);
          __ubuf__ float* v206 = v194 + v203;
          AscendC::Reg::DataCopy(v198, v206);
          AscendC::Reg::Mul(v199, v198, v197, v200);
          AscendC::Reg::DataCopy(v205, v199, v204);
        }
      }
    }
    rls_buf(PIPE_V, 2, 0);
    rls_buf(PIPE_V, 1, 0);
    get_buf(PIPE_V, 2, 0);
    get_buf(PIPE_V, 1, 0);
    AscendC::Cast<half, float>(v149, v9, AscendC::RoundMode::CAST_RINT, c4608_i64);
    rls_buf(PIPE_V, 1, 0);
    rls_buf(PIPE_V, 2, 0);
    AscendC::GlobalTensor<half> v207 = v152[v154];
    int32_t v208 = c4608_i32 - v162;
    int32_t v209 = v208 * c2_i32;
    int32_t v210 = v209 / c32_i32;
    AscendC::DataCopyExtParams v211{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v163), static_cast<uint32_t>(v210), static_cast<uint32_t>(v171), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 2, 0);
    AscendC::DataCopyPad(v207, v149, v211);
    rls_buf(PIPE_MTE3, 2, 0);
  }
  return;
}

