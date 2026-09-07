#include "kernel_operator.h"
extern "C"  __global__ __aicore__ void _gelu_register(__gm__ bfloat16_t* v1_x_ptr, __gm__ bfloat16_t* v2_y_ptr, int32_t v3_size) {
  constexpr int64_t c8192_i64 = 8192;
  constexpr int32_t c16384_i32 = 16384;
  constexpr int32_t c32_i32 = 32;
  constexpr int32_t c2_i32 = 2;
  constexpr int32_t c1_i32 = 1;
  constexpr int32_t c0_i32 = 0;
  constexpr int32_t c8191_i32 = 8191;
  bfloat16_t c0_bf16 = 0.0e+00;
  constexpr int32_t c8192_i32 = 8192;
  AscendC::TPipe v4;
  int32_t v5 = AscendC::GetBlockIdx();
  int32_t v6 = v3_size + c8191_i32;
  int32_t v7 = v6 / c8192_i32;
  int32_t v8 = AscendC::GetBlockNum();
  int32_t v9 = v7 - v5;
  int32_t v10 = v8 - c1_i32;
  int32_t v11 = v9 + v10;
  int32_t v12 = v11 / v8;
  int32_t v13 = v12 % c2_i32;
  int32_t v14 = v12 - v13;
  int32_t v15 = v14 * v8;
  int32_t v16 = v5 + v15;
  int32_t v17 = v8 * c2_i32;
  AscendC::LocalTensor<bfloat16_t> v18{AscendC::TPosition::VECCALC, 0, 8192};
  AscendC::LocalTensor<bfloat16_t> v19{AscendC::TPosition::VECCALC, 16384, 8192};
  AscendC::LocalTensor<bfloat16_t> v20{AscendC::TPosition::VECCALC, 32768, 8192};
  AscendC::LocalTensor<bfloat16_t> v21{AscendC::TPosition::VECCALC, 49152, 8192};
  for (int32_t v22 = v5; v22 < v16; v22 += v17) {
    AscendC::GlobalTensor<bfloat16_t> v23;
    v23.SetGlobalBuffer(v1_x_ptr);
    AscendC::GlobalTensor<bfloat16_t> v24;
    v24.SetGlobalBuffer(v2_y_ptr);
    int32_t v25 = v22 * c8192_i32;
    int32_t v26 = v25 + c8192_i32;
    bool v27 = v26 <= v3_size;
    int32_t v28;
    if (v27) {
      v28 = c8192_i32;
    } else {
      int32_t v29 = v3_size - v25;
      v28 = v29;
    }
    AscendC::GlobalTensor<bfloat16_t> v30 = v23[v25];
    int32_t v31 = v3_size - v25;
    bool v32 = v31 < c0_i32;
    int32_t v33 = v32 ? c0_i32 : v31;
    int32_t v34 = ((v28 < v33) ? (v28) : (v33));
    int32_t v35 = v34 * c2_i32;
    int32_t v36 = v3_size - v34;
    int32_t v37 = v35 % c32_i32;
    bool v38 = v37 == c0_i32;
    int32_t v39 = c32_i32 - v37;
    int32_t v40 = v38 ? c0_i32 : v39;
    int32_t v41 = v35 + v40;
    bool v42 = v41 < c16384_i32;
    if (v42) {
      get_buf(PIPE_V, 0, 0);
      AscendC::Duplicate(v21, c0_bf16, c8192_i64);
      rls_buf(PIPE_V, 0, 0);
    }
    int32_t v43 = v36 * c2_i32;
    int32_t v44 = v40 / c2_i32;
    int32_t v45 = c16384_i32 - v41;
    int32_t v46 = v45 / c32_i32;
    AscendC::DataCopyExtParams v47{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v35), static_cast<uint32_t>(v43), static_cast<uint32_t>(v46), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<bfloat16_t> v48{c1_i32, c0_i32, static_cast<uint8_t>(v44), c0_bf16};
    get_buf(PIPE_MTE2, 0, 0);
    AscendC::DataCopyPad(v21, v30, v47, v48);
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

        auto* in = reinterpret_cast<__ubuf__ bfloat16_t*>(v21.GetPhyAddr());
        auto* out = reinterpret_cast<__ubuf__ bfloat16_t*>(v20.GetPhyAddr());
        AscendC::Reg::RegTensor<float> x, a, z, u, p, tmp, e, y, one;
        uint32_t count = v21.GetSize();
        uint16_t loops = (count + 63) / 64;
        AscendC::Reg::MaskReg all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Duplicate(one, 1.0f, all);
        AscendC::Reg::RegTensor<float> c0,c1,c2,c3,c4,c5,c6,c7,c8;
        AscendC::Reg::Duplicate(c0, -0.82215223f, all);
        AscendC::Reg::Duplicate(c1, 1.48851587f, all);
        AscendC::Reg::Duplicate(c2, -1.13520398f, all);
        AscendC::Reg::Duplicate(c3, 0.27886807f, all);
        AscendC::Reg::Duplicate(c4, -0.18628806f, all);
        AscendC::Reg::Duplicate(c5, 0.09678418f, all);
        AscendC::Reg::Duplicate(c6, 0.37409196f, all);
        AscendC::Reg::Duplicate(c7, 1.00002368f, all);
        AscendC::Reg::Duplicate(c8, -1.26551223f, all);
        for (uint16_t i=0; i<loops; ++i) {
            auto mask = AscendC::Reg::UpdateMask<float>(count);

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(low, in+i*64);
        AscendC::Reg::Cast<float, bfloat16_t, up>(x, low, mask);

            AscendC::Reg::Abs(a, x, mask);
            AscendC::Reg::Muls(z, a, 0.7071067811865476f, mask);
            AscendC::Reg::Muls(u, z, 0.5f, mask);
            AscendC::Reg::Adds(u, u, 1.0f, mask);
            AscendC::Reg::Div(u, one, u, mask);
            AscendC::Reg::Duplicate(p, 0.17087277f, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c0, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c1, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c2, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c3, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c4, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c5, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c6, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c7, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c8, mask);
            AscendC::Reg::Mul(tmp, z, z, mask);
            AscendC::Reg::Sub(p, p, tmp, mask);
            AscendC::Reg::Exp(e, p, mask);
            AscendC::Reg::Mul(e, e, u, mask);
            AscendC::Reg::Muls(a, a, 0.5f, mask);
            AscendC::Reg::Mul(e, e, a, mask);
            AscendC::Reg::Muls(y, x, 0.5f, mask);
            AscendC::Reg::Add(y, y, a, mask);
            AscendC::Reg::Sub(y, y, e, mask);
            // Positive infinity has GeLU(+inf)=+inf; avoid inf*erfc(inf)=NaN.
            AscendC::Reg::MaskReg positiveInf;
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(
                positiveInf, (AscendC::Reg::RegTensor<uint32_t>&)x, 0x7f800000u, mask);
            AscendC::Reg::Select(y, x, y, positiveInf);

        AscendC::Reg::Cast<bfloat16_t, float, down>(low, y, mask);
        AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out+i*64, low, mask);

        }
        ;
      }
    }
    rls_buf(PIPE_V, 0, 0);
    rls_buf(PIPE_V, 1, 0);
    AscendC::GlobalTensor<bfloat16_t> v49 = v24[v25];
    int32_t v50 = c8192_i32 - v34;
    int32_t v51 = v50 * c2_i32;
    int32_t v52 = v51 / c32_i32;
    AscendC::DataCopyExtParams v53{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v35), static_cast<uint32_t>(v52), static_cast<uint32_t>(v43), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 1, 0);
    AscendC::DataCopyPad(v49, v20, v53);
    rls_buf(PIPE_MTE3, 1, 0);
    int32_t v54 = v22 + v8;
    AscendC::GlobalTensor<bfloat16_t> v55;
    v55.SetGlobalBuffer(v1_x_ptr);
    AscendC::GlobalTensor<bfloat16_t> v56;
    v56.SetGlobalBuffer(v2_y_ptr);
    int32_t v57 = v54 * c8192_i32;
    int32_t v58 = v57 + c8192_i32;
    bool v59 = v58 <= v3_size;
    int32_t v60;
    if (v59) {
      v60 = c8192_i32;
    } else {
      int32_t v61 = v3_size - v57;
      v60 = v61;
    }
    AscendC::GlobalTensor<bfloat16_t> v62 = v55[v57];
    int32_t v63 = v3_size - v57;
    bool v64 = v63 < c0_i32;
    int32_t v65 = v64 ? c0_i32 : v63;
    int32_t v66 = ((v60 < v65) ? (v60) : (v65));
    int32_t v67 = v66 * c2_i32;
    int32_t v68 = v3_size - v66;
    int32_t v69 = v67 % c32_i32;
    bool v70 = v69 == c0_i32;
    int32_t v71 = c32_i32 - v69;
    int32_t v72 = v70 ? c0_i32 : v71;
    int32_t v73 = v67 + v72;
    bool v74 = v73 < c16384_i32;
    if (v74) {
      get_buf(PIPE_V, 2, 0);
      AscendC::Duplicate(v19, c0_bf16, c8192_i64);
      rls_buf(PIPE_V, 2, 0);
    }
    int32_t v75 = v68 * c2_i32;
    int32_t v76 = v72 / c2_i32;
    int32_t v77 = c16384_i32 - v73;
    int32_t v78 = v77 / c32_i32;
    AscendC::DataCopyExtParams v79{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v67), static_cast<uint32_t>(v75), static_cast<uint32_t>(v78), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<bfloat16_t> v80{c1_i32, c0_i32, static_cast<uint8_t>(v76), c0_bf16};
    get_buf(PIPE_MTE2, 2, 0);
    AscendC::DataCopyPad(v19, v62, v79, v80);
    rls_buf(PIPE_MTE2, 2, 0);
    get_buf(PIPE_V, 3, 0);
    get_buf(PIPE_V, 2, 0);
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

        auto* in = reinterpret_cast<__ubuf__ bfloat16_t*>(v19.GetPhyAddr());
        auto* out = reinterpret_cast<__ubuf__ bfloat16_t*>(v18.GetPhyAddr());
        AscendC::Reg::RegTensor<float> x, a, z, u, p, tmp, e, y, one;
        uint32_t count = v19.GetSize();
        uint16_t loops = (count + 63) / 64;
        AscendC::Reg::MaskReg all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Duplicate(one, 1.0f, all);
        AscendC::Reg::RegTensor<float> c0,c1,c2,c3,c4,c5,c6,c7,c8;
        AscendC::Reg::Duplicate(c0, -0.82215223f, all);
        AscendC::Reg::Duplicate(c1, 1.48851587f, all);
        AscendC::Reg::Duplicate(c2, -1.13520398f, all);
        AscendC::Reg::Duplicate(c3, 0.27886807f, all);
        AscendC::Reg::Duplicate(c4, -0.18628806f, all);
        AscendC::Reg::Duplicate(c5, 0.09678418f, all);
        AscendC::Reg::Duplicate(c6, 0.37409196f, all);
        AscendC::Reg::Duplicate(c7, 1.00002368f, all);
        AscendC::Reg::Duplicate(c8, -1.26551223f, all);
        for (uint16_t i=0; i<loops; ++i) {
            auto mask = AscendC::Reg::UpdateMask<float>(count);

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(low, in+i*64);
        AscendC::Reg::Cast<float, bfloat16_t, up>(x, low, mask);

            AscendC::Reg::Abs(a, x, mask);
            AscendC::Reg::Muls(z, a, 0.7071067811865476f, mask);
            AscendC::Reg::Muls(u, z, 0.5f, mask);
            AscendC::Reg::Adds(u, u, 1.0f, mask);
            AscendC::Reg::Div(u, one, u, mask);
            AscendC::Reg::Duplicate(p, 0.17087277f, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c0, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c1, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c2, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c3, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c4, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c5, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c6, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c7, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c8, mask);
            AscendC::Reg::Mul(tmp, z, z, mask);
            AscendC::Reg::Sub(p, p, tmp, mask);
            AscendC::Reg::Exp(e, p, mask);
            AscendC::Reg::Mul(e, e, u, mask);
            AscendC::Reg::Muls(a, a, 0.5f, mask);
            AscendC::Reg::Mul(e, e, a, mask);
            AscendC::Reg::Muls(y, x, 0.5f, mask);
            AscendC::Reg::Add(y, y, a, mask);
            AscendC::Reg::Sub(y, y, e, mask);
            // Positive infinity has GeLU(+inf)=+inf; avoid inf*erfc(inf)=NaN.
            AscendC::Reg::MaskReg positiveInf;
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(
                positiveInf, (AscendC::Reg::RegTensor<uint32_t>&)x, 0x7f800000u, mask);
            AscendC::Reg::Select(y, x, y, positiveInf);

        AscendC::Reg::Cast<bfloat16_t, float, down>(low, y, mask);
        AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out+i*64, low, mask);

        }
        ;
      }
    }
    rls_buf(PIPE_V, 2, 0);
    rls_buf(PIPE_V, 3, 0);
    AscendC::GlobalTensor<bfloat16_t> v81 = v56[v57];
    int32_t v82 = c8192_i32 - v66;
    int32_t v83 = v82 * c2_i32;
    int32_t v84 = v83 / c32_i32;
    AscendC::DataCopyExtParams v85{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v67), static_cast<uint32_t>(v84), static_cast<uint32_t>(v75), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 3, 0);
    AscendC::DataCopyPad(v81, v18, v85);
    rls_buf(PIPE_MTE3, 3, 0);
  }
  AscendC::LocalTensor<bfloat16_t> v86{AscendC::TPosition::VECCALC, 65536, 8192};
  AscendC::LocalTensor<bfloat16_t> v87{AscendC::TPosition::VECCALC, 81920, 8192};
  for (int32_t v88 = v16; v88 < v7; v88 += v8) {
    AscendC::GlobalTensor<bfloat16_t> v89;
    v89.SetGlobalBuffer(v1_x_ptr);
    AscendC::GlobalTensor<bfloat16_t> v90;
    v90.SetGlobalBuffer(v2_y_ptr);
    int32_t v91 = v88 * c8192_i32;
    int32_t v92 = v91 + c8192_i32;
    bool v93 = v92 <= v3_size;
    int32_t v94;
    if (v93) {
      v94 = c8192_i32;
    } else {
      int32_t v95 = v3_size - v91;
      v94 = v95;
    }
    AscendC::GlobalTensor<bfloat16_t> v96 = v89[v91];
    int32_t v97 = v3_size - v91;
    bool v98 = v97 < c0_i32;
    int32_t v99 = v98 ? c0_i32 : v97;
    int32_t v100 = ((v94 < v99) ? (v94) : (v99));
    int32_t v101 = v100 * c2_i32;
    int32_t v102 = v3_size - v100;
    int32_t v103 = v101 % c32_i32;
    bool v104 = v103 == c0_i32;
    int32_t v105 = c32_i32 - v103;
    int32_t v106 = v104 ? c0_i32 : v105;
    int32_t v107 = v101 + v106;
    bool v108 = v107 < c16384_i32;
    if (v108) {
      get_buf(PIPE_V, 4, 0);
      AscendC::Duplicate(v87, c0_bf16, c8192_i64);
      rls_buf(PIPE_V, 4, 0);
    }
    int32_t v109 = v102 * c2_i32;
    int32_t v110 = v106 / c2_i32;
    int32_t v111 = c16384_i32 - v107;
    int32_t v112 = v111 / c32_i32;
    AscendC::DataCopyExtParams v113{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v101), static_cast<uint32_t>(v109), static_cast<uint32_t>(v112), static_cast<uint32_t>(c0_i32)};
    AscendC::DataCopyPadExtParams<bfloat16_t> v114{c1_i32, c0_i32, static_cast<uint8_t>(v110), c0_bf16};
    get_buf(PIPE_MTE2, 4, 0);
    AscendC::DataCopyPad(v87, v96, v113, v114);
    rls_buf(PIPE_MTE2, 4, 0);
    get_buf(PIPE_V, 5, 0);
    get_buf(PIPE_V, 4, 0);
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

        auto* in = reinterpret_cast<__ubuf__ bfloat16_t*>(v87.GetPhyAddr());
        auto* out = reinterpret_cast<__ubuf__ bfloat16_t*>(v86.GetPhyAddr());
        AscendC::Reg::RegTensor<float> x, a, z, u, p, tmp, e, y, one;
        uint32_t count = v87.GetSize();
        uint16_t loops = (count + 63) / 64;
        AscendC::Reg::MaskReg all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Duplicate(one, 1.0f, all);
        AscendC::Reg::RegTensor<float> c0,c1,c2,c3,c4,c5,c6,c7,c8;
        AscendC::Reg::Duplicate(c0, -0.82215223f, all);
        AscendC::Reg::Duplicate(c1, 1.48851587f, all);
        AscendC::Reg::Duplicate(c2, -1.13520398f, all);
        AscendC::Reg::Duplicate(c3, 0.27886807f, all);
        AscendC::Reg::Duplicate(c4, -0.18628806f, all);
        AscendC::Reg::Duplicate(c5, 0.09678418f, all);
        AscendC::Reg::Duplicate(c6, 0.37409196f, all);
        AscendC::Reg::Duplicate(c7, 1.00002368f, all);
        AscendC::Reg::Duplicate(c8, -1.26551223f, all);
        for (uint16_t i=0; i<loops; ++i) {
            auto mask = AscendC::Reg::UpdateMask<float>(count);

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(low, in+i*64);
        AscendC::Reg::Cast<float, bfloat16_t, up>(x, low, mask);

            AscendC::Reg::Abs(a, x, mask);
            AscendC::Reg::Muls(z, a, 0.7071067811865476f, mask);
            AscendC::Reg::Muls(u, z, 0.5f, mask);
            AscendC::Reg::Adds(u, u, 1.0f, mask);
            AscendC::Reg::Div(u, one, u, mask);
            AscendC::Reg::Duplicate(p, 0.17087277f, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c0, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c1, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c2, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c3, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c4, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c5, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c6, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c7, mask);
            AscendC::Reg::FusedMulDstAdd(p, u, c8, mask);
            AscendC::Reg::Mul(tmp, z, z, mask);
            AscendC::Reg::Sub(p, p, tmp, mask);
            AscendC::Reg::Exp(e, p, mask);
            AscendC::Reg::Mul(e, e, u, mask);
            AscendC::Reg::Muls(a, a, 0.5f, mask);
            AscendC::Reg::Mul(e, e, a, mask);
            AscendC::Reg::Muls(y, x, 0.5f, mask);
            AscendC::Reg::Add(y, y, a, mask);
            AscendC::Reg::Sub(y, y, e, mask);
            // Positive infinity has GeLU(+inf)=+inf; avoid inf*erfc(inf)=NaN.
            AscendC::Reg::MaskReg positiveInf;
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(
                positiveInf, (AscendC::Reg::RegTensor<uint32_t>&)x, 0x7f800000u, mask);
            AscendC::Reg::Select(y, x, y, positiveInf);

        AscendC::Reg::Cast<bfloat16_t, float, down>(low, y, mask);
        AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out+i*64, low, mask);

        }
        ;
      }
    }
    rls_buf(PIPE_V, 4, 0);
    rls_buf(PIPE_V, 5, 0);
    AscendC::GlobalTensor<bfloat16_t> v115 = v90[v91];
    int32_t v116 = c8192_i32 - v100;
    int32_t v117 = v116 * c2_i32;
    int32_t v118 = v117 / c32_i32;
    AscendC::DataCopyExtParams v119{static_cast<uint16_t>(c1_i32), static_cast<uint32_t>(v101), static_cast<uint32_t>(v118), static_cast<uint32_t>(v109), static_cast<uint32_t>(c0_i32)};
    get_buf(PIPE_MTE3, 5, 0);
    AscendC::DataCopyPad(v115, v86, v119);
    rls_buf(PIPE_MTE3, 5, 0);
  }
  return;
}
