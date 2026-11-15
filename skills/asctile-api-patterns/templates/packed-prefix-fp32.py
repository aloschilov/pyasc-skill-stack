"""Register-enabled 950PR one-wave FP32 SwiGLU with a partial leading vector."""

from asc.experimental import asctile


def use_packed_prefix(rows, half, pitch, height, aivs, usable_ub):
    if min(rows, half, height, aivs, usable_ub) <= 0:
        return False
    return (pitch == half and half >= 64 and half % 8 == 0
            and (half * 4) % 128 != 0
            and (rows + height - 1) // height <= aivs
            and 12 * height * half <= usable_ub)


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def packed_prefix_rows(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                       R: asctile.ConstExpr[int], H: asctile.ConstExpr[int],
                       T: asctile.ConstExpr[int]):
    block = asctile.block_idx()
    count = asctile.block_num()
    tile = (block + count // 2) % count
    r = tile * T
    valid = min(T, R - r)
    gx = asctile.global_tensor(x, [2 * R * H], l2_cache=False)
    slab = asctile.copy_in(gx, [2 * r * H], [2 * T * H],
                           real_shape=[2 * valid * H], pad_value=0.0)
    result = asctile.inline_vf(f"""
        using namespace AscendC::Reg;
        auto* src = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
        auto* dst = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
        for (uint16_t row = 0; row < {T}; ++row) {{
            {{
                uint32_t active = {H % 64};
                auto mask = UpdateMask<float>(active);
                RegTensor<float> gate, value, acc;
                LoadAlign(gate, src + (2 * row) * {H});
                LoadAlign(value, src + (2 * row + 1) * {H});
                Muls(acc, gate, -1.0f, mask);
                Exp(acc, acc, mask);
                Adds(acc, acc, 1.0f, mask);
                Div(acc, gate, acc, mask);
                Mul(acc, acc, value, mask);
                StoreAlign(dst + row * {H}, acc, mask);
            }}
            for (uint16_t c = 0; c < {H // 64}; ++c) {{
                uint32_t offset = {H % 64} + c * 64;
                auto mask = CreateMask<uint8_t>();
                RegTensor<float> gate, value, acc;
                LoadAlign(gate, src + (2 * row) * {H} + offset);
                LoadAlign(value, src + (2 * row + 1) * {H} + offset);
                Muls(acc, gate, -1.0f, mask);
                Exp(acc, acc, mask);
                Adds(acc, acc, 1.0f, mask);
                Div(acc, gate, acc, mask);
                Mul(acc, acc, value, mask);
                StoreAlign(dst + row * {H} + offset, acc, mask);
            }}
        }}
    """, [T, H], asctile.float32, [slab])
    gy = asctile.global_tensor(y, [R, H])
    asctile.copy_out(result, gy, [r, 0], real_shape=[valid, H])
