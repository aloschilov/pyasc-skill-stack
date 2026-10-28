import torch
from asc.experimental import asctile

CORE_NUM = 8
SIZE = 1024
TILE_SIZE = 256


@asctile.jit
def add_kernel(x_ptr: asctile.GlobalAddress,
               y_ptr: asctile.GlobalAddress,
               out_ptr: asctile.GlobalAddress,
               size: int,
               tile_size: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    for tile_id in asctile.range(asctile.block_idx(),
                                 asctile.ceildiv(size, tile_size),
                                 asctile.block_num(),
                                 unroll_factor=1):
        offset = tile_id * tile_size
        valid = min(size - offset, tile_size)
        x = asctile.copy_in(x_gm, [offset], [tile_size],
                            real_shape=[valid], pad_value=0)
        y = asctile.copy_in(y_gm, [offset], [tile_size],
                            real_shape=[valid], pad_value=0)
        out = x + y
        asctile.copy_out(out, out_gm, [offset], real_shape=[valid])


def main():
    x = torch.randn(SIZE, dtype=torch.float32)
    y = torch.randn(SIZE, dtype=torch.float32)
    out = torch.empty(SIZE, dtype=torch.float32)

    add_kernel[CORE_NUM](x, y, out, SIZE, TILE_SIZE)

    expected = x + y
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6), (
        "vector add result mismatch"
    )


if __name__ == "__main__":
    main()
