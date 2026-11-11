"""A finite, shape-driven specialization space shared by PyASC and the submitted host."""
from math import ceil, prod

DTYPES = {'float16': 2, 'bfloat16': 2, 'float32': 4}
WIDTHS = (16, 32, 64, 128, 256, 512, 768, 1024, 1536)


def spec(dtype, half):
    size = DTYPES[dtype]
    if dtype == 'bfloat16' and half == 1:
        return '_swiglu_pair_groups', (2048,)
    if half * size <= 32:
        # Bound one aligned slab by the input DMA target and both int32 index tiles.
        limit = min(128, 16384 // (2 * half * size), 65536 // (28 * half))
        return 'slab', (half, 1 << (limit.bit_length() - 1))
    if half * size <= 3072:
        width = next(w for w in WIDTHS if w >= half)
        return 'rows', (min(64, max(1, 6144 // (width * size))), width)
    return 'flat', (4096,)


def all_specs():
    for dtype, size in DTYPES.items():
        variants = {spec(dtype, h) for h in range(1, 3072 // size + 2)}
        for route, const in sorted(variants):
            yield dtype, route, const


def key(dtype, route, const):
    return '-'.join([dtype, route, *(str(v) for v in const)])


def layout(shape, dim, dtype):
    if not -len(shape) <= dim < len(shape):
        raise ValueError('Invalid split axis')
    axis = dim % len(shape)
    if shape[axis] % 2:
        raise ValueError('SwiGLU requires an even split extent')
    out = list(shape)
    out[axis] //= 2
    rows, half = prod(shape[:axis]), prod(out[axis:])
    if not rows or not half:
        return out, rows, half, None, (), 0
    route, const = spec(dtype, half)
    if route == 'flat':
        groups = rows * ceil(half / const[0])
    else:
        tile = const[1] if route == 'slab' else const[0]
        groups = ceil(rows / tile)
    return out, rows, half, route, const, groups
