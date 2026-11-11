"""A finite, shape-driven specialization space shared by PyASC and the submitted host."""
from math import ceil, prod

DTYPES = {'float16': 2, 'bfloat16': 2, 'float32': 4}

def natural(value, maximum):
    return type(value) is int and 1 <= value <= maximum


def validate(policy):
    """Check the OpenCode-authored decision table before lowering or submitting."""
    if not isinstance(policy,dict) or set(policy) != set(DTYPES):
        raise ValueError('policy.json must contain one ordered band list per dtype')
    for dtype,size in DTYPES.items():
        bands = policy[dtype]
        if not isinstance(bands,list) or not 1 <= len(bands) <= 32:
            raise ValueError('Expected 1..32 width bands for '+dtype)
        lower = 0
        for i,b in enumerate(bands):
            required = {'through','route','tile','unroll','tiles_per_block','block_cap','unroll_waves'}
            if not isinstance(b,dict) or not required <= b.keys() <= required | {'width_step_bytes'}:
                raise ValueError('Invalid fields in geometry band for '+dtype)
            end = b['through']
            if i == len(bands)-1:
                if end is not None or b['route'] != 'flat':
                    raise ValueError('The final band must cover any wider half using flat')
            elif not natural(end,65536) or end <= lower:
                raise ValueError('Expected increasing half-width bounds')
            if b['route'] not in ('rows','slab','flat') or not natural(b['tile'],32768):
                raise ValueError('Invalid route or tile')
            if b['route']=='slab' and end>32:
                raise ValueError('Short gather domain too large')
            if 'width_step_bytes' in b and (b['route']!='rows' or not natural(b['width_step_bytes'],65536)
                                              or b['width_step_bytes']%32):
                raise ValueError('Row width stepping must be a multiple of 32 bytes')
            if b['route']=='flat' and b['tile']*size%32:
                raise ValueError('Flat physical tiles require DMA alignment')
            if (not natural(b['unroll'],2) or not natural(b['tiles_per_block'],128)
                    or not natural(b['unroll_waves'],128)
                    or not (b['block_cap'] is None or natural(b['block_cap'],1024))):
                raise ValueError('Invalid grid/unroll policy')
            lower=end
    return policy


def band_for(policy,dtype,half):
    return next(b for b in policy[dtype] if b['through'] is None or half<=b['through'])


def specialization(dtype, half, band, unroll):
    route,tile = band['route'],band['tile']
    if route == 'slab':
        return (half,tile,unroll)
    if route == 'rows':
        align=32//DTYPES[dtype]
        limit=ceil(band['through']/align)*align
        step=band.get('width_step_bytes',limit*DTYPES[dtype])//DTYPES[dtype]
        return (tile,min(limit,ceil(half/step)*step),unroll)
    return (tile,unroll)


def slab_indices(half, tile):
    """Address constants for a slab; independent of input data and device."""
    gate = [row * 2 * half + column for row in range(tile) for column in range(half)]
    return gate, [lane + half for lane in gate]


def all_specs(policy):
    for dtype,bands in validate(policy).items():
        variants = {('_swiglu_pair_groups',(2048,))} if dtype=='bfloat16' else set()
        start=1
        for b in bands:
            end=b['through'] or start
            if b['route']=='slab':
                halves=range(start,end+1)
            elif 'width_step_bytes' in b:
                step=b['width_step_bytes']//DTYPES[dtype]
                halves=[*range((start+step-1)//step*step,end,step),end]
            else:
                halves=(end,)
            for half in halves:
                if dtype=='bfloat16' and half==1 and b['route']=='slab':
                    continue
                for u in {1,b['unroll']}:
                    variants.add((b['route'],specialization(dtype,half,b,u)))
            start=end+1
        for route,const in sorted(variants):
            yield dtype,route,const


def key(dtype, route, const):
    return '-'.join([dtype, route, *(str(v) for v in const)])


def layout(shape, dim, dtype, policy, cores):
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
    if cores < 1:
        raise ValueError('No available vector cores')
    if dtype=='bfloat16' and half==1:
        groups=ceil(rows/2048)
        return out,rows,half,'_swiglu_pair_groups',(2048,),min(groups,cores)
    b=band_for(policy,dtype,half)
    groups=rows*ceil(half/b['tile']) if b['route']=='flat' else ceil(rows/b['tile'])
    blocks=min(cores,b['block_cap'] or cores,ceil(groups/b['tiles_per_block']))
    u=b['unroll'] if groups>=blocks*b['unroll_waves'] else 1
    return out,rows,half,b['route'],specialization(dtype,half,b,u),blocks
