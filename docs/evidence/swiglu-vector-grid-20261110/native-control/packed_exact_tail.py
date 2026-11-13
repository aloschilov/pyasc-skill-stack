"""Mechanism control: compute the true final rows, holding the full-block VF fixed."""
from pathlib import Path
from shutil import copyfile
from packed_vector_loop import loop_body
from packed_row_vf import register_body

def render(source, rows, half, height, loop):
    start = source.index('    packed = asctile.copy_in',source.index('def packed_vector('))
    stop = source.index('\n\n@asctile.jit',start)
    main = source[start:stop].strip('\n')
    tail = rows % height
    assert tail and rows // height
    pitch = (half + 7)//8*8
    vf = loop_body(tail,pitch) if loop else register_body(tail,pitch,1)
    branch = f"""    if b == R // T:
        packed = asctile.copy_in(gx, [2 * (R // T) * T, 0], [{2*tail}, W],
                                 real_shape=[{2*tail}, H], pad_value=0.0)
        out = asctile.inline_vf({vf!r}, [{tail}, W], asctile.float32, [packed])
        asctile.copy_out(out, gy, [(R // T) * T, 0], real_shape=[{tail}, H])
    else:
"""
    return source[:start]+branch+'\n'.join('    '+line for line in main.splitlines())+'\n'+source[stop:]

if __name__=='__main__':
    for k,r,h,t,lp in [('v11loopcache',273,509,8,True),('v13staticcache',143,536,4,False)]:
        src = Path(k)/'swi_glu.py'
        s = render(src.read_text(),r,h,t,lp)
        assert compile(s,k,'exec')
        d = Path(k+'exact')
        d.mkdir(exist_ok=False)
        (d/'swi_glu.py').write_text(s)
        copyfile(src.parent/'_pyasc_runtime.py',d/'_pyasc_runtime.py')
        (d/'__init__.py').touch()
