"""Compare in-grid remainder schedules against the qualified FP32 stream."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

SOURCE_SHA = "1b8f41693c926f664f94f62f5b69d3141c08762147cc794e61de12cf0b3ebb1d"


def render(source, mode):
    if sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError("expected the W59 clean generation")
    start = source.index("    waves = ", source.index("def single_row_edge("))
    end = source.index("\n\n#", start)
    if mode == "rounded":
        # Keep four waves and assign a bounded final chunk in the original grid.
        body = """    waves = half // (blocks * tile)
    width = ((half + 8 * waves * blocks - 1) // (8 * waves * blocks)) * 8
    for step in asctile.range(waves, unroll_factor=2):
        col = (step * blocks + block) * width
        if col < half:
            valid = min(width, half - col)
            gate = asctile.copy_in(gx, [col], [width], real_shape=[valid], pad_value=0.0)
            value = asctile.copy_in(gx, [half + col], [width], real_shape=[valid], pad_value=0.0)
            out = (gate / (1.0 + asctile.exp(-gate))) * value
            asctile.copy_out(out, gy, [col], real_shape=[valid])
"""
    elif mode == "merged":
        # The final owner combines a sub-line remainder with its last full chunk.
        # The original dispatcher is used on all other geometries.
        body = """    waves = half // (blocks * tile)
    rem = half % (blocks * tile)
    for step in asctile.range(waves, unroll_factor=2):
        col = (step * blocks + block) * tile
        if block == blocks - 1 and step == waves - 1:
            gate = asctile.copy_in(gx, [col], [tile + rem])
            value = asctile.copy_in(gx, [half + col], [tile + rem])
            out = (gate / (1.0 + asctile.exp(-gate))) * value
            asctile.copy_out(out, gy, [col])
        else:
            gate = asctile.copy_in(gx, [col], [tile])
            value = asctile.copy_in(gx, [half + col], [tile])
            out = (gate / (1.0 + asctile.exp(-gate))) * value
            asctile.copy_out(out, gy, [col])
"""
    else:
        raise ValueError(mode)
    head = source[source.index("def single_row_edge("):start]
    kernel = "@asctile.jit(reuse_alloc=2)\n" + head.replace("single_row_edge(", "edge_variant(") + body
    marker = "            single_row_edge[aivs](x, y, half, tile, aivs)"
    assert source.count(marker) == 1
    return (source[:end] + "\n\n" + kernel + source[end:]).replace(
        marker, "            if 0 < tail < 8 and 24 * (tile + 8) <= usable_ub:\n"
        "                edge_variant[aivs](x, y, half, tile, aivs)\n"
        "            else:\n    " + marker)


if __name__ == "__main__":
    s = Path("base/swi_glu.py").read_text()
    for mode in ("merged", "rounded"):
        dest = Path(mode)
        code = render(s, mode)
        compile(code, str(dest), "exec")
        dest.mkdir()
        (dest / "__init__.py").touch()
        (dest / "swi_glu.py").write_text(code)
        copyfile("base/_pyasc_runtime.py", dest / "_pyasc_runtime.py")
