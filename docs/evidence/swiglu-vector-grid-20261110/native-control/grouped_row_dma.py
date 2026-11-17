"""Compare contiguous and interleaved row ownership with rank-two DMA."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from textwrap import indent

SOURCE_SHA = "1b8f41693c926f664f94f62f5b69d3141c08762147cc794e61de12cf0b3ebb1d"


def render(source, group):
    if group not in (2, 4) or sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError("expected W59 and a two- or four-block group")
    start = source.index("def single_wave_rows(")
    split = source.index("    r = tile_index * T", start)
    end = source.index("\n\n#", split)
    fn = source[start:split].replace("def single_wave_rows(", "def grouped_rows(")
    bounded = source[split:end].rstrip()
    # Each GM descriptor contains only complete groups of physical rows.
    # The remaining blocks use the original guarded descriptor and ownership.
    regular = bounded[bounded.index("    if W > H:"):]
    regular = regular.replace("[r, H]", "[r, (2 * lane + 1) * H]")
    regular = regular.replace("copy_in(gx, [r, 0]", "copy_in(gx, [r, 2 * lane * H]")
    regular = regular.replace("gy, [r, 0]", "gy, [r, lane * H]")
    regular = regular.replace("[valid, H]", "[T, H]")
    regular = regular.replace("copy_in(gx,", "copy_in(source_groups,")
    regular = regular.replace(", gy,", ", output_groups,")
    fn += f"""    group = tile_index // {group}
    lane = tile_index % {group}
    if group < R // ({group} * T):
        source_groups = asctile.global_tensor(x, [R // {group}, {2 * group} * H], l2_cache=False)
        output_groups = asctile.global_tensor(y, [R // {group}, {group} * H])
        r = group * T
"""
    fn += indent(regular, "    ") + "\n    else:\n" + indent(bounded, "    ")
    marker = "single_wave_rows[blocks](input, out, rows, half, h, width, rotate)"
    assert source.count(marker) == 1
    return (source[:end] + "\n\n@asctile.jit(reuse_alloc=2)\n" + fn + "\n" +
            source[end:]).replace(marker, marker.replace("single_wave_rows", "grouped_rows"))


if __name__ == "__main__":
    base = Path("base/swi_glu.py").read_text()
    for group in (2, 4):
        dest = Path(f"group{group}")
        code = render(base, group)
        compile(code, str(dest), "exec")
        dest.mkdir()
        (dest / "__init__.py").touch()
        (dest / "swi_glu.py").write_text(code)
        copyfile("base/_pyasc_runtime.py", dest / "_pyasc_runtime.py")
