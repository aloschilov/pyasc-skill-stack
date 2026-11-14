"""Factor fixed row height and column parallelism on aligned FP32 row bands."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

SOURCE_SHA = "26c6dad852aa263be1b44d7a8b258c498ba36915ec72d07bc5ee264f98a02453"


def render(source, bands):
    if sha256(source.encode()).hexdigest() != SOURCE_SHA or bands not in (1, 2, 4):
        raise ValueError("expected the W62 source and one, two or four column bands")
    marker = "                row_single_wave_fp32[blocks](input, out, R, H, height, W)"
    if source.count(marker) != 1:
        raise ValueError("unexpected FP32 entry point")
    insertion = source.index("@asctile.jit", source.index("def row_single_wave_fp32("))
    # A control holds height at eleven rows. Each owner gets a complete
    # rectangle. A shorter terminal band is compiled with its own exact width.
    head = f"""@asctile.jit(reuse_alloc=2)
def exact_row_bands(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                    R: asctile.ConstExpr[int], H: asctile.ConstExpr[int],
                    T: asctile.ConstExpr[int], S: asctile.ConstExpr[int]):
    block = asctile.block_idx()
    row = (block // {bands}) * T
    col = (block % {bands}) * S
"""
    def body(width, pad):
        lines = [
            "src = asctile.global_tensor(x + 2 * row * H, [T, 2 * H], l2_cache=False)",
            "dst = asctile.global_tensor(y + row * H, [T, H])",
            f"gate = asctile.copy_in(src, [0, col], [T, {width}])",
            f"value = asctile.copy_in(src, [0, col + H], [T, {width}])",
            "result = (gate / (1.0 + asctile.exp(-gate))) * value",
            "asctile.copy_out(result, dst, [0, col])",
        ]
        return "".join(" " * pad + line + "\n" for line in lines)

    if bands == 1:
        head += body("H", 4)
    else:
        head += f"    if block % {bands} == {bands - 1}:\n"
        head += body(f"H - {bands - 1} * S", 8)
        head += "    else:\n" + body("S", 8)
    launch = f"""                exact_h = 11
                bands = {bands}
                band = ((H + 8 * bands - 1) // (8 * bands)) * 8
                owners = (R // exact_h) * bands
                if (R % exact_h == 0 and H % 8 == 0 and 0 < owners <= aivs
                        and (bands - 1) * band < H and 28 * exact_h * band <= usable_ub):
                    exact_row_bands[owners](input, out, R, H, exact_h, band)
                else:
    {marker}"""
    return (source[:insertion] + head + "\n\n" + source[insertion:]).replace(marker, launch)


if __name__ == "__main__":
    base = Path("base/swi_glu.py")
    for bands in (1, 2, 4):
        name = Path(f"bands{bands}")
        source = render(base.read_text(), bands)
        compile(source, str(name), "exec")
        name.mkdir()
        (name / "swi_glu.py").write_text(source)
        (name / "__init__.py").touch()
        copyfile(base.with_name("_pyasc_runtime.py"), name / "_pyasc_runtime.py")
