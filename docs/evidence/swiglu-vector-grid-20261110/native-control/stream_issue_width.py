"""Compare chunk ownership and vector alignment for a four-wave FP32 stream."""
import re
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from fp32_stream_remainder import SOURCE_SHA


def render(source, mode):
    if sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError("expected the clean W59 source")
    begin = source.index("@asctile.jit", source.index("def paired_rows("))
    end = source.index("\n\n#", begin)
    fn = source[begin:end]
    assert "def single_row_edge(" in fn
    fn = fn.replace("single_row_edge", "stream_issue")
    offset = "        col = (step * blocks + block) * tile"
    if mode == "local":
        fn = fn.replace(offset, "        col = (block * waves + step) * tile")
    elif mode == "mixed":
        # The last wave absorbs the difference in DMA-aligned chunks.
        a = fn.index("    for step in ")
        b = fn.index("    tail = ", a)
        loop, body = fn[a:b].split(offset + "\n")
        regular = body.replace("[tile]", "[bulk]")
        last = body.replace("[tile]", "[edge]")
        last = re.sub(r"\b(gate|value|g|out)\b", r"edge_\1", last)
        fn = (fn[:a] + "    bulk = tile // 64 * 64\n"
              "    edge = waves * tile - (waves - 1) * bulk\n"
              + loop.replace("range(waves, unroll_factor=2)", "range(waves - 1, unroll_factor=2)")
              + offset.replace("* tile", "* bulk") + "\n" + regular
              + "    col = (waves - 1) * blocks * bulk + block * edge\n"
              + "\n".join(line[4:] for line in last.splitlines()) + "\n" + fn[b:])
    else:
        raise ValueError(mode)
    mark = "            single_row_edge[aivs](x, y, half, tile, aivs)"
    assert source.count(mark) == 1
    guard = ("            if waves == 4 and tile >= 64 and "
             "24 * (4 * tile - 3 * (tile // 64 * 64)) <= usable_ub:\n")
    return (source[:end] + "\n\n" + fn + source[end:]).replace(
        mark, guard + "    " + mark.replace("single_row_edge", "stream_issue") + "\n            else:\n    " + mark)


if __name__ == "__main__":
    base = Path("base/swi_glu.py").read_text()
    for mode in ("local", "mixed"):
        text = render(base, mode)
        dest = Path(mode)
        compile(text, str(dest), "exec")
        dest.mkdir()
        (dest / "swi_glu.py").write_text(text)
        (dest / "__init__.py").touch()
        copyfile("base/_pyasc_runtime.py", dest / "_pyasc_runtime.py")
