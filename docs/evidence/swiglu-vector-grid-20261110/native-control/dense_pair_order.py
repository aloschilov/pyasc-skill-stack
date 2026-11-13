"""Requalify dense-pair launch geometry and block order on an unedited generation."""
import csv
import hashlib
import json
from pathlib import Path
from shutil import copyfile

SOURCE_SHA = "18ea91c301937c1f8c901f6d9fa529e952c8cea40c06a4458f322b51113fd607"
POLICIES = ("base", "small", "pair", "smallpair")


def render(source, policy):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError("expected the unedited a7f4048 generation")
    if policy not in POLICIES:
        raise ValueError(policy)
    if policy in ("small", "smallpair"):
        target = "                idx = ordinary_gate_index(H, tile, input.device, _GATE_CACHE)"
        assert source.count(target) == 1
        source = source.replace(target, """                if H == 1:
                    tiles = (R + tile - 1) // tile
                    compact = 2 * ((tiles + 7) // 8)
                    if compact <= blocks and 3 * blocks < tiles <= 4 * blocks:
                        blocks = compact
""" + target)
    if policy in ("pair", "smallpair"):
        start = source.index("def ordinary_bf16_slab(")
        end = source.index("def bounded_rows(", start)
        body = source[start:end]
        target = "    for t in asctile.range(asctile.block_idx(), count, asctile.block_num(),"
        assert body.count(target) == 1
        source = source[:start] + body.replace(target, """    block = asctile.block_idx()
    blocks = asctile.block_num()
    owner = (block % 2) * (blocks // 2) + block // 2 if half == 1 and blocks % 2 == 0 else block
    for t in asctile.range(owner, count, blocks,""") + source[end:]
    return source


if __name__ == "__main__":
    source = Path("base/swi_glu.py").read_text()
    for policy in POLICIES[1:]:
        dest = Path(policy)
        code = render(source, policy)
        compile(code, str(dest), "exec")
        dest.mkdir()
        (dest / "__init__.py").touch()
        (dest / "swi_glu.py").write_text(code)
        copyfile("base/_pyasc_runtime.py", dest / "_pyasc_runtime.py")
    task = Path("generation/task")
    neighbor = Path("neighbor/generation/task")
    neighbor.mkdir(parents=True)
    with (task / "cases.csv").open() as f:
        row, = (r for r in csv.DictReader(f) if int(r["case_id"]) == 12)
    row.update(input_shape=json.dumps([[911117, 2]]), value_range="[-3, 3]",
               note="Finite off-suite dense BF16 pair stream")
    with (neighbor / "cases.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader()
        w.writerow(row)
    copyfile(task / "golden.py", neighbor / "golden.py")
    for policy in POLICIES:
        (Path("neighbor") / policy).symlink_to(Path(policy).resolve(), target_is_directory=True)
