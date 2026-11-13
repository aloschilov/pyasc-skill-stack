"""Compare repeated public launches with and without fresh argument transfers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys


def qualify(profiles):
    schedule = []
    for r in (1, 2):
        order = [(r, c, mode) for c in (11, 13) for mode in ("staged", "resident")]
        random.Random(11890 + r).shuffle(order)
        schedule.extend(order)
    Path("residency").mkdir()
    Path("residency/plan.json").write_text(json.dumps(schedule))
    for r, c, mode in schedule:
        pkg = f"c{c}w1r0"
        cache = profiles / pkg / "cache"
        image, = cache.glob("*/staged_rows.o")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        expected = {11: "d3458bbf1036399401b675c9198e5b4fef4439753fe752c76632010d8023f42b",
                    13: "bf3fa34312f61d30d31a195e1202bf59bb13faa73cc5a9e6def1b7d3211c0890"}
        if digest != expected[c]:
            raise ValueError(f"starting cache differs from measured wheel ELF for {c}: {digest}")
        stem = Path(f"residency/{r}.{c}.{mode}")
        env = {**os.environ, "PYASC_CACHE_DIR": str(cache.resolve()),
               "PYASC_DUMP_PATH": str((profiles / pkg / "dump").resolve())}
        with stem.with_suffix(stem.suffix + ".json").open("w") as out, stem.with_suffix(stem.suffix + ".log").open("w") as log:
            proc = subprocess.run([sys.executable, str(Path(__file__).with_name("resident_replay.py")),
                                   pkg, str(c), mode], stdout=out, stderr=log, env=env)
        if list(cache.glob("*/staged_rows.o")) != [image] or hashlib.sha256(image.read_bytes()).hexdigest() != digest:
            raise ValueError(f"measured code changed for {c}")
        with Path("residency/status").open("a") as status:
            print(r, c, mode, proc.returncode, digest, file=status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path)
    qualify(parser.parse_args().profiles)
