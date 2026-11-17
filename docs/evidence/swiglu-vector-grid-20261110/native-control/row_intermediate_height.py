"""Measure intermediate heights on the generated one-wave FP32 row path."""
import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from row_register_chains import W65


def render(source, quarters):
    if sha256(source.encode()).hexdigest() != W65 or quarters not in (3, 5):
        raise ValueError("expected W65 and a three-quarter or five-quarter tile")
    marker = "        if blocks >= tiles and dtype == torch.float32:\n"
    if source.count(marker) != 1:
        raise ValueError("missing the qualified single-wave dispatch")
    # Keep the original eligibility decision. The alternate geometry must
    # independently fit one wave and the same conservative UB budget.
    return source.replace(marker, marker + f"""            trial = min(R, max(1, height * {quarters} // 4))
            trial_tiles = (R + trial - 1) // trial
            if trial_tiles <= aivs and trial * width * bytes_per_lane_one <= usable_ub:
                height = trial
                blocks = trial_tiles
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("quarters", type=int, choices=(3, 5))
    args = parser.parse_args()
    source = render((args.package / "swi_glu.py").read_text(), args.quarters)
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(source)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
