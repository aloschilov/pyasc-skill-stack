"""Isolate the two input DMA pad parameters in a process-local W65 compiler control."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

from ub_spacing import SOURCES
from row_register_chains import W65
from vf_disassembly import sections


def render(source, case, mode):
    if sha256(source.encode()).hexdigest() != SOURCES[case]:
        raise ValueError("unexpected AscendC input to the pad control")
    variable, right = {11: ("v27", 3), 13: ("v30", 0)}[case]
    if mode not in ("base", "zero", "implicit"):
        raise ValueError(mode)
    prefix = f"AscendC::DataCopyPadExtParams<float> {variable}"
    def initializer(enabled, lanes):
        return f"{prefix}{{c{enabled}_i32, c0_i32, static_cast<uint8_t>(c{lanes}_i32), c0_f32}};"
    old = initializer(1, right)
    if source.count(old) != 1 or source.count(f", {variable});") != 2:
        raise ValueError("expected one pad configuration shared by the two input reads")
    if mode == "base":
        return source
    return source.replace(old, initializer(int(mode == "zero"), 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("base", "zero", "implicit"))
    parser.add_argument("case", type=int, choices=(11, 13))
    parser.add_argument("package", type=Path)
    parser.add_argument("dump", type=Path)
    parser.add_argument("original_dump", type=Path)
    parser.add_argument("worker", type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sha256((args.package / "swi_glu.py").read_bytes()).hexdigest() != W65:
        raise ValueError("expected the independent W65 Python module")
    from asc.runtime.compiler import Compiler
    original_translate = Compiler.run_translation
    emitted = []

    def translate(module):
        source = render(original_translate(module), args.case, args.mode)
        emitted.append(source)
        return source

    sys.argv = [str(args.worker), *args.worker_args]
    with patch.object(Compiler, "run_translation", staticmethod(translate)):
        try:
            runpy.run_path(str(args.worker), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    if len(emitted) != 1 or (args.dump / "ascendc.cpp").read_text() != emitted[0]:
        raise ValueError("expected one fresh specialization with the requested DMA configuration")
    original = sections((args.original_dump / "binary.o").read_bytes())[".text"]
    actual = sections((args.dump / "binary.o").read_bytes())[".text"]
    if args.mode == "base" and original != actual:
        raise ValueError("baseline rebuild changed the qualified device instructions")
    record = dict(case=args.case, mode=args.mode, cpp=sha256(emitted[0].encode()).hexdigest(),
                  text=sha256(actual).hexdigest(), original_text=sha256(original).hexdigest(),
                  text_bytes=len(actual))
    (args.dump / "dma-pad.json").write_text(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
