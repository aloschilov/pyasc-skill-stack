"""Recover the device text from an ASC executable and compare to the wheel ELF."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def sections(data):
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected little-endian ELF64")
    shoff, = struct.unpack_from("<Q", data, 40)
    size, count, strings = struct.unpack_from("<HHH", data, 58)
    if size != 64 or not 0 < strings < count or shoff + size * count > len(data):
        raise ValueError("invalid ELF section table")
    headers = list(struct.iter_unpack("<IIQQQQIIQQ", data[shoff:shoff + size * count]))
    off, length = headers[strings][4:6]
    names = data[off:off + length]
    result = {}
    for name, kind, _, _, off, length, *_ in headers:
        if kind != 8:
            if off + length > len(data):
                raise ValueError("section outside file")
            result[names[name:names.index(0, name)].decode()] = data[off:off + length]
    return result


def device_elf(executable):
    payload = sections(executable)["__aicore_reloc_binary"]
    if payload.count(b"\x7fELF") != 1:
        raise ValueError("expected one packaged device image")
    image = payload[payload.index(b"\x7fELF"):]
    sections(image)
    return image


def collect(build, profiles, objdump, out):
    out.mkdir()
    report = {}
    for case in (11, 13):
        for mode in ("scope", "call", "wheel"):
            path = (profiles / f"c{case}w1r0/dump/binary.o" if mode == "wheel"
                    else build / f"{mode}{case}")
            raw = path.read_bytes()
            image = raw if mode == "wheel" else device_elf(raw)
            text = sections(image)[".text"]
            stem = f"{mode}{case}"
            binary = out / (stem + ".elf")
            binary.write_bytes(image)
            with (out / (stem + ".asm")).open("w") as stream:
                subprocess.run([objdump, "-d", str(binary)], stdout=stream, check=True)
            report[stem] = dict(file_sha256=hashlib.sha256(raw).hexdigest(),
                                text_sha256=hashlib.sha256(text).hexdigest(), text_bytes=len(text))
    (out / "digests.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("build", "profiles", "objdump", "out"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    collect(args.build, args.profiles, args.objdump, args.out)
