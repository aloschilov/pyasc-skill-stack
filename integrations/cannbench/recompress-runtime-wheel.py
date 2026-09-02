#!/usr/bin/env python3
"""Recompress the internal pyasc transport wheel with ZIP-LZMA.

The runtime wheel is unpacked by ``merge_wheels.py`` during the CANNBench
build; it is not installed directly.  ZIP-LZMA therefore reduces upload size
without changing the final standards-compatible submission wheel.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()

    wheel = args.wheel.resolve()
    with tempfile.NamedTemporaryFile(
        prefix=f".{wheel.name}.", suffix=".tmp", dir=wheel.parent, delete=False
    ) as temporary:
        output = Path(temporary.name)

    try:
        with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_LZMA, allowZip64=True
        ) as target:
            for source_info in source.infolist():
                target_info = zipfile.ZipInfo(
                    filename=source_info.filename,
                    date_time=source_info.date_time,
                )
                target_info.external_attr = source_info.external_attr
                target_info.internal_attr = source_info.internal_attr
                target_info.create_system = source_info.create_system
                target_info.comment = source_info.comment
                target_info.extra = source_info.extra
                target_info.compress_type = zipfile.ZIP_LZMA
                with source.open(source_info) as source_file, target.open(
                    target_info, "w", force_zip64=True
                ) as target_file:
                    shutil.copyfileobj(source_file, target_file, length=1024 * 1024)
        os.replace(output, wheel)
    finally:
        output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
