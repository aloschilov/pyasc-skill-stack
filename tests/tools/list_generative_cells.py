#!/usr/bin/env python3
"""Emit generative nightly matrix cells from capabilities.yaml."""

from __future__ import annotations

import argparse
import json
import sys

from load_capability_cells import list_generative_cells, load_capabilities_yaml


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="List capability cells with prompts for nightly generative runs",
    )
    ap.add_argument("--platform", default="Ascend950PR_9599")
    ap.add_argument("--format", choices=("shell", "json"), default="shell")
    args = ap.parse_args(argv)

    cap = load_capabilities_yaml()
    rows = list_generative_cells(cap, platform=args.platform)

    if args.format == "json":
        payload = [{"op": op, "dtype": dtype, "tier": tier, "tier_level": level}
                   for op, dtype, tier, level in rows]
        print(json.dumps(payload, indent=2))
        return 0

    for op, dtype, _tier, _level in rows:
        print(f"{op}:{dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
