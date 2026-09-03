#!/usr/bin/env python3
"""Assemble immutable two-operator submission bundles for three variants."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
BASE = REPO_ROOT / "integrations/cannbench/submission"
VARIANTS = ("handwritten", "no_skills", "with_skills")
OPS = ("gelu", "foreach_addcdiv_scalar")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_base(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        BASE, destination,
        ignore=shutil.ignore_patterns(
            "dist", "build", ".base-dist", "*.egg-info", "__pycache__"))


def zip_tree(source: Path, output: Path) -> None:
    output.unlink(missing_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def main() -> int:
    validation = json.loads(
        (ROOT / "local_validation_summary.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "benchmark_slug": "official-tasks",
        "benchmark_version_observed": "1.1.1",
        "selected_operators": list(OPS),
        "intersection": {
            "gelu": "python/test/asc2/target/test_gelu.py",
            "foreach_addcdiv_scalar": "python/test/asc2/target/test_addcdiv.py",
        },
        "pyasc_target_ref": "v2",
        "pyasc_target_commit": "4d1db41d61cabf565bca1cfb0b11ef5ec4f84c7f",
        "submission_runtime_commit": "ac1222a48c8914d3f81297c7570d1a84f0f26778",
        "variants": {},
    }
    for variant in VARIANTS:
        destination = ROOT / "submissions" / variant
        copy_base(destination)
        candidates = {}
        for op in OPS:
            source = ROOT / variant / "candidates" / f"{op}.py"
            target = destination / "cann_bench" / f"{op}.py"
            shutil.copy2(source, target)
            candidates[op] = {
                "path": str(source.relative_to(REPO_ROOT)),
                "sha256": sha256(source),
            }
        variant_validation = [
            row for row in validation["results"] if row["variant"] == variant
        ]
        variant_meta = {
            "variant": variant,
            "selected_operators": list(OPS),
            "candidates": candidates,
            "local_validation": variant_validation,
        }
        if variant == "handwritten":
            variant_meta["provenance"] = json.loads(
                (ROOT / variant / "provenance.json").read_text(encoding="utf-8"))
        elif variant == "no_skills":
            variant_meta["provenance"] = {
                op: json.loads((ROOT / variant / op / "provenance.json").read_text(encoding="utf-8"))
                for op in OPS
            }
        else:
            variant_meta["provenance"] = {
                op: json.loads((ROOT / variant / "provenance" / f"{op}.json").read_text(encoding="utf-8"))
                for op in OPS
            }
        (destination / "COMPARISON_PROVENANCE.json").write_text(
            json.dumps(variant_meta, indent=2) + "\n", encoding="utf-8")
        archive = ROOT / "submissions" / f"{variant}.zip"
        zip_tree(destination, archive)
        variant_meta["submission_zip"] = {
            "path": str(archive.relative_to(REPO_ROOT)),
            "sha256": sha256(archive),
            "size_bytes": archive.stat().st_size,
        }
        manifest["variants"][variant] = variant_meta
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        name: data["submission_zip"]
        for name, data in manifest["variants"].items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
