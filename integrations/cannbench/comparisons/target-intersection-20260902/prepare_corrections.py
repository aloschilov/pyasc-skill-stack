#!/usr/bin/env python3
"""Assemble immutable corrective bundles without rewriting the first run."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
BASE = REPO_ROOT / "integrations/cannbench/submission"
OPS = ("gelu", "foreach_addcdiv_scalar")
VARIANTS = ("target_corrected", "with_skills_corrected")

SOURCES = {
    "target_corrected": {
        "gelu": BASE / "cann_bench/gelu.py",
        "foreach_addcdiv_scalar": BASE / "cann_bench/foreach_addcdiv_scalar.py",
    },
    "with_skills_corrected": {
        "gelu": ROOT / "correction/with_skills/work/candidate.py",
        "foreach_addcdiv_scalar": (
            ROOT / "with_skills/candidates/foreach_addcdiv_scalar.py"
        ),
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_base(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        BASE,
        destination,
        ignore=shutil.ignore_patterns(
            "dist", "build", ".base-dist", "*.egg-info", "__pycache__"
        ),
    )


def zip_tree(source: Path, output: Path) -> None:
    output.unlink(missing_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def main() -> int:
    output_root = ROOT / "correction/submissions"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "benchmark_slug": "official-tasks",
        "selected_operators": list(OPS),
        "submission_runtime_commit": (
            "ac1222a48c8914d3f81297c7570d1a84f0f26778"
        ),
        "source_audit": {
            "first_run_source_repository": (
                "https://gitcode.com/compiler-team/pyasc"
            ),
            "first_run_source_commit": (
                "4d1db41d61cabf565bca1cfb0b11ef5ec4f84c7f"
            ),
            "first_run_source_directory": "python/test/asc2/target",
            "first_run_source_access": (
                "read from Git objects, not copied from the current worktree"
            ),
            "current_origin_v2_commit": (
                "030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d"
            ),
            "current_origin_v2_directory": "python/test/asctile/target",
        },
        "variants": {},
    }

    for variant in VARIANTS:
        destination = output_root / variant
        copy_base(destination)
        candidates = {}
        for op in OPS:
            source = SOURCES[variant][op]
            target = destination / "cann_bench" / f"{op}.py"
            shutil.copy2(source, target)
            candidates[op] = {
                "source": str(source.relative_to(REPO_ROOT)),
                "sha256": sha256(source),
            }

        provenance = {
            "variant": variant,
            "selected_operators": list(OPS),
            "candidates": candidates,
        }
        if variant == "target_corrected":
            provenance["method"] = (
                "manual CANNBench contract completion of the target-derived "
                "kernels: FP32 internal addcdiv and stable exact/tanh GELU"
            )
            provenance["classification"] = (
                "target-derived corrective adapter, not a verbatim target test"
            )
            provenance["prior_hardware_evidence"] = {
                "gelu": "job_cd51d6c2ca67: 20/20",
                "foreach_addcdiv_scalar": (
                    "job_75a7fee4ae6f and job_ae3bfdefd087: 20/20"
                ),
            }
        else:
            provenance["method"] = (
                "OpenCode repair with repository-local skill invocation and "
                "measured compile feedback"
            )
            provenance["worker_sessions"] = {
                "accepted_repair": "ses_f9c952882ffeAnIEP58bPaIy7l",
                "accepted_review": "ses_f9c8e48c5ffeFcOrr6Qvqf72Tj",
                "timed_out_review": "ses_f9c93e480ffeq5Hu4yhp750SLO",
            }
            provenance["required_skill"] = "pyasc-cannbench-kernel"
            provenance["skill_loaded_in_accepted_phases"] = True

        (destination / "COMPARISON_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        archive = output_root / f"{variant}.zip"
        zip_tree(destination, archive)
        provenance["submission_zip"] = {
            "path": str(archive.relative_to(REPO_ROOT)),
            "sha256": sha256(archive),
            "size_bytes": archive.stat().st_size,
        }
        manifest["variants"][variant] = provenance

    (ROOT / "correction/MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["variants"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
