"""Package the exact locally qualified catalog with the measured runtime wheel."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from generation_gate import sha256, validate_generation


def package_bundle(catalog, run_root: Path, repo_root: Path, destination: Path):
    records = validate_generation(catalog, run_root, repo_root / "skills")
    runtime = json.loads((repo_root / "integrations/cannbench/ci/runtime.json").read_text())
    runtime_path = Path(runtime["vm_runtime_wheel"])
    pybind_path = Path(runtime["vm_pybind11_wheel"])
    if sha256(runtime_path) != runtime["runtime_wheel_sha256"] or sha256(pybind_path) != runtime["pybind11_sha256"]:
        raise ValueError("Pinned runtime wheel hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    contents = {}
    with zipfile.ZipFile(runtime_path) as archive:
        for name in archive.namelist():
            if name.startswith(("asc/", "asctile/")) and not name.endswith("/"):
                contents[name] = archive.read(name)
    with zipfile.ZipFile(pybind_path) as archive:
        for name in archive.namelist():
            if name.startswith("pybind11/") and not name.endswith("/"):
                contents[name] = archive.read(name)
            if name.endswith(".dist-info/LICENSE"):
                contents["cann_bench-1.1.2.dist-info/licenses/PYBIND11-LICENSE"] = archive.read(name)
    if "asctile/__init__.py" not in contents or not any(n.startswith("asc/_C/libpyasc") and n.endswith(".so") for n in contents):
        raise ValueError("Runtime lacks the public tile API or native compiler")
    contents["cann_bench-1.1.2.dist-info/licenses/PYASC-LICENSE"] = (repo_root / "integrations/cannbench/submission/vendor/PYASC-LICENSE").read_bytes()
    for path in (run_root / "locally_qualified/cann_bench").glob("*.py"):
        contents["cann_bench/" + path.name] = path.read_bytes()
    metadata = "cann_bench-1.1.2.dist-info/"
    contents[metadata + "METADATA"] = ("Metadata-Version: 2.1\nName: cann_bench\nVersion: 1.1.2\nRequires-Python: >=3.12,<3.13\n"
            + "X-PyAsc-Source-Commit: " + runtime["pyasc_commit"] + "\n\n").encode()
    contents[metadata + "WHEEL"] = b"Wheel-Version: 1.0\nGenerator: pyasc-cannbench-ci\nRoot-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n"
    records_csv = io.StringIO()
    writer = csv.writer(records_csv, lineterminator="\n")
    for name, data in sorted(contents.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((metadata + "RECORD", "", ""))
    contents[metadata + "RECORD"] = records_csv.getvalue().encode()
    dist = destination / "dist"; dist.mkdir()
    wheel = dist / "cann_bench-1.1.2-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            archive.writestr(name, data)
    digest = sha256(wheel)
    manifest = {"benchmark_slug": catalog["benchmark_slug"], "benchmark_version": catalog["benchmark_version"],
                "catalog_sha256": catalog["catalog_sha256"], "runtime": runtime,
                "wheel_sha256": digest, "operators": records}
    manifest["generation_sources"] = {str(p.relative_to(repo_root)): sha256(p)
        for p in sorted((repo_root / "skills").rglob("*")) if p.is_file()}
    for name in ("driver.py", "prompts.py", "contracts.py", "local_compile_gate.py"):
        relative = "integrations/cannbench/workers/" + name
        manifest["generation_sources"][relative] = sha256(repo_root / relative)
    (destination / "BUNDLE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "build.sh").write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'
            + "printf '%s  %s\\n' '" + digest + "' 'dist/" + wheel.name + "' | sha256sum --check\n")
    (destination / "build.sh").chmod(0o755)
    return manifest, wheel
