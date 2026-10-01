#!/usr/bin/env python3
"""Install a dedicated GitHub Actions runner on the CANN VM.

Only the cannbench-vm label is registered, avoiding historical arm64 jobs.
Requires VM-local gh authentication. Registration credentials are never printed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.request import urlopen


def gh_json(route, method="GET"):
    result = subprocess.run(["gh", "api", "--method", method, route], capture_output=True)
    if result.returncode:
        raise RuntimeError("GitHub API request failed")
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repo", default="aloschilov/pyasc-skill-stack")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    destination = args.directory.resolve()
    if destination.exists():
        raise SystemExit("Runner directory already exists; inspect it before any continuation")
    release = gh_json("repos/actions/runner/releases/latest")
    assets = [a for a in release["assets"] if a["name"].startswith("actions-runner-linux-arm64-")]
    if len(assets) != 1:
        raise RuntimeError("Expected exactly one official Linux ARM64 runner release")
    asset = assets[0]
    expected = asset.get("digest", "").removeprefix("sha256:")
    if len(expected) != 64 or not asset["browser_download_url"].startswith("https://github.com/actions/runner/releases/download/"):
        raise RuntimeError("Official asset digest or download origin unavailable")
    destination.mkdir(parents=True)
    archive = destination / asset["name"]
    digest = hashlib.sha256()
    with urlopen(asset["browser_download_url"], timeout=120) as source, archive.open("xb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError("Runner asset SHA256 mismatch")
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(destination)], check=True)
    token = gh_json(f"repos/{args.repo}/actions/runners/registration-token", "POST")["token"]
    result = subprocess.run([str(destination / "config.sh"), "--unattended", "--url", "https://github.com/" + args.repo,
        "--token", token, "--name", "pyasc-cannbench-vm", "--no-default-labels", "--labels", "cannbench-vm", "--work", "_work"],
        cwd=destination, capture_output=True)
    if result.returncode:
        raise RuntimeError("Runner registration failed; inspect the VM-local diagnostics")
    data = {"release": release["tag_name"], "asset_sha256": expected, "name": "pyasc-cannbench-vm", "labels": ["cannbench-vm"]}
    if args.start:
        env = {**os.environ, "PATH": str(Path.home() / ".bun/bin") + ":" + os.environ["PATH"]}
        with (destination / "runner.log").open("ab") as log:
            process = subprocess.Popen([str(destination / "run.sh")], cwd=destination,
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        data["pid"] = process.pid
    (destination / "INSTALL.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
