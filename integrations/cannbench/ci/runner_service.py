#!/usr/bin/env python3
"""Persist the dedicated runner as a user service after checking it is idle."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    install = json.loads((directory / "INSTALL.json").read_text())
    if install["name"] != "pyasc-cannbench-vm" or install["labels"] != ["cannbench-vm"]:
        raise ValueError("Runner is not the dedicated installation")
    payload = subprocess.run(["gh", "api", "repos/aloschilov/pyasc-skill-stack/actions/runners"], capture_output=True, check=True, text=True)
    matches = [r for r in json.loads(payload.stdout)["runners"] if r["name"] == install["name"]]
    if len(matches) != 1 or matches[0]["busy"]:
        raise ValueError("Dedicated runner must exist and be idle")
    unit = Path.home() / ".config/systemd/user/pyasc-cannbench-runner.service"
    content = ("[Unit]\nDescription=Dedicated pyasc CANNBench GitHub runner\nAfter=network-online.target\n\n"
               "[Service]\nType=simple\nWorkingDirectory=" + str(directory) + "\nExecStart=" + str(directory / "run.sh")
               + "\nEnvironment=PATH=" + str(Path.home() / ".bun/bin") + ":/usr/local/bin:/usr/bin:/bin\nRestart=on-failure\nRestartSec=10\n\n[Install]\nWantedBy=default.target\n")
    if unit.exists() and unit.read_text() != content:
        raise ValueError("Existing user service differs; inspect it before changing")
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(content)
    if subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit.name]).returncode == 0:
        print("Dedicated runner user service is already active")
        return
    pid = install.get("pid")
    if pid and Path(f"/proc/{pid}/cmdline").exists():
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        if str(directory / "run.sh") not in command or os.getpgid(pid) != pid:
            raise ValueError("Recorded process no longer belongs to this runner")
        os.killpg(pid, signal.SIGTERM)
        for _ in range(20):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.5)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", unit.name], check=True)
    subprocess.run(["systemctl", "--user", "is-active", unit.name], check=True)
    print("Dedicated runner user service enabled; enable loginctl linger for boot persistence")


if __name__ == "__main__":
    main()
