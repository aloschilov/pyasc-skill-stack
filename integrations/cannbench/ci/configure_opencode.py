#!/usr/bin/env python3
"""Configure GitHub GLM-5.3 credentials from OpenCode on the CANN VM.

Never displays the key or saves it in a checkout. --apply explicitly writes
the encrypted GitHub secret and the endpoint variable; default is inspection.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path.home() / ".config/opencode/opencode.json")
    parser.add_argument("--repo", default="aloschilov/pyasc-skill-stack")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    gateway = json.loads(args.config.read_text())["provider"]["gateway"]
    if "glm-5.3" not in gateway.get("models", {}):
        raise SystemExit("OpenCode gateway does not declare glm-5.3")
    options = gateway["options"]
    key = options.get("apiKey", "")
    match = re.fullmatch(r"\{env:([^}]+)\}|\$\{([^}]+)\}", key)
    if match:
        key = os.environ.get(match[1] or match[2], "")
    file_match = re.fullmatch(r"\{file:([^}]+)\}", key)
    if file_match:
        key = Path(file_match[1]).expanduser().read_text().strip()
    if not key or key.startswith("{"):
        raise SystemExit("OpenCode gateway key could not be resolved")
    endpoint = options["baseURL"]
    if not endpoint.startswith("https://"):
        raise SystemExit("Gateway endpoint must use HTTPS")
    if args.apply:
        subprocess.run(["gh", "secret", "set", "OPENCODE_API_KEY", "--repo", args.repo],
                       input=key, text=True, check=True, capture_output=True)
        route = f"repos/{args.repo}/actions/variables"
        existing = subprocess.run(["gh", "api", route + "/OPENCODE_BASE_URL"], capture_output=True)
        if existing.returncode == 0:
            command = ["gh", "api", "--method", "PATCH", route + "/OPENCODE_BASE_URL"]
        else:
            # A missing variable is a 404; authentication/network errors must
            # not be interpreted as permission to issue a write.
            if b"404" not in existing.stderr:
                raise SystemExit("Cannot inspect the GitHub endpoint variable")
            command = ["gh", "api", "--method", "POST", route]
        subprocess.run(command + ["-f", "name=OPENCODE_BASE_URL", "-f", "value=" + endpoint],
                       check=True, capture_output=True)
    print(json.dumps({"model": "gateway/glm-5.3", "credential_present": True,
                      "secret_name": "OPENCODE_API_KEY", "endpoint_variable": "OPENCODE_BASE_URL",
                      "applied": args.apply}))


if __name__ == "__main__":
    main()
