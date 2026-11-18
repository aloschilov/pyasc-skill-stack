#!/usr/bin/env bash
# Networked packaging smoke test, separate from the offline PR gate.
set -euo pipefail
stack=$(cd "$(dirname "$0")/../.." && pwd)
project=$(mktemp -d)
trap 'rm -rf "$project"' EXIT
cd "$project"
export DISABLE_TELEMETRY=1 NO_COLOR=1
cli=(npx --yes skills@1.9.0)
"${cli[@]}" add "$stack" --list > list.log

for manifest in "$stack"/skills/*/SKILL.md; do
    name=$(basename "$(dirname "$manifest")")
    grep -Fwq -- "$name" list.log || { echo "Not discovered: $name" >&2; exit 1; }
done
test ! -e .agents
"${cli[@]}" add "$stack" -a opencode --skill '*' -y

# Check the distributed files, including resources used by sibling skills.
while IFS= read -r file; do
    cmp "$stack/$file" ".agents/$file"
done < <(git -C "$stack" ls-files skills)

# Parent-relative Markdown links previously resolved into the original checkout.
python3 - <<'CHECK_LINKS'
import re
from pathlib import Path

for doc in Path(".agents/skills").rglob("*.md"):
    for target in re.findall(r"\]\((\.\./[^# )]+)(?:#[^ )]*)?\)", doc.read_text()):
        if not (doc.parent / target).exists():
            raise SystemExit(f"{doc}: missing installed resource {target}")
CHECK_LINKS

workflow=.agents/skills/asctile-codegen-workflow
bash "$workflow/scripts/init_kernel_project.sh" example
cp "$workflow/templates/kernel-template.py" example/kernel.py
python3 -m py_compile example/kernel.py
test -f example/conftest.py
echo "PASS: discovered the stack, installed for OpenCode, and scaffolded a kernel."
