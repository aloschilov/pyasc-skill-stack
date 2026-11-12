"""Print the pinned public contract and skill sections used for the second draw."""
import subprocess
import sys

SKILLS_REV = "463783ed5b31a80ff8b5a10913cbc07bb28a4d68"
PATTERNS = "skills/asctile-api-patterns/"
TASK = "integrations/cannbench/tasks/swi_glu/"
SECTIONS = (
    (TASK + "proto.yaml", "", ""),
    (TASK + "cases.yaml", "", ""),
    (PATTERNS + "SKILL.md", "### Choose a memory layout", "### Batch short rows"),
    (PATTERNS + "references/reindex-slabs.md", "# Row geometry", "## Reducing the index reads"),
    (PATTERNS + "references/strided-row-tiles.md", "# Height and unroll", "Exercise the selector"),
    (PATTERNS + "references/streamed-halves.md", "# Flat SwiGLU", "Compare consecutive chunks"),
    (PATTERNS + "templates/streamed-swiglu-flat.py", "", ""),
)
REQUEST = "docs/evidence/swiglu-vector-grid-20261110/clean-opencode/request.txt"


def pinned_file(checkout, path):
    return subprocess.check_output(["git", "-C", checkout, "show", f"{SKILLS_REV}:{path}"]).decode()


def prompt(checkout):
    parts = []
    for path, start, end in SECTIONS:
        body = pinned_file(checkout, path)
        if start:
            body = body[body.index(start):]
        if end:
            body = body[:body.index(end)]
        parts.extend((f"= {path}\n", body))
    return "".join(parts) + pinned_file(checkout, REQUEST)


if __name__ == "__main__":
    sys.stdout.write(prompt(sys.argv[1]))
