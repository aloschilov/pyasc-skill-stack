#!/usr/bin/env python3
"""Resolve a tracked opencode profile into a project-level ``opencode.json``.

This exists because two entry points need the same answer and used to get
different ones. ``collect_generative_evidence.py`` rendered a profile from
``integrations/opencode/profiles/``; ``tests/integration/test-kernel-generation.sh``
wrote no config at all and silently inherited the user's global
``~/.config/opencode/opencode.json``. On at least one developer machine that
global config caps ``gateway/glm-5.3`` at ``output: 8192`` with reasoning
enabled, which is the documented cause of the truncated generations in
``docs/verification-status.md``. A harness that inherits an unknown ceiling
is not measuring the thing it claims to measure, so both paths now render
the same tracked profile through this module.

Two placeholder syntaxes appear in the profile templates and they are
resolved at different times:

``${VAR}``
    Substituted here, from this process's environment, via
    :class:`string.Template`. Unset names are left verbatim.

``{env:VAR}``
    Left alone — opencode itself expands these when it loads the config.

Both forms share a failure mode worth naming: when ``VAR`` is unset the
placeholder degrades to an empty string somewhere downstream, and what
surfaces is an opaque authentication error from the provider rather than
"you forgot to export a key". :func:`missing_env` finds those names up
front so callers can say so plainly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO_ROOT / "integrations" / "opencode" / "profiles"

#: Used when a profile name resolves to nothing on disk. Grants the agent the
#: tool permissions a kernel-writing run needs and says nothing about models,
#: which means the caller's global config supplies them — acceptable as a
#: fallback, never as the configuration under test.
MINIMAL_OPENCODE_JSON: dict = {
    "$schema": "https://opencode.ai/config.json",
    "permission": {
        "read": "allow",
        "edit": "allow",
        "bash": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "skill": "allow",
        "task": "allow",
    },
}

# Braced forms only. The templates also contain the bare `$schema` key, which
# string.Template treats as a placeholder and leaves verbatim when unset; it
# must not be reported as a missing variable.
_ENV_PATTERNS = (
    re.compile(r"\$\{(\w+)\}"),
    re.compile(r"\{env:(\w+)\}"),
)


def missing_env(raw: str) -> list[str]:
    """Return the environment variable names *raw* references but lacks.

    Covers both the ``${VAR}`` and ``{env:VAR}`` spellings, deduplicated and
    sorted so the diagnostic is stable enough to assert against.
    """
    names: set[str] = set()
    for pattern in _ENV_PATTERNS:
        names.update(pattern.findall(raw))
    return sorted(n for n in names if not os.environ.get(n))


def profile_path(profile: str, override_path: str | None = None) -> Path:
    """Where a profile name resolves to on disk, whether or not it exists."""
    if override_path:
        return Path(override_path)
    return PROFILE_DIR / f"{profile}.json"


def available_profiles() -> list[str]:
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def resolve(
    profile: str, override_path: str | None = None,
) -> tuple[dict, Path | None, list[str]]:
    """Render a profile into a config dict.

    Returns the dict, the template it came from (``None`` for the minimal
    fallback), and the list of environment variables the template references
    but which are unset. A non-empty third element is not fatal here — some
    callers legitimately want the config anyway — but it is what
    :func:`require_env` turns into a hard stop.
    """
    path = profile_path(profile, override_path)
    if not path.exists():
        return dict(MINIMAL_OPENCODE_JSON), None, []

    raw = path.read_text()
    absent = missing_env(raw)
    expanded = Template(raw).safe_substitute(os.environ)
    try:
        data = json.loads(expanded)
    except json.JSONDecodeError as exc:
        print(
            f"  WARN: profile template {path} is not valid JSON after env "
            f"substitution ({exc}); falling back to the minimal config",
            file=sys.stderr,
        )
        return dict(MINIMAL_OPENCODE_JSON), None, absent
    return data, path, absent


def grant_repo_access(cfg: dict, extra_roots: tuple[str, ...] = ()) -> dict:
    """Let a deny-by-default profile read this checkout.

    The test projects are temporary directories whose ``skills`` entry is a
    symlink back into this repository, so from opencode's point of view the
    skill stack is an *external* directory. Profiles that sensibly close that
    door with ``{"*": "deny"}`` therefore also close it on the thing under
    test, and the agent runs with no skills while the harness reports
    ``skills_mode=on``. That is a silent wrong answer, not a crash.

    The alternative the repository used to carry was worse: ``cloud-default``
    hardcodes ``~/workspace/pyasc-skill-stack/*``, which is one developer's
    layout and grants nothing on any other machine. Resolving the root at
    render time works from any clone.

    An exact key already present in the template wins, so a profile can still
    deliberately deny a path this function would otherwise allow.
    """
    perms = cfg.get("permission")
    if not isinstance(perms, dict):
        return cfg
    existing = perms.get("external_directory")
    if not isinstance(existing, dict):
        return cfg

    granted: dict[str, str] = {}
    for root in (str(REPO_ROOT), *extra_roots):
        if not root:
            continue
        resolved = Path(root).expanduser()
        granted[f"{resolved}/*"] = "allow"
        granted[str(resolved)] = "allow"

    # Insertion order matters: the allows precede the template's catch-all
    # deny. update() keeps the position of keys that already exist, so an
    # explicit template entry overrides the grant without being reordered.
    granted.update(existing)
    perms["external_directory"] = granted
    return cfg


def build(
    profile: str,
    skills_mode: str,
    override_path: str | None = None,
    extra_roots: tuple[str, ...] = (),
) -> dict:
    """Render a profile for one (profile, skills) combination.

    ``skills_mode="on"`` points skill discovery at the project's ``./skills``
    tree; ``"off"`` strips the section so the run measures raw opencode with
    no skill-stack hints. The A/B between those two is the whole point of the
    generative evidence harness, so the difference must live here and not in
    the templates.
    """
    cfg, _, _ = resolve(profile, override_path)
    if skills_mode == "on":
        cfg["skills"] = {"paths": ["./skills"]}
    else:
        cfg.pop("skills", None)
    # Unconditional: the skills-off baseline still symlinks `golden/` out of
    # the project, so both arms of the A/B need the grant. Withholding it from
    # one arm would make the comparison measure filesystem permissions.
    return grant_repo_access(cfg, extra_roots)


def require_env(profile: str, absent: list[str], source: Path | None) -> None:
    """Exit non-zero with a readable diagnostic when credentials are missing.

    Deliberately not a warning. A cloud generation run that starts without a
    key burns several minutes before failing inside the provider with a
    message that names neither the profile nor the variable.
    """
    if not absent:
        return
    origin = source if source is not None else f"profile {profile!r}"
    print(
        f"ERROR: {origin} references environment variables that are not set:\n"
        + "".join(f"  - {name}\n" for name in absent)
        + "\nExport them and retry. For the gateway profiles this is the\n"
        "OpenAI-compatible endpoint and its key, e.g.\n"
        "  export OPENCODE_BASE_URL='https://<host>/v1'\n"
        "  export OPENCODE_API_KEY=\"$(cat ~/.config/opencode/gateway.key)\"\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render a tracked opencode profile to a project config.",
    )
    ap.add_argument(
        "--profile", default="cloud-glm-5.3",
        help="profile name under integrations/opencode/profiles "
             f"(available: {', '.join(available_profiles()) or 'none'})",
    )
    ap.add_argument(
        "--config", default=None,
        help="path to a profile template, overriding --profile",
    )
    ap.add_argument(
        "--skills", choices=("on", "off"), default="on",
        help="enable or strip ./skills discovery (default: on)",
    )
    ap.add_argument(
        "--out", default="-",
        help="write the rendered config here ('-' for stdout, the default)",
    )
    ap.add_argument(
        "--require-credentials", action="store_true",
        help="exit 1 if the template references unset environment variables",
    )
    ap.add_argument(
        "--allow-root", action="append", default=[], metavar="DIR",
        help="additionally grant external_directory access to DIR; this "
             "repository's root is always granted when --skills is on",
    )
    args = ap.parse_args(argv)

    cfg, source, absent = resolve(args.profile, args.config)
    if args.require_credentials:
        require_env(args.profile, absent, source)

    if source is None:
        print(
            f"  WARN: no profile template for {args.profile!r}; using the "
            "minimal config, so models come from your global opencode.json",
            file=sys.stderr,
        )

    if args.skills == "on":
        cfg["skills"] = {"paths": ["./skills"]}
    else:
        cfg.pop("skills", None)
    cfg = grant_repo_access(cfg, tuple(args.allow_root))

    text = json.dumps(cfg, indent=2) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text)
        print(f"  Wrote {args.out} from {source or 'minimal fallback'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
