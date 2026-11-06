# Local AscTile kernel report (template)

- Operator / contract:
- Target: SoC, CANN version, PyAsc version/commit, torch version, model/agent
- Candidate: file, sha256
- Reference: torch version, tolerance, seed
- Compile: status + evidence path
- Numerical: status, mismatch count, MARE/max-diff, evidence path
- Optimization: budget (time/attempts), baseline candidate, best correct
  candidate, measured improvement (only if measured)
- Hardware: status (`passed | failed | blocked | not-run`) + evidence path
- Honest result: `passed | failed | blocked | not-run`

Record the PyAsc source commit, not just the version string. For a published
wheel:

```bash
python -c "import importlib.metadata as m,re; d=m.metadata('pyasc-dev')['Description']; print(re.search(r'Source: .([0-9a-f]{40}).', d).group(1))"
```

Every gate gets its own status. Leaving a gate out is not the same as passing
it; say `not-run` and why.
