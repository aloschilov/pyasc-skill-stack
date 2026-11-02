# Execution environment and CANNBot integration

Installing PyASC skills installs agent instructions, not CANN or a Python
runtime. OpenCode runs the skills in the execution environment selected by the
user. A checkout supplies the verifier and runtime pins as well as the skills;
a skills-only installation consumes an already provisioned environment.

## Reuse CANNBot's environment procedure

The intended CANNBot integration is a PyASC-specific step after its existing
CANN environment check, rather than another CANN installer or image matrix.
This handoff is the integration contract; PyASC registration in CANNBot's
installer has not been implemented or qualified in this PR.

CANNBot's [installation guide](https://gitcode.com/cann/cannbot-skills/blob/03cbbb9976901654d1fb27edb923bcd1b771f9bb/docs/installation-guide.md)
separates full plugin installation (skills, agents, workflows and external
repositories) from individual skill installation. Follow that guide to install
CANNBot; installing its skills alone does not establish runtime readiness.
Its [ascendc-env-check](https://gitcode.com/cann/cannbot-skills/tree/03cbbb9976901654d1fb27edb923bcd1b771f9bb/ops/ascendc-env-check)
provides `scripts/check_env.sh` for CANN paths, Toolkit/Ops and dependencies,
and `scripts/get_npu_arch.py --json` for actual device architecture. Source the
selected CANN `set_env.sh` in the same shell before using that environment.

The handoff to PyASC should be:

1. **Existing environment.** Reuse CANNBot's CANN check in the selected container
   or host environment. Retain Toolkit/Ops versions, `ASCEND_HOME_PATH`, host
   architecture and, for NPU work, the reported full SoC and device visibility.
   A CANN environment check does not demonstrate PyASC compilation or execution.
2. **PyASC runtime.** Select one Python executable and a compatible, pinned PyASC
   wheel for its Python ABI, CPU architecture and CANN installation. Record the
   wheel SHA256 and source revision. Reuse a matching installation; otherwise
   install the selected wheel into an isolated environment. Do not upgrade CANN,
   replace drivers or select an unpinned latest wheel as part of loading a skill.
3. **PyASC qualification.** Verify imports and the public tile API, then compile
   and numerically check a small kernel with that same interpreter. Report
   `blocked` if a dependency is missing; do not fall back from NPU to simulator
   and report hardware success.
4. **Agent execution.** Run OpenCode with the installed PyASC skills and that
   environment. One configured model suffices. Record the actual model and
   runtime identities with each generated candidate and measured result.

For a compatible wheel using the repository's current module layout:

```bash
export PYTHON=/absolute/path/to/venv/bin/python
"$PYTHON" -c 'import asc; from asc.experimental import asctile; print(asc.__file__, asctile.__file__); assert callable(asctile.jit)'
"$PYTHON" tests/tools/run_and_verify.py /absolute/path/to/kernel.py \
  --runtime-backend host --mode execute --backend NPU --platform "$FULL_SOC" --json
```

`FULL_SOC` is the runtime's actual supported SoC, not a machine-name guess.
Historical runtimes may expose top-level `asctile`; use their matching API and
record that runtime rather than rewriting imports silently. `host` means the
current interpreter, including inside a container. Use that interpreter's
`-m pip` and `-m pytest`, not an assumed global `python3.10` installation.

## Repository CI and external evaluation

For development without an NPU, use the digest-pinned simulator image in
[ENV.json](../reference/images/ENV.json) and the [README workflow](../README.md).
This is the reproducible repository CI environment, not an additional CANNBot
installation requirement. Upstream wheel/image availability does not qualify
all architecture, CANN and SoC combinations; see [verification status](verification-status.md).

For external hardware evaluation, [CANNBench](../integrations/cannbench/README.md)
runs the submitted package in its own runner environment. The submitter needs
service access and a compatible packaged runtime, not a local NPU or CANN
installation. Record the service's actual environment and exact package hash;
a local simulator pass is a separate claim.
