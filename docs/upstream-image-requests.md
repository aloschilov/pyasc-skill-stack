# Requests to the image-build session

This repository consumes simulator images; it does not build them. Which
images exist, and what they are qualified for, is owned by the session that
works on `compiler-team-ru/pyasc-images`. Keeping that boundary is deliberate:
this repo pins one digest and verifies against it, and nothing here should
start making image-build decisions by implication.

This file is the standing ask to that session, version-controlled so the
request and its answers stay on the record rather than living in a chat.

Everything below the rule is the prompt. It is written to be self-contained —
the reader needs the `pyasc-images` repository, not this one.

**Status: answered and acted on (2026-10-26).** All five requests were
delivered: a dev overlay built on the pinned base, a stated rebuild cadence, a
stable qualification-report URL, the restored `latest` pointer, and an answer
on test scope. What the answers mean for this repository is recorded in
[`evidence-sources.md`](evidence-sources.md), which is the document to read
first; this one is kept as the record of what was asked. Every state claim in
the prompt below — case counts, `suite_status`, which overlays existed — is a
snapshot from the day it was sent and has since moved. Read it as a question
that was asked, never as a description of upstream today.

---

## Context

`pyasc-skill-stack` consumes exactly one cell of your published matrix, pinned
by digest, and compile-checks the vendored upstream AscTile target suite
inside it. Its current pin:

| | |
|---|---|
| base matrix release | `matrix-1.1.1.dev20261024081924-d162e39cbacc2dc9` |
| cell | `u22.04-c9.2.0-p3.11`, both architectures |
| manifest digest | `sha256:6256a47b167ba05b2b7b2280ab0460e1da42a188bcec2c643f6a985718fd4b67` |
| upstream source | `d0f64a63c432761aa57648e67be7582780a17561` |

Five requests follow. Items 1 and 4 are blocking in practice; 2, 3 and 5 are
policy questions whose answers decide how much machinery the consumer builds.

## 1. A dev overlay built on the base we pin

`dev-matrix.json` currently exists only for base
`matrix-1.1.1.dev20261023173239-d162e39cbacc2dc9`. The newest dev release is
`dev-matrix-1.1.1.dev20261023173239-d162e39cbacc2dc9-35ee3787583d`, whose
`base_report` is that 10-23 matrix. There is a published base matrix for
`1.1.1.dev20261024081924`, but no dev overlay against it.

So the overlay line and the pin are one base apart, and the consumer cannot
adopt `pyasc-sim-dev` without either moving its pin backwards or knowingly
running a `d0f64a63` test suite against an `ea2dbf9d` runtime.

**Request:** run `dev-images.yaml` with `report` set to
`matrix-1.1.1.dev20261024081924-d162e39cbacc2dc9`, at minimum for cells
`u22.04-c9.2.0-p3.11-amd64` and `u22.04-c9.2.0-p3.11-arm64`.

For sizing: between `ea2dbf9d` and `d0f64a63` exactly one file in
`experimental/asctile/python/test/target/` changes
(`test_fused_infer_attention_score.py`); the other 24 are byte-identical. So
this is a cheap rebuild, and afterwards the consumer's pin and your overlay
line agree without anyone having to reason about the gap.

## 2. Overlay cadence

The README says `dev-images.yaml` "discovers completed base matrices hourly",
but the 10-24 base has had none built for it, so discovery either does not
cover it or has not reached it.

**Question:** is the intended steady state one dev overlay for every published
base matrix, or overlays only on explicit request? The consumer's sync script
behaves differently in each case — it either follows the dev line
automatically, or treats a matching overlay as something to wait for and fall
back from. Please answer the policy, not just this instance.

## 3. Machine-readable suite status

The dev report records `suite.suite_status: pending`, `executed: 0`, and
per-arch `suite_status` on each manifest entry. The README points at a durable
`qualification-<recipe>.json` for actual results, but no asset by that name
exists in any release yet.

**Questions:**

- What is the stable URL for `qualification-<recipe>.json`, and which
  `<recipe>` does it key on — the overlay recipe (`35ee3787583d...`) or the
  base recipe (`d162e39cbacc2dc9`)?
- What is its schema, specifically how per-module and per-case outcomes are
  represented?
- Does it distinguish a module that *failed* from one that could not be run —
  resource exhaustion, timeout, kill signal? The consumer's gate vocabulary is
  `passed | failed | blocked | not-run` and it keeps `blocked` strictly apart
  from `failed`, because a runner that cannot hold a 7.4 GiB input tensor says
  nothing about the kernel. If your report collapses those two, we have to
  re-derive the distinction from logs, which is fragile.

Being able to read suite status programmatically is what lets a consumer cite
your qualification instead of guessing from `published`.

## 4. Bug: `dev-tests-*` releases have taken the `latest` pointer

The dev test-result releases are being published as normal (non-prerelease,
non-draft) releases, so GitHub now treats one of them as the repository's
latest release. The latest release is currently
`dev-tests-7599aec3a59eadc83ba4e15e2dd5cda85c707f042f0fa28612ef611164a84235-f731ab20ccbd`,
which carries only `tests-*.tar.gz` assets.

That breaks the documented stable entry point to the matrix report:

```
$ curl -sIL -o /dev/null -w '%{http_code} %{url_effective}\n' \
    https://github.com/compiler-team-ru/pyasc-images/releases/latest/download/matrix.json
404 https://github.com/compiler-team-ru/pyasc-images/releases/download/dev-tests-7599aec.../matrix.json
```

Any consumer that resolves `releases/latest/download/matrix.json` — which the
report URL recorded in published matrices itself points at — now gets a 404.

**Request:** either mark `dev-tests-*` (and probably `dev-matrix-*`) releases
as prereleases so they do not claim the `latest` pointer, or publish a stable
alias that always resolves to the newest *base* matrix report. Please say
which, since the consumer's fallback differs: without a stable pointer it has
to enumerate releases and filter by tag prefix, which couples it to your tag
naming.

## 5. Are `unit/` and `operations/` meant to be consumed?

Your collection check covers the whole upstream test tree. From the dev
report's inventory, at `ea2dbf9d`:

| Path | Cases |
|---|---|
| `experimental/asctile/python/test/unit` | 1520 |
| `experimental/asctile/python/test/operations` | 1435 |
| `experimental/asctile/python/test/target` (incl. `matmul_v3/`, `kimi_k3/`) | 1155 |
| `experimental/asctile/python/test/kernels` (incl. `kimi_k3/`) | 183 |
| total | 4293 |

The consumer currently vendors and gates only `target/`, on the understanding
that it is the upstream-authored correctness reference.

**Question:** is that the right subset to treat as the correctness contract,
or are `unit/` and `operations/` also intended as consumable gates? If they
are, is their pass/fail expected to be clean on the simulator backend, or do
they carry hardware or platform assumptions that make them unsuitable for a
camodel-only consumer? We would rather not vendor 2955 more cases and
discover that half of them were never meant to go green here.

## What we are not asking for

No changes to the minimal images, no source bundled into the dev images, and
no relaxation of the publication criteria. The current separation — minimal
images stay minimal, `published` means environment-qualified, full-suite
status lives in a separate report — is exactly the shape a consumer wants,
because it keeps us from quoting an environment check as if it were a suite
pass. Please keep it.
