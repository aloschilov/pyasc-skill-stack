# Reference material

Material in this directory is **upstream-authored**. It is the standard other
work is judged against, so nothing written by a model belongs here.

## `asctile/`

A pinned snapshot of the PyAsc AscTile target test suite — 24 test modules,
headed `Copyright (c) 2026 Huawei Technologies Co., Ltd.` under CANN Open
Software License 2.0. The collected case count moves with the pin; derive it
with `--collect-only` rather than trusting a number written down here.

The layout mirrors upstream exactly:

```
reference/asctile/conftest.py       <- upstream experimental/asctile/python/test/conftest.py
reference/asctile/target/           <- upstream experimental/asctile/python/test/target/
```

That is deliberate, and not merely tidiness. The suite is a package and uses
package-relative imports throughout: every module does `from .helpers import
...`, and the `matmul_v3/` subpackage reaches a level further up with
`from ..helpers import parametrize_is_static`. Flattening the snapshot,
dropping the subdirectories, or renaming any directory to something containing
a hyphen breaks all of it.

### Which version is this?

See [`asctile/PIN.json`](asctile/PIN.json). The pin is managed by
[`../scripts/sync-target-suite.sh`](../scripts/sync-target-suite.sh), which
defaults to the source commit of the PyAsc runtime you currently have
installed. Check for drift with
[`../scripts/check-target-pin.sh`](../scripts/check-target-pin.sh).

Target tests move between PyAsc versions — the upstream path has been
`python/test/asc2/target/`, then `python/test/asctile/target/`, and the latter
was deleted outright when AscTile was relocated under `experimental/`. Treat
this snapshot as a pin to refresh, never as a permanent fixture.

That is why the sync finds the suite by **content** — files using both
`asctile.jit` and `asctile.global_tensor` — rather than by a remembered path.
Matching on path or filename would be wrong twice over: the path has already
moved, and `test_vadd.py` exists three times in a typical checkout, the other
two being low-level `asc.jit` kernels that would import cleanly and verify
entirely the wrong thing.

A real sync shows the mechanism working. Moving the pin to
`d0f64a63` located the suite at its **current** path under `experimental/`,
rather than the `python/test/asctile/target` the previous pin recorded, and
none of the `test_vadd.py` decoys were picked up:

```
$ scripts/sync-target-suite.sh --repo https://github.com/compiler-team-ru/pyasc.git \
    --ref d0f64a63c432761aa57648e67be7582780a17561 --dry-run
[sync] suite: experimental/asctile/python/test/target (22 files at the top level)
[sync] suite includes 7 file(s) in subpackages
  added     kimi_k3/test_attn_residual.py
  added     test_gather.py
  ...
  removed   test_add_inplace.py
  removed   test_matmul_v3.py
```

Note what moved: `test_matmul_v3*.py` were top-level modules and are now the
`matmul_v3/` subpackage, and `test_add_inplace.py` is gone upstream entirely.
Both the sync and the drift check therefore work on suite-relative paths and
recurse, instead of globbing `target/*.py`.

### Local modifications

Two, both confined to `asctile/conftest.py`. The test files themselves are
kept **byte-identical to upstream** so the sync script can compare them by
hash.

1. An alias registering `asc.experimental.asctile` as the top-level `asctile`
   module, so that the suite's bare `import asctile` resolves against a wheel
   install. It raises a diagnostic `RuntimeError` when no usable runtime is
   present, rather than letting all 24 modules die with
   `AttributeError: module 'asctile' has no attribute 'jit'`.
2. `pytest_addoption` registers each option defensively, tolerating a
   `ValueError` from an already-registered name. `pyasc-dev` declares no entry
   points and so ships no pytest plugin, which means `--backend` and
   `--platform` exist only because this file adds them; registering
   defensively also keeps working if a future runtime does start providing
   them.

Everything else in `conftest.py` — the `asctile_xfail` marker and its
`apply_xfail` fixture, `pytest_make_parametrize_id`, the profiling summary — is
upstream's and must stay. `test_rms_norm.py` marks five UB-overflow cases with
`xfail(...)`; without the fixture those become hard failures.

### Running it

Needs `pytest` and `torch`, neither of which `pyasc-dev` depends on:

```bash
pip install --pre pyasc-dev pytest torch
pytest reference/asctile/target --collect-only -q    # no CANN needed; 1155 at d0f64a63
```

Executing rather than collecting additionally needs CANN and the simulator
libraries on `LD_LIBRARY_PATH`.
