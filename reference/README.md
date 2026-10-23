# Reference material

Material in this directory is **upstream-authored**. It is the standard other
work is judged against, so nothing written by a model belongs here.

## `asctile/`

A pinned snapshot of the PyAsc AscTile target test suite — 18 test modules,
891 collected cases, headed `Copyright (c) 2026 Huawei Technologies Co., Ltd.`
under CANN Open Software License 2.0.

The layout mirrors upstream `python/test/asctile/` exactly:

```
reference/asctile/conftest.py       <- upstream python/test/asctile/conftest.py
reference/asctile/target/           <- upstream python/test/asctile/target/
```

That is deliberate, and not merely tidiness. `target/test_matmul_v3_l1_transpose.py`
contains `from ..target.test_matmul_v3 import FullLoadMode`, a package-relative
import that only resolves if the parent package is preserved. Flattening the
snapshot, or renaming the directory to anything containing a hyphen, breaks it.

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

A dry run against a local checkout confirms both halves:

```
$ scripts/sync-target-suite.sh --source ~/workspace/pyasc-fork --dry-run
[sync] suite: experimental/asctile/python/test/target (23 files)
  added     test_gather.py
  added     test_rms_norm.py
  ...
  removed   test_matmul_k_tiled.py
```

It located the suite at its **current** path under `experimental/`, not the one
recorded in the pin, and the decoys did not appear. The pending drift is real:
nine tests added upstream, three no longer in `target/`.

### Local modifications

Exactly one, and it is confined to `asctile/conftest.py`: an alias registering
`asc.experimental.asctile` as the top-level `asctile` module. The test files
themselves are kept **byte-identical to upstream** so the sync script can
compare them by hash.

### Running it

Needs `pytest` and `torch`, neither of which `pyasc-dev` depends on:

```bash
pip install --pre pyasc-dev pytest torch
pytest reference/asctile/target --collect-only -q    # 891 tests, no CANN needed
```

Executing rather than collecting additionally needs CANN and the simulator
libraries on `LD_LIBRARY_PATH`.
