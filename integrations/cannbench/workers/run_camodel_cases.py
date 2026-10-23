#!/usr/bin/env python3
"""Full-matrix CaModel numerical runner for a generated CANNBench bundle.

Mirrors ``kernel_eval.eval.evaluator.evaluate_case`` from the cann-bench
repository on the Ascend950PR_9599 simulator:

1. Official ``DataGenerator`` inputs with a seed derived from the canonical
   catalog case ID and ``--seed`` (official eval_seed default: 0).
2. ``golden.get_input`` hook (if present) with proto-order tensor mapping,
   case attrs and ``skip2_exist=True`` default, exactly as the official
   evaluator invokes it.
3. Parameter construction: positional golden-signature mapping after
   ``get_input``, otherwise the official ``ParamBuilder``.
4. Golden reference on CPU with fp64-promoted floating inputs.
5. Native-dtype CPU golden (``cpu_out``) used by the small-value check.
6. Candidate execution with pyasc switched to ``Backend.Model`` /
   ``Platform.Ascend950PR_9599`` under the shared VM execution locks.
7. Official ``AccuracyEvaluator`` comparison (MERE/MARE, small-value,
   cancellation, integer exactness, ``compare: false`` output masking,
   per-operator ``precision_thresholds``).
8. Hash-bound upstream ``index_gather`` validation using raw proto declarations.

Only the thin glue (``_update_params`` identity-transfer variant, dtype and
threshold helpers) is re-implemented here; all numerically relevant logic is
imported from the cann-bench checkout given by ``--kernel-eval-src`` (plus
the separately pinned upstream index checker) and every
imported file is SHA256-recorded in the result provenance.

Structure: a parent process enumerates the catalog and spawns one bounded
child process per case so huge cases cannot destabilise the matrix. Each child
writes a durable per-case JSON record (atomic rename); the parent records a
harness error if a child dies before writing.

Run children only on the designated Linux VM with the pinned native pyasc v2
environment active (CANN ``set_env.sh`` sourced); the compile campaign shares
the same execution locks, so no extra isolation is needed.

Not a hardware claim: CaModel numerics are simulator evidence, and tick
deltas are not silicon performance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import resource
import shutil
import tempfile
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

WORKERS_DIR = Path(__file__).resolve().parent
CANNBENCH_DIR = WORKERS_DIR.parent
CI_DIR = CANNBENCH_DIR / "ci"

DEFAULT_PLATFORM = "Ascend950PR_9599"
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_MAX_RSS_GB = 24.0
RESULT_VERSION = 1

# kernel_eval leaf modules imported in dependency order. The intermediate
# packages are stubbed (their __init__ files pull in torch_npu-dependent
# device management that a CaModel host does not need).
KERNEL_EVAL_LEAVES = (
    "kernel_eval.utils.dtype_mapper",
    "kernel_eval.utils.naming",
    "kernel_eval.utils.tensor_utils",
    "kernel_eval.utils.precision",
    "kernel_eval.security.type_checker",
    "kernel_eval.data.case_loader",
    "kernel_eval.data.operator_loader",
    "kernel_eval.data.golden_loader",
    "kernel_eval.data.data_generator",
    "kernel_eval.utils.param_builder",
    "kernel_eval.eval.accuracy_eval",
    "kernel_eval.config",
)


def case_number(case):
    """Numeric catalog position; modern CaseSpec keeps its textual ID separately."""
    number = getattr(case, "case_num", None)
    return case.case_id if number is None else number


def official_input_seed(task_dir: Path, case_number: int, eval_seed: int):
    """Bind deterministic input generation to the catalog's canonical case ID."""
    task_dir = Path(task_dir).resolve()
    catalog = task_dir.parent.parent
    manifest_path = catalog / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    matches = [op for op in manifest["operators"]
               if op["function_name"] == task_dir.name]
    if len(matches) != 1:
        raise ValueError("Task must identify exactly one official catalog operator")
    operator = matches[0]
    case_id = f"{operator['rel_path']}_{case_number}"
    if case_id not in operator["case_ids"]:
        raise ValueError("Case is absent from official catalog")
    cases_path = task_dir / "cases.yaml"
    key = str(cases_path.relative_to(catalog))
    if _sha256_file(cases_path) != operator["task_files"][key]:
        raise ValueError("Official case source hash mismatch for seed binding")
    if not isinstance(eval_seed, int) or isinstance(eval_seed, bool):
        raise ValueError("Official eval_seed must be an integer")
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    case_seed = (eval_seed + int.from_bytes(digest[:8], "big") % (2**31)) % (2**31)
    return {"case_id": case_id, "eval_seed": eval_seed, "case_seed": case_seed,
            "catalog_manifest_sha256": _sha256_file(manifest_path),
            "algorithm": "sha256-first8-big-endian-mod-2**31-plus-eval_seed"}


def reference_functions(golden_loader, rel_path, golden_func):
    """Select oracle/native hooks using the official e8609f6b naming contract.

    Legacy loaders lack the two public accessors; resolve their cached module
    and proto function name exactly as the official accessors do.
    """
    selected = []
    for suffix in ("oracle", "bench"):
        accessor = getattr(golden_loader, f"get_{suffix}_function", None)
        if accessor is not None:
            hook = accessor(rel_path)
        else:
            module = golden_loader._load_module(rel_path)
            name = golden_loader._get_function_name(rel_path)
            hook = getattr(module, f"{name}_{suffix}", None)
        if hook is not None and not callable(hook):
            raise ValueError(f"Non-callable {suffix} reference hook for {rel_path}")
        selected.append(golden_func if hook is None else hook)
    return tuple(selected)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_json(value):
    """Convert arbitrary values into JSON-serialisable data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(v) for v in value]
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "tensor": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
        }
    if hasattr(value, "__dataclass_fields__"):
        return _sanitize_json(
            {k: getattr(value, k) for k in value.__dataclass_fields__})
    return str(value)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically and durably write JSON (fsync file + directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass  # directory fsync is best-effort (not portable)


def create_case_loader(kem, bench_root):
    """Serialize the official loader's automatic torch_npu device probe.

    Release before spawning children or entering the candidate execution lock.
    The upstream evaluator sources are kept unchanged.
    """
    sys.path.insert(0, str(CI_DIR))
    from vm_locks import shared_locks
    with shared_locks("execution"):
        return kem.modules["kernel_eval.data.case_loader"].CaseLoader(str(bench_root))


class KernelEvalModules:
    """Import official kernel_eval leaf modules with stubbed packages.

    The real ``kernel_eval/__init__`` only re-exports ``Config`` (light), but
    the data/utils/eval subpackage ``__init__`` files pull in device
    management that requires torch_npu. Stubbing the intermediate packages
    lets us execute the official leaf modules verbatim on a CaModel host.
    """

    def __init__(self, src_root: Path):
        src_root = src_root.resolve()
        package_dir = src_root / "kernel_eval"
        if not package_dir.is_dir():
            raise ValueError(f"kernel_eval package not found under {src_root}")
        self.src_root = src_root
        sys.path.insert(0, str(src_root))
        importlib.import_module("kernel_eval")  # real, light __init__
        self.modern = (package_dir / "benches/cann_loader.py").is_file()
        for sub in ("data", "utils", "eval", "security", "base", "benches", "registry", "checkers"):
            self._stub_package(f"kernel_eval.{sub}", package_dir / sub)
        self.modules = {}
        leaves = KERNEL_EVAL_LEAVES
        if self.modern:
            excluded = {"kernel_eval.utils.precision", "kernel_eval.data.case_loader",
                        "kernel_eval.data.operator_loader", "kernel_eval.data.golden_loader",
                        "kernel_eval.eval.accuracy_eval"}
            leaves = tuple(n for n in leaves if n not in excluded) + (
                "kernel_eval.benches.cann_loader", "kernel_eval.utils.thresholds",
                "kernel_eval.utils.compare", "kernel_eval.registry.checker_registry",
                "kernel_eval.checkers.relative_error_checker", "kernel_eval.eval.accuracy_eval")
        for name in leaves:
            self.modules[name] = importlib.import_module(name)
        if self.modern:
            loader = self.modules["kernel_eval.benches.cann_loader"]
            self.modules["kernel_eval.data.case_loader"] = types.SimpleNamespace(CaseLoader=loader.CannCaseLoader)
            self.modules["kernel_eval.data.operator_loader"] = types.SimpleNamespace(OperatorLoader=loader.CannTaskLoader)
            self.modules["kernel_eval.data.golden_loader"] = types.SimpleNamespace(GoldenLoader=loader.GoldenLoader)
            registry = self.modules["kernel_eval.registry.checker_registry"]
            if registry.get_correctness_checker("relative_error") is None:
                checker = self.modules["kernel_eval.checkers.relative_error_checker"].RelativeErrorChecker()
                registry.CheckerRegistry.register("relative_error", checker)

        self.file_sha256 = {}
        for name, module in sorted(self.modules.items()):
            file_path = getattr(module, "__file__", None)
            if file_path:
                self.file_sha256[name] = _sha256_file(Path(file_path))
        # Include the real top-level __init__ we executed.
        init_file = package_dir / "__init__.py"
        self.file_sha256["kernel_eval/__init__.py"] = _sha256_file(init_file)
        if self.modern:
            # Include transitive/lazy modules, not merely the explicitly imported leaves.
            for source in sorted(package_dir.rglob("*.py")):
                self.file_sha256[str(source.relative_to(src_root))] = _sha256_file(source)


    @staticmethod
    def _stub_package(name: str, path: Path) -> None:
        if name in sys.modules:
            return
        stub = types.ModuleType(name)
        stub.__path__ = [str(path)]
        sys.modules[name] = stub

    def __getattr__(self, item: str):
        for name, module in self.modules.items():
            if name.endswith(f".{item}"):
                return module
        raise AttributeError(item)


# ---------------------------------------------------------------------------
# Ports of the thin evaluator glue (identity device transfer for CaModel).
# ---------------------------------------------------------------------------

def positional_params_from_signature(golden_func, tensors, case_attrs):
    """Build params positionally after a get_input reorder.

    Verbatim port of the official evaluator's get_input branch: golden
    signature order, tensor-annotated parameters consume tensors in order
    (None when exhausted), attrs fill matching names, defaults fill the rest.
    """
    params = {}
    tensor_idx = 0
    for param_name, param in inspect.signature(golden_func).parameters.items():
        annotation = (
            str(param.annotation)
            if param.annotation != inspect.Parameter.empty
            else ""
        )
        if "Tensor" in annotation:
            if tensor_idx < len(tensors):
                params[param_name] = tensors[tensor_idx]
                tensor_idx += 1
            else:
                params[param_name] = None
        elif param_name in case_attrs:
            params[param_name] = case_attrs[param_name]
        elif param.default != inspect.Parameter.empty:
            params[param_name] = param.default
    return params


def update_params_identity(params, tensors, tensor_type):
    """``OpRunner._update_params`` with identity device transfer.

    CaModel executes on host tensors, so the official device move collapses
    to returning the same objects; structure handling (None skipping,
    list-vs-single, exhausted fallbacks) is preserved verbatim.
    """
    updated = {}
    tensor_idx = 0
    for key, value in params.items():
        if isinstance(value, tensor_type):
            while tensor_idx < len(tensors) and tensors[tensor_idx] is None:
                tensor_idx += 1
            if tensor_idx < len(tensors):
                item = tensors[tensor_idx]
                updated[key] = item if isinstance(item, tensor_type) else item[0]
                tensor_idx += 1
            else:
                updated[key] = value
        elif isinstance(value, list) and value and isinstance(value[0], tensor_type):
            while tensor_idx < len(tensors) and tensors[tensor_idx] is None:
                tensor_idx += 1
            if tensor_idx < len(tensors):
                updated[key] = (
                    tensors[tensor_idx]
                    if isinstance(tensors[tensor_idx], list)
                    else [tensors[tensor_idx]]
                )
                tensor_idx += 1
            else:
                updated[key] = value
        else:
            updated[key] = value
    return updated


def determine_dtype(outputs, op_output_dtypes, case_dtypes):
    """Port of ``Evaluator._determine_dtype``."""
    dtype = None
    if outputs is not None:
        first = None
        if isinstance(outputs, (list, tuple)) and outputs:
            first = outputs[0]
        elif not isinstance(outputs, (list, tuple)):
            first = outputs
        if first is not None and hasattr(first, "dtype"):
            dtype = str(first.dtype).replace("torch.", "")
    if dtype is None and op_output_dtypes:
        first_list = op_output_dtypes[0]
        dtype = first_list[0] if isinstance(first_list, list) else first_list
    return dtype or (case_dtypes[0] if case_dtypes else "float32")


def merged_thresholds(base_thresholds, op_thresholds):
    """Port of ``Evaluator._get_merged_thresholds``."""
    if op_thresholds:
        merged = dict(base_thresholds)
        merged.update(op_thresholds)
        return merged
    return base_thresholds


BOOL_GENERATOR_SHA256 = "85ed3923ed8fd3cec795a729e7ab707cb9c3d19672dea7ac7179d5146d734979"


def install_official_bool_generation(data_generator, dtype_mapper):
    """Patch only bool dispatch; keep existing recursion and non-bool rules."""
    import torch
    path = WORKERS_DIR / "upstream_index_check" / "bool_generator.py"
    if _sha256_file(path) != BOOL_GENERATOR_SHA256:
        raise ValueError("Official bool generator hash mismatch")
    spec = importlib.util.spec_from_file_location("cannbench_bool_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = data_generator.generate_input_tensor

    def generate(self, shape, dtype, value_range=None):
        if dtype_mapper.str_to_torch_dtype(dtype) != torch.bool:
            return original(shape, dtype, value_range)
        if shape is None or (isinstance(shape, list) and None in shape):
            return None
        lo, hi = self._parse_range(value_range, dtype)
        return module._gen_bool(self, shape, lo, hi)

    data_generator.generate_input_tensor = types.MethodType(generate, data_generator)
    return {"source_commit": "08d519c503843bce5fd4672ffa2259abeb22fb00",
            "source_sha256": BOOL_GENERATOR_SHA256, "scope": "bool-only"}


INDEX_CHECK_SHA256 = "5ba91699fd586345a0bc715a469259feddb33cee9afca3674d8a009d69efdc12"


def validate_declared_indices(task_dir, params, case_attrs, candidate_out):
    """Use raw official declarations: older OperatorLoader drops index_gather."""
    import yaml

    raw = yaml.safe_load((Path(task_dir) / "proto.yaml").read_text())
    outputs = raw["operator"].get("outputs", [])
    declared = [out for out in outputs if "index_gather" in out]
    if not declared:
        return {"required": False, "passed": True}
    for out in declared:
        if not isinstance(out["index_gather"], dict) or not out["index_gather"]:
            raise ValueError("Invalid official index_gather declaration")
    path = WORKERS_DIR / "upstream_index_check" / "index_check.py"
    if _sha256_file(path) != INDEX_CHECK_SHA256:
        raise ValueError("Official index checker hash mismatch")
    spec = importlib.util.spec_from_file_location("cannbench_index_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    op = types.SimpleNamespace(outputs=[types.SimpleNamespace(**out)
                                        for out in outputs])
    passed, message = module.validate_index_gather_outputs(
        op, params, case_attrs, candidate_out)
    return {"required": True, "passed": bool(passed), "message": message,
            "source_commit": "08d519c503843bce5fd4672ffa2259abeb22fb00",
            "source_sha256": INDEX_CHECK_SHA256,
            "declarations": declared}


def ignore_output_indices(op_outputs):
    """Port of ``Evaluator._get_ignore_output_indices``."""
    return [idx for idx, output in enumerate(op_outputs or [])
            if not getattr(output, "compare", True)]


# ---------------------------------------------------------------------------
# Child: evaluate one case and write a durable JSON record.
# ---------------------------------------------------------------------------

def _set_address_space_limit(max_rss_gb: float) -> None:
    if max_rss_gb <= 0:
        return
    limit = int(max_rss_gb * (1 << 30))
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _set_fd_limit() -> None:
    """Raise the open-file soft limit to the hard limit.

    The CaModel simulator opens hundreds of descriptor-backed files (config,
    microcode, per-core dump logs). With the default 1024 soft limit the
    simulator aborts the process with EMFILE (SIGABRT, no Python traceback),
    which left stale-looking case records before the crash-reporting fix.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = hard if hard != resource.RLIM_INFINITY else 1 << 20
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


def _git_provenance(repo: Path) -> dict:
    def _git(*argv):
        proc = subprocess.run(
            ["git", "-C", str(repo), *argv],
            capture_output=True, text=True, timeout=30)
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"

    if not (repo / ".git").exists():
        return {"head": "unknown", "dirty": "unknown"}
    status = _git("status", "--porcelain")
    return {
        "head": _git("rev-parse", "HEAD"),
        "dirty": (status != "") if status != "unknown" else "unknown",
    }


def clone_inputs(tensors, *, promote_fp64: bool, torch):
    """Return a structurally identical, storage-isolated copy.

    The official helpers alias host CPU tensors (``Tensor.cpu()`` returns the
    same object, and integer/bool tensors keep their object through fp64
    promotion), so golden and candidate would otherwise share storage and an
    in-place mutation in one consumer could corrupt another. Every consumer
    (fp64 golden, native CPU golden, candidate) gets its own clone of the same
    logical inputs; nested lists are handled recursively.
    """
    result = []
    for item in tensors:
        if isinstance(item, torch.Tensor):
            clone = item.detach().clone()
            if promote_fp64 and item.is_floating_point():
                clone = clone.double()
            result.append(clone)
        elif isinstance(item, (list, tuple)):
            result.append([
                sub.detach().clone().double()
                if (isinstance(sub, torch.Tensor) and promote_fp64
                    and sub.is_floating_point())
                else (sub.detach().clone()
                      if isinstance(sub, torch.Tensor) else sub)
                for sub in item
            ])
        else:
            result.append(item)
    return result


def nan_inf_report(candidate_out, golden_out, *, torch,
                   max_elements: int = 64 << 20):
    """Explicit NaN/Inf position evidence alongside the official compare.

    The official ``AccuracyEvaluator`` already implements NaN/Inf semantics;
    this records raw position mismatches as extra evidence. Large tensors are
    skipped (recorded as such) to respect the child's memory cap.
    """
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    report = []
    for idx, (actual, expected) in enumerate(
            zip(_as_list(candidate_out), _as_list(golden_out))):
        entry = {"index": idx}
        if not (isinstance(actual, torch.Tensor)
                and isinstance(expected, torch.Tensor)):
            entry["skipped"] = "non_tensor_output"
        elif tuple(actual.shape) != tuple(expected.shape):
            entry["skipped"] = "shape_mismatch"
        elif actual.numel() > max_elements:
            entry["skipped"] = "too_large"
            entry["numel"] = actual.numel()
        else:
            actual_nan = torch.isnan(actual)
            expected_nan = torch.isnan(expected)
            actual_inf = torch.isinf(actual)
            expected_inf = torch.isinf(expected)
            entry.update({
                "actual_nan_count": int(actual_nan.sum()),
                "expected_nan_count": int(expected_nan.sum()),
                "actual_inf_count": int(actual_inf.sum()),
                "expected_inf_count": int(expected_inf.sum()),
                "nan_position_mismatches": int(
                    (actual_nan != expected_nan).sum()),
                "inf_position_mismatches": int(
                    (actual_inf != expected_inf).sum()),
            })
        report.append(entry)
    return report


def runtime_provenance(*, expect_version: str = None,
                       expect_source: str = None) -> dict:
    """Record (and optionally enforce) the installed pyasc runtime pin."""
    import importlib.metadata as metadata

    record = {
        "expect_version": expect_version,
        "expect_source": expect_source,
        "asc_file": "unknown",
        "asc_version": "unknown",
        "asctile_file": "unknown",
        "asctile_version": "unknown",
        "extension_sha256": {},
        "asc_extension_sha256": {},
        "asctile_extension_sha256": {},
        "wheel_metadata_sha256": "unknown",
    }
    try:
        asc = importlib.import_module("asc")
        asc_file = Path(getattr(asc, "__file__"))
        record["asc_file"] = str(asc_file)
        record["asc_version"] = str(
            getattr(asc, "__version__", _safe_version(metadata, "pyasc")))
        for so_path in sorted(asc_file.resolve().parent.rglob("*.so")):
            record["asc_extension_sha256"][so_path.name] = _sha256_file(so_path)
    except Exception as exc:  # pragma: no cover - runtime dependent
        record["asc_version"] = f"error: {type(exc).__name__}"
    try:
        asctile = importlib.import_module("asctile")
        asctile_file = Path(getattr(asctile, "__file__"))
        record["asctile_file"] = str(asctile_file)
        record["asctile_version"] = str(
            getattr(asctile, "__version__",
                    _safe_version(metadata, "pyasc")))
        for so_path in sorted(asctile_file.resolve().parent.rglob("*.so")):
            record["asctile_extension_sha256"][so_path.name] = _sha256_file(
                so_path)
    except Exception as exc:  # pragma: no cover - runtime dependent
        record["asctile_version"] = f"error: {type(exc).__name__}"
    record["extension_sha256"] = {
        **record["asc_extension_sha256"],
        **record["asctile_extension_sha256"],
    }
    record["has_native_extension"] = bool(record["extension_sha256"])
    try:
        dist = metadata.distribution("pyasc")
        metadata_file = None
        for candidate in ("METADATA", "RECORD"):
            located = dist.locate_file(f"pyasc-{dist.version}.dist-info/{candidate}")
            if Path(located).exists():
                metadata_file = Path(located)
                break
        if metadata_file is not None:
            record["wheel_metadata_sha256"] = _sha256_file(metadata_file)
    except Exception:  # pragma: no cover - runtime dependent
        pass
    return _enforce_pins(record, expect_version, expect_source)


def _enforce_pins(record, expect_version, expect_source):
    """Fail-closed pin enforcement on the runtime provenance record.

    Compare against the VERSION stamp only (never the install path, which
    would make the check a path-substring tautology). The runtime stamps the
    source pin as setuptools_scm's short hash (+g<8 hex>), so the full pin is
    matched by its leading 8 characters in the version stamp.
    """
    version_stamp = " ".join(str(record.get(key)) for key in
                             ("asc_version", "asctile_version"))
    record["observed_version_stamp"] = version_stamp
    checks = []
    if expect_version:
        checks.append(("version", expect_version, expect_version))
    if expect_source:
        checks.append(("source", expect_source, expect_source[:8]))
    for label, expected, needle in checks:
        if not needle or needle not in version_stamp:
            record["pin_mismatch"] = (
                f"expected {label} {expected!r} not present in observed "
                f"runtime version stamp {version_stamp!r}")
            raise RuntimeError(record["pin_mismatch"])
    return record


def _safe_version(metadata, name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return "unknown"


def terminate_process_group(proc, grace_seconds: float = 10.0) -> None:
    """Terminate a child and every descendant (bounded, SIGTERM→SIGKILL)."""
    import signal

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            break  # Descendants can outlive the leader and retain its pipes.
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _free_bytes_stable(path, attempts=4, interval=0.5):
    """Free bytes of path, tolerating transient visibility errors.

    Shared/network filesystems (e.g. Parallels Shared Folders) can briefly
    flap a directory entry out of view; a single failed stat must not be
    read as a disk-full condition nor crash the monitor. Only a persistent
    failure is raised.
    """
    last = None
    for _ in range(attempts):
        try:
            return shutil.disk_usage(path).free
        except (FileNotFoundError, OSError) as exc:
            last = exc
            time.sleep(interval)
    raise last


def run_bounded_case(command, timeout_seconds, scratch_parent,
                     min_free_bytes=10 << 30, poll_seconds=2.0):
    """Own simulator scratch in the parent, including timeout/error cleanup."""
    scratch_parent = Path(scratch_parent)
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="camodel-case-",
                                     dir=scratch_parent) as scratch:
        if _free_bytes_stable(scratch) < min_free_bytes:
            return None, "", "", False, "disk_reserve"
        env = os.environ.copy()
        env["CAMODEL_LOG_PATH"] = scratch
        proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                start_new_session=True, env=env)
        deadline = time.monotonic() + timeout_seconds
        timed_out = False
        resource_failure = None
        try:
            while True:
                remaining = deadline - time.monotonic()
                try:
                    free_bytes = _free_bytes_stable(scratch)
                except OSError:
                    # Persistent scratch loss (not a full disk): classify as a
                    # recoverable infrastructure failure, never crash the loop.
                    resource_failure = "scratch_io"
                    timed_out = False
                    terminate_process_group(proc)
                    stdout, stderr = proc.communicate()
                    break
                if remaining <= 0 or free_bytes < min_free_bytes:
                    timed_out = remaining <= 0
                    resource_failure = None if timed_out else "disk_reserve"
                    terminate_process_group(proc)
                    stdout, stderr = proc.communicate()
                    break
                try:
                    stdout, stderr = proc.communicate(
                        timeout=min(poll_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            # The session leader can exit before descendants. Kill the exact
            # group we created before removing its exclusively owned scratch.
            import signal
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        return proc.returncode, stdout, stderr, timed_out, resource_failure


PHASE_LOCK_WAIT = "lock_wait"
PHASE_CANDIDATE = "candidate"
_COMPUTE_PHASES = ("contract", "generate", "get_input", "params", "golden",
                   "cpu_native")


def _write_phase_marker(result_file: Path, phase: str) -> None:
    marker = result_file.with_name(result_file.name + ".phase")
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_suffix(marker.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps({"phase": phase, "ts": time.time()}),
                       encoding="utf-8")
        os.replace(tmp, marker)
    except OSError:
        pass  # best-effort progress marker, never fatal


def _read_phase_marker(result_file: Path):
    marker = result_file.with_name(result_file.name + ".phase")
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("phase")
    except (OSError, ValueError):
        return None


def classify_timeout(phase):
    """(status, failure_class, rerun_pending) for a timed-out case.

    A child killed while waiting for the execution lock is NOT a numerical
    failure: it is a queued case to rerun later. Compute (golden/generate) and
    candidate timeouts are genuine resource/time data about the case itself.
    """
    if phase == PHASE_LOCK_WAIT:
        return "lock_wait_timeout", "lock_wait_timeout", True
    if phase in _COMPUTE_PHASES:
        return "timeout", "compute_timeout", False
    if phase in (PHASE_CANDIDATE, "compare", "done"):
        return "timeout", "candidate_timeout", False
    return "timeout", "timeout", False


def load_candidate_func(candidate_root: Path, op_name: str):
    """Import the candidate callable for ``op_name`` from ``candidate_root``.

    Handles both campaign states: completed (re-exporting ``__init__.py`` and
    a real ``_pyasc_runtime.py``) and in-progress (flat op modules only).
    For the in-progress layout the runtime helper is synthesized exactly like
    the compile gate's ``load_candidate``: an inert module whose
    ``ensure_npu_platform`` is a no-op, so the candidate's relative import
    resolves and the platform override below still applies.
    """
    sys.path.insert(0, str(candidate_root.parent))
    package = importlib.import_module(candidate_root.name)
    runtime_name = f"{candidate_root.name}._pyasc_runtime"
    runtime_path = candidate_root / "_pyasc_runtime.py"
    if runtime_name not in sys.modules and not runtime_path.is_file():
        import types

        shim = types.ModuleType(runtime_name)
        shim.ensure_npu_platform = lambda: None
        sys.modules[runtime_name] = shim
    try:
        return getattr(package, op_name)
    except AttributeError:
        # No re-exporting __init__.py yet: import the op submodule.
        module = importlib.import_module(f"{candidate_root.name}.{op_name}")
        return getattr(module, op_name)


def load_execution_case(case_loader, task_dir, case_id, overlay_root=None,
                        overlay_manifest_sha256=None):
    """Use the official parser for a source-verified restoration, when supplied."""
    op = Path(task_dir).name
    if overlay_root is not None:
        from reconcile_case_source import load_verified_overlay
        restored = load_verified_overlay(overlay_root, overlay_manifest_sha256,
                                         task_dir, case_id)
        if restored is not None:
            raw, evidence = restored
            case = case_loader._parse_case(
                raw, op, str(Path(task_dir) / 'cases.yaml'), op)
            if case is None or case_number(case) != case_id:
                raise ValueError('official parser rejected restored case')
            return case, evidence
    return next(c for c in case_loader.scan_by_rel_path(op)
                if case_number(c) == case_id), None


def run_case_child(args) -> int:
    started = time.time()
    task_dir = Path(args.task_dir).resolve()
    candidate_root = Path(args.candidate_root).resolve()
    result_file = Path(args.result_file)
    result = {
        "schema_version": RESULT_VERSION,
        "evidence": "camodel-case",
        "operator": task_dir.name,
        "case_id": args.case_id,
        "status": "error",
        "phase": "init",
        "platform": args.platform,
        "seed": args.seed,
        "max_rss_gb": args.max_rss_gb,
        "candidate_root": str(candidate_root),
        "task_dir": str(task_dir),
        "kernel_eval_src": str(Path(args.kernel_eval_src).resolve()),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "started_epoch": started,
    }

    def _mark(phase):
        result["phase"] = phase
        _write_phase_marker(result_file, phase)

    _mark("init")
    try:
        _mark("candidate_presence")
        if not (candidate_root / f"{task_dir.name}.py").is_file():
            result["failure_class"] = "missing_candidate"
            raise FileNotFoundError(f"candidate module missing for {task_dir.name}")
        _set_address_space_limit(args.max_rss_gb)
        _set_fd_limit()
        kem = KernelEvalModules(args.kernel_eval_src)
        import torch  # noqa: F401  (lazy: module must import without torch)

        input_seed = official_input_seed(task_dir, args.case_id, args.seed)
        result["input_generation"] = input_seed

        result["provenance"] = {
            "kernel_eval_file_sha256": kem.file_sha256,
            "cann_bench_git": _git_provenance(
                Path(args.kernel_eval_src).resolve().parent),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
        }
        _mark("contract")
        bench_root = task_dir.parent
        case_loader = create_case_loader(kem, bench_root)
        op_name = task_dir.name
        case, restoration = load_execution_case(
            case_loader, task_dir, args.case_id,
            getattr(args, "source_overlay", None),
            getattr(args, "source_overlay_sha256", None))
        if restoration is not None:
            result["provenance"]["source_restoration"] = restoration
        operator_loader = kem.modules[
            "kernel_eval.data.operator_loader"].OperatorLoader(str(bench_root))
        op_info = operator_loader.get_operator(op_name)
        golden_loader = kem.modules[
            "kernel_eval.data.golden_loader"].GoldenLoader(str(bench_root))
        golden_func = golden_loader.get_golden_function(op_name)
        oracle_func, native_func = reference_functions(
            golden_loader, op_name, golden_func)
        result["reference_functions"] = {
            "oracle": oracle_func.__name__, "native": native_func.__name__,
            "selection_source": "cann-bench/e8609f6b5891361fb6f0afb141eb8909c1b4a59b",
        }
        get_input_func = golden_loader.get_input_function(op_name)
        data_generator = kem.modules[
            "kernel_eval.data.data_generator"].DataGenerator()
        if kem.modern:
            result["provenance"]["bool_generator_policy"] = {
                "scope": "official-generator-unmodified",
                "source_sha256": kem.file_sha256["kernel_eval.data.data_generator"],
            }
        else:
            result["provenance"]["bool_generator_policy"] = install_official_bool_generation(
                data_generator, kem.modules["kernel_eval.utils.dtype_mapper"])
        param_builder = kem.modules[
            "kernel_eval.utils.param_builder"].ParamBuilder(golden_loader)
        base_thresholds = kem.modules["kernel_eval.config"].Config() \
            .precision_thresholds
        result["provenance"]["task_files_sha256"] = {
            name: _sha256_file(task_dir / name)
            for name in ("proto.yaml", "cases.yaml", "golden.py")
        }

        _mark("generate")
        input_tensors = data_generator.generate_input_tensors_from_case(
            input_shapes=case.input_shapes,
            dtypes=case.dtypes,
            value_ranges=case.value_ranges,
            seed=input_seed["case_seed"],
        )
        case_attrs = case.attrs or {}

        _mark("get_input")
        if get_input_func is not None:
            params_for_get_input = {}
            for i, input_info in enumerate(op_info.inputs):
                params_for_get_input[input_info.name] = (
                    input_tensors[i] if i < len(input_tensors) else None)
            for attr_key, attr_val in case_attrs.items():
                if attr_key not in params_for_get_input:
                    params_for_get_input[attr_key] = attr_val
            if "skip2_exist" not in params_for_get_input:
                params_for_get_input["skip2_exist"] = case_attrs.get(
                    "skip2_exist", True)
            input_tensors = get_input_func(**params_for_get_input)
            if isinstance(input_tensors, tuple):
                input_tensors = list(input_tensors)
        result["has_get_input"] = get_input_func is not None

        _mark("params")
        if get_input_func is not None:
            params = positional_params_from_signature(
                golden_func, input_tensors, case_attrs)
        else:
            params = param_builder.build_call_params(
                golden_func, case, input_tensors)

        _mark("golden")
        # Storage-isolated inputs: official helpers alias host CPU tensors and
        # integer/bool tensors survive fp64 promotion as the same object, so
        # each consumer gets its own clone of the identical logical inputs.
        fp64_inputs = clone_inputs(input_tensors, promote_fp64=True,
                                   torch=torch)
        if get_input_func is not None:
            golden_params = update_params_identity(params, fp64_inputs,
                                                   torch.Tensor)
        else:
            golden_params = param_builder.build_call_params(
                oracle_func, case, fp64_inputs)
        t0 = time.perf_counter()
        with torch.no_grad():
            golden_out = oracle_func(**golden_params)
        golden_us = (time.perf_counter() - t0) * 1_000_000

        _mark("cpu_native")
        cpu_inputs = clone_inputs(input_tensors, promote_fp64=False,
                                  torch=torch)
        if get_input_func is not None:
            cpu_params = positional_params_from_signature(
                native_func, cpu_inputs, case_attrs)
        else:
            cpu_params = param_builder.build_call_params(
                native_func, case, cpu_inputs)
        with torch.no_grad():
            cpu_out = native_func(**cpu_params)

        _mark(PHASE_LOCK_WAIT)
        sys.path.insert(0, str(CI_DIR))
        from vm_locks import shared_locks

        with shared_locks("execution"):
            _mark(PHASE_CANDIDATE)
            import asc.runtime.config as config
            from asc.lib import runtime as rt

            result["provenance"]["runtime"] = runtime_provenance(
                expect_version=args.expect_pyasc_version,
                expect_source=args.expect_pyasc_source)
            config.set_platform(
                config.Backend.Model, config.Platform(args.platform))
            candidate_func = load_candidate_func(candidate_root, op_name)
            overridden = "ensure_npu_platform" in candidate_func.__globals__
            if overridden:
                candidate_func.__globals__["ensure_npu_platform"] = (
                    lambda: None)
            result["candidate_sha256"] = _sha256_file(
                candidate_root / f"{op_name}.py")
            result["ensure_npu_platform_overridden"] = overridden
            candidate_inputs = clone_inputs(
                input_tensors, promote_fp64=False, torch=torch)
            candidate_params = update_params_identity(
                params, candidate_inputs, torch.Tensor)
            tick0 = rt.current_tick()
            t0 = time.perf_counter()
            candidate_out = candidate_func(**candidate_params)
            candidate_us = (time.perf_counter() - t0) * 1_000_000
            tick1 = rt.current_tick()
        result["timing"] = {
            "golden_us": golden_us,
            "candidate_us": candidate_us,
        }
        result["camodel_ticks"] = (
            tick1 - tick0 if tick0 is not None and tick1 is not None else None)

        _mark("compare")
        op_output_dtypes = (
            [out.dtype for out in op_info.outputs]
            if op_info and op_info.outputs else [])
        dtype = determine_dtype(candidate_out, op_output_dtypes, case.dtypes)
        thresholds = merged_thresholds(
            base_thresholds,
            op_info.precision_thresholds if op_info else None)
        ignores = ignore_output_indices(
            op_info.outputs if op_info else None)
        accuracy_evaluator = kem.modules[
            "kernel_eval.eval.accuracy_eval"].AccuracyEvaluator(base_thresholds)
        accuracy = accuracy_evaluator.evaluate(
            ai_output=candidate_out,
            golden_output=golden_out,
            dtype=dtype,
            custom_thresholds=thresholds,
            **({"native_output": cpu_out} if kem.modern else {"cpu_output": cpu_out}),
            ignore_output_indices=ignores,
        )
        result["accuracy"] = _sanitize_json(
            accuracy.to_dict() if kem.modern else accuracy)
        result["accuracy"]["format_all_outputs"] = (
            accuracy.format_all_outputs()
            if hasattr(accuracy, "format_all_outputs") else None)
        result["nan_inf_check"] = nan_inf_report(
            candidate_out, golden_out, torch=torch)
        result["dtype"] = dtype
        result["thresholds"] = _sanitize_json(thresholds)
        result["ignore_output_indices"] = ignores
        # Use untouched logical inputs, never candidate-mutated clones.
        result["index_gather_check"] = validate_declared_indices(
            task_dir, params, case_attrs, candidate_out)
        result["status"] = ("passed" if accuracy.passed and
                            result["index_gather_check"]["passed"] else "failed")
        _mark("done")
    except MemoryError as exc:
        result["status"] = "error"
        result["failure_class"] = "resource_limit"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc()[-4000:],
        }
    except Exception as exc:  # durable record, never raise past the child
        result["status"] = "error"
        result["failure_class"] = result.get("failure_class", "exception")
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc()[-4000:],
        }
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["resources"] = {
            "max_rss_kb": usage.ru_maxrss,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
        }
        result["wall_seconds"] = time.time() - started
        _atomic_write_json(result_file, result)
    return 0 if result["status"] == "passed" else 1


# ---------------------------------------------------------------------------
# Parent: enumerate the catalog and spawn one bounded child per case.
# ---------------------------------------------------------------------------

def _child_command(args, task_dir: Path, case_id: int,
                   result_file: Path) -> list:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--task-dir", str(task_dir),
        "--case-id", str(case_id),
        "--candidate-root", str(args.candidate_root),
        "--kernel-eval-src", str(args.kernel_eval_src),
        "--result-file", str(result_file),
        "--max-rss-gb", str(args.max_rss_gb),
        "--platform", args.platform,
    ]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.expect_pyasc_version:
        command += ["--expect-pyasc-version", args.expect_pyasc_version]
    if args.expect_pyasc_source:
        command += ["--expect-pyasc-source", args.expect_pyasc_source]
    if getattr(args, "source_overlay", None) is not None:
        command += ["--source-overlay", str(args.source_overlay),
                    "--source-overlay-sha256", args.source_overlay_sha256]
    return command


def run_parent(args) -> int:
    started = time.time()
    catalog_dir = Path(args.catalog).resolve()
    output_dir = Path(args.output_dir).resolve()
    if getattr(args, "source_overlay", None) is not None:
        # Restored contracts require new reports. Never silently reuse errors
        # or passes recorded for the captured null-valued contract.
        if output_dir.exists() and any(output_dir.rglob("case_*.json")):
            raise ValueError('source overlay requires a fresh result directory')
        overlay_manifest = Path(args.source_overlay) / "MANIFEST.json"
        if _sha256_file(overlay_manifest) != args.source_overlay_sha256:
            raise ValueError('overlay manifest hash mismatch')
    manifest = json.loads(
        (catalog_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_sha = _sha256_file(catalog_dir / "manifest.json")
    tasks_root = catalog_dir / "tasks"
    # function_name is the snake_case task directory name; "operator" is the
    # PascalCase display name (e.g. "Sigmoid"), which would never match a dir.
    manifest_ops = [entry["function_name"] for entry in manifest["operators"]]
    if args.ops:
        wanted = {op.strip() for op in args.ops.split(",") if op.strip()}
        unknown = wanted.difference(manifest_ops)
        if unknown:
            raise ValueError(f"operators not in manifest: {sorted(unknown)}")
        op_order = [op for op in manifest_ops if op in wanted]
    else:
        op_order = manifest_ops

    kem = KernelEvalModules(args.kernel_eval_src)
    case_loader = create_case_loader(kem, tasks_root)
    all_cases = case_loader.scan_all_cases()
    cases_by_op = {}
    for case in all_cases:
        cases_by_op.setdefault(case.rel_path, []).append(case)
    for op_cases in cases_by_op.values():
        op_cases.sort(key=lambda c: case_number(c))

    summary = {
        "schema_version": RESULT_VERSION,
        "evidence": "camodel-matrix",
        "platform": args.platform,
        "seed": args.seed,
        "catalog_manifest_sha256": manifest_sha,
        "catalog": str(catalog_dir),
        "candidate_root": str(Path(args.candidate_root).resolve()),
        "kernel_eval_src": str(Path(args.kernel_eval_src).resolve()),
        "kernel_eval_file_sha256": kem.file_sha256,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "timeout_seconds": args.timeout_seconds,
        "max_rss_gb": args.max_rss_gb,
        "expect_pyasc_version": args.expect_pyasc_version,
        "expect_pyasc_source": args.expect_pyasc_source,
        "started_epoch": started,
        "operators": {},
        "totals": {"passed": 0, "failed": 0, "error": 0, "timeout": 0,
                   "lock_wait_timeout": 0, "skipped": 0, "planned": 0},
        "status": "running",
    }
    print(f"[camodel] planned operators: {len(op_order)}", flush=True)
    for op in op_order:
        op_cases = cases_by_op.get(op, [])
        op_dir = output_dir / op
        op_summary = {"planned": len(op_cases), "passed": 0, "failed": 0,
                      "error": 0, "timeout": 0, "lock_wait_timeout": 0,
                      "skipped": 0}
        summary["operators"][op] = op_summary
        summary["totals"]["planned"] += len(op_cases)
    schedule = load_case_schedule(getattr(args, "case_plan", None), tasks_root,
                                  manifest_sha, op_order, cases_by_op,
                                  source_overlay_sha256=getattr(args, "source_overlay_sha256", None))
    summary["source_overlay_sha256"] = getattr(args, "source_overlay_sha256", None)
    summary["deferred_cases"] = []
    summary["deferrals"] = {}
    interrupted = False
    for op, case, defer_reason in schedule:
        stop_file = getattr(args, "stop_file", None)
        if stop_file is not None and Path(stop_file).exists():
            interrupted = True
            break
        if defer_reason == "requires_source_reconciliation":
            summary["deferred_cases"].append({"operator": op, "case_id": case_number(case)})
            continue
        if defer_reason:
            # Explicit scheduling deferral (e.g. measured cost beyond the
            # scope's per-case budget, prior definitive outcome, or missing
            # candidate). NOT a source-contract claim and NOT a numerical
            # result; the case stays in the full backlog under its reason.
            summary["deferrals"].setdefault(defer_reason, []).append(
                {"operator": op, "case_id": case_number(case)})
            continue
        op_dir = output_dir / op
        op_summary = summary["operators"][op]
        result_file = op_dir / f"case_{case_number(case):04d}.json"
        if result_file.exists() and not args.force:
            try:
                prior = json.loads(result_file.read_text(
                    encoding="utf-8"))
                prior_status = prior.get("status", "error")
                if prior_status == "passed":
                    validate_resumed_pass(
                        prior, op=op, case_id=case_number(case), summary=summary,
                        task_dir=tasks_root / op,
                        candidate=Path(args.candidate_root) / f"{op}.py")
            except json.JSONDecodeError:
                prior_status = "error"
            op_summary[prior_status] = op_summary.get(prior_status, 0) + 1
            op_summary["skipped"] += 1
            summary["totals"][prior_status] = (
                summary["totals"].get(prior_status, 0) + 1)
            summary["totals"]["skipped"] += 1
            continue
        command = _child_command(args, tasks_root / op, case_number(case),
                                 result_file)
        print(f"[camodel] {op} case {case_number(case)}", flush=True)
        # Never let a stale record mask a child that dies before writing
        # (e.g. SIGABRT inside the simulator): clear any previous result
        # and progress marker so the parent reports the real outcome.
        result_file.unlink(missing_ok=True)
        result_file.with_name(result_file.name + ".phase").unlink(
            missing_ok=True)
        scratch_parent = getattr(args, "scratch_parent", None) or (output_dir / ".scratch")
        returncode, stdout, stderr, timed_out, resource_failure = run_bounded_case(
            command, args.timeout_seconds, scratch_parent)
        _atomic_write_json(
            op_dir / f"case_{case_number(case):04d}.logs.json",
            {
                "returncode": returncode,
                "timed_out": timed_out,
                "stdout_tail": (stdout or "")[-65536:],
                "stderr_tail": (stderr or "")[-65536:],
            },
        )
        if resource_failure:
            status = "error"
            _atomic_write_json(result_file, {
                "schema_version": RESULT_VERSION,
                "evidence": "camodel-case", "operator": op,
                "case_id": case_number(case), "status": status,
                "failure_class": "resource_limit",
                "resource": resource_failure,
                "runner_sha256": summary["runner_sha256"],
            })
        elif timed_out:
            # Classify by the child's last progress marker: lock-wait
            # timeouts are queue contention (rerun later), compute and
            # candidate timeouts are genuine case time/resource data.
            phase = _read_phase_marker(result_file)
            status, failure_class, rerun_pending = classify_timeout(phase)
            _atomic_write_json(result_file, {
                "schema_version": RESULT_VERSION,
                "evidence": "camodel-case",
                "operator": op,
                "case_id": case_number(case),
                "status": status,
                "phase": phase or "unknown",
                "failure_class": failure_class,
                "rerun_pending": rerun_pending,
                "timeout_seconds": args.timeout_seconds,
                "runner_sha256": summary["runner_sha256"],
            })
        elif not result_file.exists():
            status = "error"
            _atomic_write_json(result_file, {
                "schema_version": RESULT_VERSION,
                "evidence": "camodel-case",
                "operator": op,
                "case_id": case_number(case),
                "status": "error",
                "phase": "unknown",
                "failure_class": "harness_error",
                "returncode": returncode,
                "runner_sha256": summary["runner_sha256"],
            })
        else:
            try:
                record = json.loads(result_file.read_text(
                    encoding="utf-8"))
                status = record.get("status", "error")
            except json.JSONDecodeError:
                status = "error"
        op_summary[status] = op_summary.get(status, 0) + 1
        summary["totals"][status] = summary["totals"].get(status, 0) + 1
    summary["wall_seconds"] = time.time() - started
    summary["status"] = ("interrupted" if interrupted else
                         "awaiting_source_reconciliation" if summary["deferred_cases"]
                         else "complete_with_deferrals" if summary["deferrals"]
                         else "complete")
    _atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps({"totals": summary["totals"]}, indent=2), flush=True)
    return 0 if summary["status"] in ("complete", "complete_with_deferrals") and \
        summary["totals"]["failed"] == 0 and \
        summary["totals"]["error"] == 0 and \
        summary["totals"]["timeout"] == 0 else 1


def validate_resumed_pass(prior, *, op, case_id, summary, task_dir, candidate):
    """A previous pass is reusable only under the same execution contract."""
    provenance = prior.get("provenance", {})
    runtime = provenance.get("runtime", {})
    expected_tasks = {name: _sha256_file(Path(task_dir) / name)
                      for name in ("proto.yaml", "cases.yaml", "golden.py")}
    check = prior.get("index_gather_check", {})
    valid = (
        prior.get("operator") == op and type(prior.get("case_id")) is int
        and prior["case_id"] == case_id
        and prior.get("phase") == "done"
        and prior.get("accuracy", {}).get("passed") is True
        and prior.get("runner_sha256") == summary["runner_sha256"]
        and Path(candidate).is_file()
        and prior.get("candidate_sha256") == _sha256_file(Path(candidate))
        and provenance.get("task_files_sha256") == expected_tasks
        and provenance.get("kernel_eval_file_sha256") == summary["kernel_eval_file_sha256"]
        and runtime.get("expect_source") == summary["expect_pyasc_source"]
        and json.dumps(prior.get("seed")) == json.dumps(summary["seed"])
        and check.get("passed") is True
        and (op not in {"cummin", "arg_max", "top_k"} or
             (check.get("required") is True and
              check.get("source_sha256") == INDEX_CHECK_SHA256)))
    if not valid:
        raise ValueError(f"Cannot reuse stale numerical pass {op}/{case_id}; "
                         "preserve it and use a fresh output scope")


def load_case_schedule(plan_path, tasks_root, manifest_sha, op_order, cases_by_op,
                       *, source_overlay_sha256=None):
    """Validate complete case identity and immutable catalog binding before use."""
    expected = {(op, case_number(c)): c for op in op_order
                for c in cases_by_op.get(op, [])}
    if plan_path is None:
        return [(op, c, None) for (op, _), c in expected.items()]
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("catalog_manifest_sha256") != manifest_sha:
        raise ValueError("schedule catalog manifest hash mismatch")
    required_overlay = plan.get("requires_source_overlay_sha256")
    if required_overlay is not None and required_overlay != source_overlay_sha256:
        raise ValueError("schedule requires its verified source overlay")
    seen = set()
    schedule = []
    hashes = {}
    for row in plan["cases"]:
        key = (row["operator"], row["case_id"])
        if key in seen or key not in expected:
            raise ValueError("schedule duplicate or unknown case")
        seen.add(key)
        op = key[0]
        if op not in hashes:
            hashes[op] = _sha256_file(Path(tasks_root) / op / "cases.yaml")
        if row.get("cases_sha256") != hashes[op]:
            raise ValueError("schedule cases hash mismatch")
        stage = row.get("stage")
        if (stage not in {"operator_first_case", "remaining_by_input_size",
                          "requires_source_reconciliation"}
                and not (isinstance(stage, str)
                         and stage.startswith("deferred_"))):
            raise ValueError("unknown schedule stage")
        # Deferred rows keep the completeness invariant (every official case
        # is accounted for) while carrying an EXPLICIT deferral reason:
        # "requires_source_reconciliation" is reserved for unresolved source
        # contracts; "deferred_<reason>" records other scoped deferrals such
        # as scheduling budget, prior definitive outcomes, or missing
        # candidate - never as a source-defect or numerical claim.
        defer_reason = (stage if stage == "requires_source_reconciliation"
                        or stage.startswith("deferred_") else None)
        schedule.append((op, expected[key], defer_reason))
    if seen != set(expected):
        raise ValueError("schedule omits official cases")
    return schedule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true",
                        help="run a single case in-process (parent spawns)")
    parser.add_argument("--catalog", type=Path,
                        help="catalog root containing manifest.json and tasks/")
    parser.add_argument("--task-dir", type=Path,
                        help="(child) operator task directory")
    parser.add_argument("--case-id", type=int,
                        help="(child) case number")
    parser.add_argument("--candidate-root", type=Path,
                        help="cann_bench package directory")
    parser.add_argument("--kernel-eval-src", type=Path,
                        help="cann-bench src directory (contains kernel_eval/)")
    parser.add_argument("--scratch-parent", type=Path, default=None,
                        help="Directory for per-case simulator scratch "
                             "(default: <output-dir>/.scratch). Point this at "
                             "large disposable storage; the disk reserve "
                             "checks then measure that filesystem.")
    parser.add_argument("--output-dir", type=Path,
                        help="(parent) durable result directory")
    parser.add_argument("--result-file", type=Path,
                        help="(child) per-case result JSON path")
    parser.add_argument("--case-plan", type=Path,
                        help="complete hash-bound case order; unresolved inputs deferred")
    parser.add_argument("--source-overlay", type=Path,
                        help="verified nonfinite restoration bundle; requires fresh reports")
    parser.add_argument("--source-overlay-sha256",
                        help="reviewed restoration MANIFEST.json SHA256")
    parser.add_argument("--stop-file", type=Path,
                        help="stop between cases when this file exists")
    parser.add_argument("--ops", default=None,
                        help="comma-separated operator subset (manifest order)")
    parser.add_argument("--timeout-seconds", type=int,
                        default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-rss-gb", type=float,
                        default=DEFAULT_MAX_RSS_GB)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--seed", type=int, default=0,
                        help="official eval_seed added to canonical case ID hash (default: 0)")
    parser.add_argument("--force", action="store_true",
                        help="re-run cases with existing results")
    parser.add_argument("--expect-pyasc-version", default=None,
                        help="fail the case if this version stamp is not "
                             "present in the observed runtime provenance")
    parser.add_argument("--expect-pyasc-source", default=None,
                        help="fail the case if this source pin (commit) is "
                             "not present in the observed runtime provenance")
    args = parser.parse_args()
    if bool(args.source_overlay) != bool(args.source_overlay_sha256):
        parser.error('--source-overlay and --source-overlay-sha256 must be paired')
    if args.child:
        for required in ("task_dir", "case_id", "candidate_root",
                         "kernel_eval_src", "result_file"):
            if getattr(args, required) is None:
                parser.error(f"--child requires --{required.replace('_', '-')}")
        return run_case_child(args)
    for required in ("catalog", "candidate_root", "kernel_eval_src",
                     "output_dir"):
        if getattr(args, required) is None:
            parser.error(f"missing --{required.replace('_', '-')}")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
