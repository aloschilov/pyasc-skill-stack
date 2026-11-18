"""Submit or collect one retained Exp + SwiGLU package on the standard CANNBench pool."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import smoke
from single_operator import ci_client, download

OPERATORS = (("Exp", "exp"), ("SwiGlu", "swi_glu"))
MAX_ZIP_BYTES = 10 * 1024 * 1024  # CANNBench WEB-ZIP-001


def check_upload_size(path):
    size = path.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise ValueError(
            f"CANNBench WEB-ZIP-001: {path.name} is {size} bytes; "
            f"the compressed ZIP limit is {MAX_ZIP_BYTES} bytes (10 MiB)"
        )


def qualified_source(source):
    if source.get("passed_cases") == 20:
        return True
    model = source.get("camodel") or {}
    checks = model.get("numerical_pairs") or []
    case_ids = model.get("compiled_case_ids") or []
    return (
        model.get("source_sha256") == source["source_sha256"]
        and bool(re.fullmatch(r"[0-9a-f]{64}", model.get("evidence_sha256", "")))
        and len(case_ids) == 20
        and set(case_ids) == set(range(1, 21))
        and all(type(case) is int for case in case_ids)
        and all(type(pair.get("seed")) is int for pair in checks)
        and len({pair["seed"] for pair in checks}) >= 2
        and all(pair.get("launch_matches_golden") is True and
                pair.get("suppressed_rejected") is True for pair in checks)
    )


def aot_binding_sha256(qualification):
    """Bind source provenance, every build/host input and the resulting wheel."""
    payload = {
        "aot": qualification["aot"],
        "wheel_sha256": qualification["wheel_sha256"],
        "sources": {
            fn: qualification["operators"][fn]["source_sha256"]
            for _, fn in OPERATORS
        },
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def verify_target_dispatch(aot, entries):
    """Check declared coverage and derive route-origin counts from binary proofs."""
    target = aot.get("target_dispatch")
    if target is None:
        # Immutable historical jobs may still be collected. New submissions
        # require this evidence before creating a client or submission intent.
        return {}

    def require(condition, message):
        if not condition:
            raise ValueError("Invalid AOT target dispatch: " + message)

    require(isinstance(target, dict) and target.get("hardware") == "950pr",
            "missing target hardware")
    require(isinstance(target.get("evidence_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", target["evidence_sha256"]),
            "missing trace evidence digest")
    required = target.get("required_effective_aivs")
    require(isinstance(required, list) and bool(required)
            and all(type(n) is int and n > 0 for n in required)
            and len(set(required)) == len(required), "invalid required AIV capacities")
    profiles = target.get("profiles")
    require(isinstance(profiles, list) and bool(profiles), "missing resource profiles")
    capacities = set()
    origin_counts = {}
    expected = {(fn, case) for _, fn in OPERATORS for case in range(1, 21)}
    for profile in profiles:
        require(isinstance(profile, dict), "invalid resource profile")
        capacity = profile.get("effective_aivs")
        require(type(capacity) is int and capacity > 0 and capacity not in capacities,
                "invalid or duplicate AIV capacity")
        capacities.add(capacity)
        counts = {fn: {origin: 0 for origin in ("retained", "reference", "fallback")}
                  for _, fn in OPERATORS}
        origin_counts[capacity] = counts
        ub = profile.get("usable_ub_bytes")
        require(type(ub) is int and ub > 0, "missing usable UB capacity")
        rows = profile.get("cases")
        require(isinstance(rows, list) and len(rows) == len(expected),
                "profile must cover all 40 official cases")
        seen = set()
        for row in rows:
            require(isinstance(row, dict), "invalid case binding")
            fn, case = row.get("operator"), row.get("case_id")
            require(isinstance(fn, str) and type(case) is int
                    and (fn, case) in expected and (fn, case) not in seen,
                    "invalid or duplicate official case")
            seen.add((fn, case))
            key = row.get("key")
            require(isinstance(key, str) and key in entries, "missing specialization key")
            entry = entries[key]
            require(json.loads(key)[0] == fn, "wrong specialization operator")
            used_ub = entry.get("ub_bytes")
            require(type(used_ub) is int and 0 < used_ub <= ub,
                    "selected specialization exceeds or lacks valid UB allocation")
            proof = aot["native_binaries"][entry["file"].removesuffix(".o")]
            origin = proof["origin"]
            require(origin in ("retained", "reference"), "official case selected fallback")
            require((origin != "reference" or row.get("origin") == origin)
                    and ("origin" not in row or row["origin"] == origin),
                    "case origin differs from binary proof")
            counts[fn][origin] += 1
            require(row.get("binary_sha256") == entry["binary_sha256"],
                    "selected binary differs from specialization")
            grid = row.get("grid")
            require(type(grid) is int and 0 < grid <= capacity,
                    "launch grid exceeds effective AIV capacity")
    require(set(required) == capacities, "traced capacities differ from required target capacities")
    return origin_counts


def require_target_dispatch(run):
    qualification = json.loads((run / "qualification.json").read_bytes())
    if (qualification.get("packaging_mode") == "source-aot-v1"
            and not qualification.get("aot", {}).get("target_dispatch")):
        raise ValueError(
            "New AOT staging/submission requires target-capacity dispatch evidence; "
            "use collect for an already submitted historical package"
        )


def verify_aot(z, q):
    """Verify an attested source-only AOT build, separately from source numerics.

    This checks evidence bindings, not a replacement execution of that build.
    Numerical/source receipts and the installed-wheel adapter receipt remain
    distinct, and changing any submitted build input invalidates the latter.
    """
    def require(condition, message):
        if not condition:
            raise ValueError("Invalid source-aot-v1 qualification: " + message)

    def digest(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)

    def safe_name(name):
        return (isinstance(name, str) and bool(name) and "\\" not in name
                and ":" not in name and not name.startswith("/")
                and all(part not in ("", ".", "..") for part in name.split("/")))

    aot = q["aot"]
    inputs = aot["inputs"]
    require(set(inputs) == set(z.namelist()), "incomplete build input inventory")
    for name, expected in inputs.items():
        require(safe_name(name), "unsafe archive member")
        entry = z.getinfo(name)
        require((entry.external_attr >> 16) & 0o170000 != 0o120000, "symlink input")
        require(PurePosixPath(name).suffix in {
            ".py", ".cpp", ".cce", ".c", ".h", ".hpp", ".sh", ".json",
            ".toml", ".cfg", ".txt", ".md", ".in",
        } or PurePosixPath(name).name == "LICENSE", "archive is not source-only")
        content = z.read(name)
        require(not content.startswith((b"\x7fELF", b"PK\x03\x04")), "binary input")
        content.decode("utf-8")
        require(digest(expected) and hashlib.sha256(content).hexdigest() == expected,
                "changed build input " + name)

    require(re.fullmatch(r"[0-9a-f]{40}", aot.get("runtime_revision", ""))
            and digest(aot.get("runtime_extension_sha256")), "missing runtime provenance")
    require(all(digest(aot.get("compiler_sha256", {}).get(tool))
                for tool in ("bisheng", "ld.lld"))
            and digest(aot.get("build_evidence_sha256")), "missing build provenance")
    flags = json.loads(z.read("flags.json"))
    entries = json.loads(z.read("cann_bench/specializations.json"))
    native = aot["native_binaries"]
    wheel = aot["wheel_files"]
    require(bool(flags) and set(native) == set(flags), "incomplete native binary map")
    require(all(safe_name(name) and digest(value) for name, value in wheel.items()),
            "invalid wheel inventory")
    # The wheel contains only the pinned host files, rebuilt device binaries and
    # standard dist-info. In particular no compiler/runtime package is installed.
    host = {name: value for name, value in inputs.items() if name.startswith("cann_bench/")}
    require(all(wheel.get(name) == value for name, value in host.items()),
            "wheel host sources differ from qualified inputs")
    binaries = {"cann_bench/device/" + name + ".o" for name in flags}
    metadata = {name for name in wheel if re.fullmatch(
        r"cann_bench-[^/]+\.dist-info/(METADATA|WHEEL|top_level\.txt|RECORD)", name
    )}
    require({PurePosixPath(name).name for name in metadata} >= {"METADATA", "WHEEL", "RECORD"},
            "incomplete wheel metadata")
    require(set(wheel) == set(host) | binaries | metadata,
            "unexpected wheel contents or missing device binaries")
    seen = set()
    for key, entry in entries.items():
        signature = json.loads(key)
        name = entry["file"].removesuffix(".o")
        require(name not in seen and name in native and entry["file"] == name + ".o",
                "duplicate or unmapped specialization")
        seen.add(name)
        proof, build = native[name], flags[name]
        operator, symbol = signature[:2]
        require(operator in {fn for _, fn in OPERATORS}
                and proof["operator"] == operator and entry["symbol"] == symbol,
                "specialization operator or symbol mismatch")
        require(proof.get("origin") in ("retained", "reference", "fallback"), "unknown kernel origin")
        source = proof.get("pyasc_source")
        require(isinstance(source, str) and source in inputs and source.startswith("pyasc_source/")
                and source.endswith(".py"), "missing original kernel source")
        if proof["origin"] == "retained":
            require(source == f"pyasc_source/{operator}.py", "wrong retained kernel source")
        elif proof["origin"] == "reference":
            require(source not in {f"pyasc_source/{fn}.py" for _, fn in OPERATORS}
                    and digest(proof.get("source_sha256"))
                    and proof["source_sha256"] == inputs[source],
                    "reference kernel needs a separate hash-bound source")
        require(build["source"] in inputs and build["source"].endswith((".cpp", ".cce"))
                and bool(build.get("commands")), "missing AscendC source or build commands")
        binary = proof["native_sha256"]
        require(digest(binary) and binary == proof["rebuilt_sha256"]
                == build["binary_sha256"] == entry["binary_sha256"]
                == wheel.get("cann_bench/device/" + entry["file"]),
                "rebuilt/native/wheel binary mismatch")
    require(seen == set(native), "unreferenced native binary")
    verify_target_dispatch(aot, entries)

    binding = aot_binding_sha256(q)
    adapter = q["host_adapter"]
    require(q.get("packaging_binding_sha256") == binding
            and adapter.get("binding_sha256") == binding
            and adapter.get("wheel_sha256") == q["wheel_sha256"]
            and digest(adapter.get("evidence_sha256")), "unbound host adapter evidence")
    require(all(adapter.get(check) is True for check in (
        "dispatch_matches_retained", "caller_stream_preserved", "launch_grid_preserved"
    )), "host dispatch, stream or launch grid not qualified")
    for _, fn in OPERATORS:
        wrapper = f"cann_bench/{fn}.py"
        require(wrapper in inputs, "missing public operator wrapper")
        # Do not accept passed_cases=20 as a substitute for adapter qualification.
        require(qualified_source({"source_sha256": inputs[wrapper],
                                  "camodel": adapter["operators"][fn]}),
                "missing installed-wheel CaModel adapter evidence for " + fn)
        pairs = adapter["operators"][fn]["numerical_pairs"]
        origins = {proof["origin"] for proof in native.values() if proof["operator"] == fn}
        require("retained" in origins, "missing retained kernels for " + fn)
        require(all(isinstance(pair.get("scope"), str) for pair in pairs)
                and {pair["scope"] for pair in pairs} == origins
                and all(len({pair["seed"] for pair in pairs if pair["scope"] == scope}) >= 2
                        for scope in origins),
                "adapter scopes must match kernel origins with two seeds each for " + fn)
        for pair in pairs:
            if pair["scope"] != "reference":
                continue
            key = pair.get("key")
            require(isinstance(key, str) and key in entries, "missing reference check key")
            entry = entries[key]
            proof = native[entry["file"].removesuffix(".o")]
            require(proof["operator"] == fn and proof["origin"] == "reference"
                    and pair.get("binary_sha256") == entry["binary_sha256"],
                    "reference check differs from selected operator or binary")
        if "fallback" not in origins:
            check = adapter["operators"][fn].get("missing_specialization_check")
            require(isinstance(check, dict) and check.get("rejected") is True
                    and type(check.get("launches")) is int and check["launches"] == 0
                    and digest(check.get("evidence_sha256")),
                    "missing installed-wheel missing-specialization rejection for " + fn)
            key = check.get("key")
            require(isinstance(key, str) and key in entries
                    and native[entries[key]["file"].removesuffix(".o")]["operator"] == fn,
                    "missing-specialization check needs a supported operator key for " + fn)


def verify_package(run, manifest_sha):
    if smoke.sha(run / "manifest.json") != manifest_sha:
        raise ValueError("Qualification manifest differs from the selected digest")
    manifest = smoke.verify(run)
    required = {"task.json", "submission.zip", "qualification.json"}
    if not required <= set(manifest["files"]):
        raise ValueError("Missing prepared package, qualification or task snapshots")
    catalog = manifest["catalog"]
    if (
        manifest.get("target_hardware") != "950pr"
        or catalog.get("benchmark_slug") != "official-tasks"
        or catalog.get("required_cases") != 40
        or catalog.get("required_operators") != 2
        or manifest.get("visibility") not in ("public", "private")
    ):
        raise ValueError("Expected the standard 950PR two-operator L1 qualification")
    selected = [(op["operator"], op["function_name"]) for op in catalog["operators"]]
    if selected != list(OPERATORS):
        raise ValueError("Wrong L1 operators")
    q = json.loads((run / "qualification.json").read_bytes())
    mode = q.get("packaging_mode", "source-jit")
    if mode not in ("source-jit", "source-aot-v1"):
        raise ValueError("Unknown packaging mode")
    if (
        q.get("build_passed") is not True
        or q.get("package_sha256") != manifest["files"]["submission.zip"]
        or not re.fullmatch(r"[0-9a-f]{64}", q.get("wheel_sha256", ""))
    ):
        raise ValueError("Missing build evidence for these submitted ZIP bytes")
    with ZipFile(run / "submission.zip") as z:
        if (
            len(z.namelist()) != len(set(z.namelist()))
            or "build.sh" not in z.namelist()
        ):
            raise ValueError("Missing build.sh or duplicate archive members")
        for op, fn in OPERATORS:
            source = q["operators"][fn]
            source_root = "pyasc_source" if mode == "source-aot-v1" else "cann_bench"
            if (
                source.get("package_source") != f"{source_root}/{fn}.py"
                or not qualified_source(source)
                or not re.fullmatch(r"[0-9a-f]{40}", source["skills_revision"])
                or hashlib.sha256(z.read(source["package_source"])).hexdigest()
                != source["source_sha256"]
            ):
                raise ValueError(
                    "Invalid qualification or generated source for " + op
                )
        if mode == "source-aot-v1":
            verify_aot(z, q)
    return manifest


def result_receipt(run, manifest, report, queue):
    job_bytes = (run / "job.json").read_bytes()
    job = json.loads(job_bytes)["job"]
    selected = job.get("selected_operators")
    if (
        not isinstance(selected, list) or len(selected) != len(OPERATORS)
        or set(selected) != {fn for _, fn in OPERATORS}
        or job.get("case_set") != "standard"
        or job.get("is_independent_environment") is not False
        or job.get("is_private") is not (manifest["visibility"] == "private")
    ):
        raise ValueError(
            "Measured job scope or visibility differs from the requested evaluation"
        )
    # Revalidate the retained raw response, not a caller's assertion of 40/40.
    checked = smoke.evaluate(manifest["catalog"], [{"job": job}], manifest["target_hardware"])
    if (checked["gate_passed"] is not True
            or (checked["required_operators"], checked["required_cases"],
                checked["correct_cases"], checked["measured_cases"]) != (2, 40, 40, 40)
            or {(op["operator"], op["function_name"]) for op in checked["operators"]} != set(OPERATORS)
            or any(op["required_cases"] != 20 for op in checked["operators"])):
        raise ValueError("Saved CANNBench job did not qualify the exact 40-case pair")
    # generated_at is the time of gate evaluation; all substantive fields must
    # match the original collector report, including its job/package binding.
    checked.pop("generated_at")
    checked.update(scope=manifest["scope"], archive_sha256=manifest["files"]["submission.zip"],
                   job_id=job["id"], submission_id=job.get("submission_id"),
                   **{key: job.get(key) for key in
                      ("queued_at", "claimed_at", "started_at", "finished_at")})
    if (not isinstance(report, dict) or report.get("gate_passed") is not True
            or any(key not in report or report[key] != value for key, value in checked.items())):
        raise ValueError("Hardware report differs from the saved CANNBench job or manifest")
    def no_failure(value, names):
        return (all(type(value[key]) is bool and value[key] is False
                    for key in names if key in value)
                and value.get("profile_exception_type") in (None, ""))

    case_flags = ("perf_integrity_failed", "perf_collection_failed", "io_cleanup_failed")

    def valid_perf(value):
        # Raw metadata can contain timing fields that the public projection omits.
        # Inspect the same known flags recursively without discarding failures.
        return value is None or (isinstance(value, dict) and no_failure(value, case_flags)
                                 and valid_perf(value.get("metadata")))

    for operator in job["results"]["operators"]:
        if not no_failure(operator, ("zeroed_by_integrity", "zeroed_by_no_npu_perf", "zeroed_by_cpu_fallback")):
            raise ValueError("Operator reported a profiling or integrity failure: " + operator["operator"])
        for case in operator["cases"]:
            if (not no_failure(case, case_flags) or not valid_perf(case.get("_perf_result"))
                    or not valid_perf(case.get("metadata"))):
                raise ValueError("Case reported a profiling or integrity failure: " + case["case_id"])
            accuracy = case.get("accuracy")
            if not isinstance(accuracy, dict) or accuracy.get("passed") is not True:
                raise ValueError("Missing explicit numerical pass: " + case["case_id"])
            timings = case.get("op_times")
            kernels = timings.get("device_kernels") if isinstance(timings, dict) else None
            if (not isinstance(kernels, dict) or not kernels
                    or any(not isinstance(name, str) or not name.strip()
                           or type(duration) not in (int, float)
                           or not math.isfinite(duration) or duration <= 0
                           for name, duration in kernels.items())):
                raise ValueError("Missing valid device-kernel measurements: " + case["case_id"])
    score = job.get("result_score")
    if (
        not isinstance(score, (float, int))
        or isinstance(score, bool)
        or not math.isfinite(score)
    ):
        raise ValueError("CANNBench has not returned a finite service score")
    evaluated_zip = download(queue, job["submission_download_url"])
    evaluated_sha256 = hashlib.sha256(evaluated_zip).hexdigest()
    if evaluated_sha256 != manifest["files"]["submission.zip"]:
        raise ValueError("The archive returned by CANNBench differs from the prepared ZIP")
    public_job_url = None
    if manifest["visibility"] == "public":
        if not re.fullmatch(r"job_[0-9a-f]+", job["id"]):
            raise ValueError("Invalid public job ID")
        public_job_url = queue._client._url("/api/jobs/" + job["id"])
        with urllib.request.urlopen(public_job_url, timeout=30) as response:
            public_job = json.load(response)["job"]
        for field in (
            "id", "submission_id", "status", "benchmark_slug", "benchmark_version",
            "target_hardware", "case_set", "result_score", "is_private",
        ):
            if public_job.get(field) != job[field]:
                raise ValueError(f"Public CANNBench job differs in {field}")
        if set(public_job.get("selected_operators") or ()) != set(job["selected_operators"]):
            raise ValueError("Public CANNBench job differs in selected_operators")
    return {
        "job": job["id"],
        "public_job_url": public_job_url,
        "submission": job["submission_id"],
        "version": job["benchmark_version"],
        "hardware": job["target_hardware"],
        "visibility": manifest["visibility"],
        "score": score,
        "job_sha256": hashlib.sha256(job_bytes).hexdigest(),
        "package_sha256": evaluated_sha256,
        "qualification_sha256": manifest["files"]["qualification.json"],
        "operators": report["operators"],
        "queued_at": job["queued_at"],
        "claimed_at": job["claimed_at"],
        "finished_at": job["finished_at"],
    }


def stage(folder, visibility, base):
    folder = smoke.external(folder)
    check_upload_size(folder / "submission.zip")
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(dir=base) as temporary:
        run = Path(temporary) / "prepared"
        run.mkdir()
        for filename in ("submission.zip", "qualification.json"):
            shutil.copyfile(folder / filename, run / filename)
        # Check the retained copy in case the source changed during staging.
        check_upload_size(run / "submission.zip")
        require_target_dispatch(run)
        q = ci_client()
        catalog, tasks = smoke.group_snapshot(q, OPERATORS)
        smoke.save(run / "task.json", tasks)
        manifest = {
            "scope": "Level 1 Exp and SwiGLU, all standard public cases",
            "target_hardware": q.target_hardware,
            "visibility": visibility,
            "catalog": catalog,
            "aggregation_token": os.environ["CANNBENCH_AGGREGATION_TOKEN"],
            "files": {
                n: smoke.sha(run / n)
                for n in ("submission.zip", "qualification.json", "task.json")
            },
        }
        smoke.save(run / "manifest.json", manifest)
        digest = smoke.sha(run / "manifest.json")
        verify_package(run, digest)
        destination = base / digest
        if destination.exists():
            verify_package(destination, digest)
        else:
            run.rename(destination)
    print(digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "stage-private",
            "stage-public",
            "submit-private",
            "submit-public",
            "retry-partial",
            "check-size",
            "collect",
        ],
    )
    parser.add_argument(
        "input", help="Staging directory, or SHA-256 of an already staged manifest"
    )
    args = parser.parse_args()
    base = Path.home() / ".local/state"
    if args.action.startswith("stage-"):
        stage(
            Path(args.input),
            args.action.removeprefix("stage-"),
            base / "pyasc-cannbench-l1",
        )
        return
    if not re.fullmatch(r"[0-9a-f]{64}", args.input):
        parser.error("Expected the SHA-256 of a staged qualification manifest")
    run = base / "pyasc-cannbench-l1" / args.input
    if args.action != "collect":
        check_upload_size(run / "submission.zip")
    if args.action == "check-size":
        return
    manifest = verify_package(run, args.input)
    if args.action != "collect":
        require_target_dispatch(run)
    ledger = base / "pyasc-cannbench-smoke"
    ledger.mkdir(exist_ok=True, parents=True, mode=0o700)
    q = ci_client()
    deadline = time.monotonic() + 4 * 3600
    if args.action != "collect":
        partial = args.action == "retry-partial"
        public = manifest["visibility"] == "public" if partial else args.action == "submit-public"
        if manifest["visibility"] != ("public" if public else "private"):
            raise ValueError("Dispatch and manifest visibility differ")
        if (
            q._client._get_json("/api/public/service-status")["maintenance"]["enabled"]
            is not False
        ):
            raise RuntimeError("CANNBench is in maintenance")
        with (ledger / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = smoke.submit_once(
                q, run, ledger, selections=OPERATORS, public=public,
                allow_confirmed_jobs=True, retry_partial=partial
            )
        print("Submission state:", state["phase"], state.get("job_id"), flush=True)
        if state.get("other_submitted_jobs") and state["phase"] == "submitted":
            # The single service runner must also be able to collect the earlier job.
            message = (
                f"Accepted CANNBench job `{state['job_id']}`. Hardware score pending. "
                f"Use `l1-collect` with manifest `{args.input}` for the final receipt."
            )
            print(message, flush=True)
            summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary:
                with open(summary, "a", encoding="utf-8") as output:
                    print(message, file=output)
            return
    while time.monotonic() < deadline:
        code, report = smoke.collect(q, run, ledger)
        if code == 0:
            receipt = result_receipt(run, manifest, report, q)
            smoke.save(run / "service-receipt.json", receipt)
            print(
                "40/40 cases passed. CANNBench score:",
                receipt["score"],
                "job:",
                receipt["job"],
                flush=True,
            )
            return
        if code != 2:
            raise RuntimeError(
                "CANNBench did not qualify all 40 cases; see the retained report"
            )
        time.sleep(45)
    raise TimeoutError("Resume this job with collect and the same manifest digest")


if __name__ == "__main__":
    main()
