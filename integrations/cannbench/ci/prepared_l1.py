"""Submit or collect one retained Exp + SwiGLU package on the standard CANNBench pool."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
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
            if (
                source.get("package_source") != f"cann_bench/{fn}.py"
                or not qualified_source(source)
                or not re.fullmatch(r"[0-9a-f]{40}", source["skills_revision"])
                or hashlib.sha256(z.read(source["package_source"])).hexdigest()
                != source["source_sha256"]
            ):
                raise ValueError(
                    "Invalid qualification or generated source for " + op
                )
    return manifest


def result_receipt(run, manifest, report, queue):
    job = json.loads((run / "job.json").read_bytes())["job"]
    if (
        set(job.get("selected_operators") or []) != {fn for _, fn in OPERATORS}
        or job.get("case_set") != "standard"
        or job.get("is_independent_environment") is not False
        or job.get("is_private") is not (manifest["visibility"] == "private")
    ):
        raise ValueError(
            "Measured job scope or visibility differs from the requested evaluation"
        )
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
        "package_sha256": evaluated_sha256,
        "qualification_sha256": manifest["files"]["qualification.json"],
        "operators": report["operators"],
        "queued_at": job["queued_at"],
        "claimed_at": job["claimed_at"],
        "finished_at": job["finished_at"],
    }


def stage(folder, visibility, q, base):
    folder = smoke.external(folder)
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(dir=base) as temporary:
        run = Path(temporary) / "prepared"
        run.mkdir()
        for filename in ("submission.zip", "qualification.json"):
            shutil.copyfile(folder / filename, run / filename)
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
            ci_client(),
            base / "pyasc-cannbench-l1",
        )
        return
    if not re.fullmatch(r"[0-9a-f]{64}", args.input):
        parser.error("Expected the SHA-256 of a staged qualification manifest")
    run = base / "pyasc-cannbench-l1" / args.input
    manifest = verify_package(run, args.input)
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
