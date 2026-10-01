"""Replay all official dispatches from the installed evaluator wheel on the VM."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from generation_gate import sha256
    wheel = next((args.bundle / "dist").glob("*.whl"))
    bundle = json.loads((args.bundle / "BUNDLE.json").read_text())
    if sha256(wheel) != bundle["wheel_sha256"]:
        raise ValueError("Evaluator wheel changed after packaging")
    with tempfile.TemporaryDirectory(prefix="evaluator-wheel-") as temporary:
        target = Path(temporary)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)], check=True)
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.startswith(("asc/", "asctile/", "cann_bench/")) and not name.endswith("/"):
                    if not (target / name).is_file() or hashlib.sha256((target / name).read_bytes()).digest() != hashlib.sha256(archive.read(name)).digest():
                        raise ValueError("Installed wheel content mismatch: " + name)
        sys.path.insert(0, str(target))
        sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "workers"))
        from local_compile_gate import evaluate
        from validate_catalog import validate
        from contracts import load_contract
        catalog = validate(args.catalog.parent)
        reports = []
        for spec in catalog["operators"]:
            name = spec["function_name"]
            task = args.catalog.parent / "tasks" / name
            if sha256(target / "cann_bench" / (name + ".py")) != bundle["operators"][name]["candidate_sha256"]:
                raise ValueError("Evaluator source mismatch: " + name)
            try:
                report = evaluate(target / "cann_bench" / (name + ".py"), name, task / "cases.yaml", load_contract(task))
            except Exception as exc:
                report = {"status": "failed", "operator": name, "error": type(exc).__name__ + ": " + str(exc)}
            reports.append(report)
        document = {"wheel_sha256": sha256(wheel), "catalog_sha256": catalog["catalog_sha256"],
                    "status": "passed" if all(r.get("status") == "passed" for r in reports) else "failed",
                    "operators": reports, "limitations": ["Numerical correctness and performance require the final CANNBench hardware job"]}
        args.output.write_text(json.dumps(document, indent=2) + "\n")
    return 0 if document["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
