"""Materialize finite, aligned short-row cases for read-order transfer tests."""
import argparse
import csv
import json
from pathlib import Path
import shutil


def prepare(task, output):
    with (task / "cases.csv").open(newline="") as f:
        public, = (r for r in csv.DictReader(f) if r["case_id"] == "13")
    if json.loads(public["input_shape"]) != [[11, 13, 16, 67]] or json.loads(public["attrs"]) != {"dim": 2}:
        raise ValueError("expected the public FP32 split-axis schema")
    for name, shape in (("lower", [139, 1, 16, 66]), ("upper", [141, 1, 16, 68])):
        dest = output / name / "tasks/level1/swi_glu"
        dest.mkdir(parents=True)
        entry = dict(operator=public["operator"], case_id=13, input_shape=[shape],
                     dtype=["float32"], attrs={"dim": 2}, value_range=[-2, 2],
                     note="Private finite short-row transfer control")
        # JSON is a YAML 1.2 subset. Serialize the same record to both inputs.
        (dest / "cases.yaml").write_text(json.dumps({"cases": [entry]}, indent=2) + "\n")
        with (dest / "cases.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=public)
            writer.writeheader()
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in entry.items()})
        for file in ("proto.yaml", "golden.py"):
            shutil.copyfile(task / file, dest / file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.task, args.output)
