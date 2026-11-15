"""Join the 950PR msprof block packets to previously verified full-grid task intervals."""
import argparse
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path


# CANN 9.2 StructFmt.PMU_BLOCK_FMT and StarsConstant.FFTS_BLOCK_PMU_TAG.
PACKET = struct.Struct("<4HQ2BH2B3HL12Q")
BLOCK_TAG = 0b101001


def match(task, packets):
    expected = {core: (start, round(duration * 1000)) for core, start, duration in task["pmu"]}
    actual = {p[9]: (p[24], p[25] - p[24]) for p in packets}
    if len(actual) != task["grid"] or actual.keys() != expected.keys():
        return False
    first = min(start for start, _ in actual.values())
    ref = min(start for start, _ in expected.values())
    return all(abs((start - first) - (expected[c][0] - ref)) <= 1
               and abs(duration - expected[c][1]) <= 1
               for c, (start, duration) in actual.items())


def extract(root, case, metric="PipeUtilization"):
    verified = json.loads((root / f"r-{case}.json").read_text())
    device, = (root / f"profile-{case}").glob("PROF*/device_0")
    sample = json.loads((device / "sample.json").read_text())
    if sample["aiv_metrics"] != metric:
        raise ValueError(f"expected {metric}, captured {sample['aiv_metrics']}")
    raw, = (device / "data").glob("ffts_profile.data.*.slice_0")
    if list(raw.parent.glob("ffts_profile.data.*.slice_[1-9]*")):
        raise ValueError("expected a single raw slice for this short profile")
    data = raw.read_bytes()
    groups = defaultdict(list)
    for p in PACKET.iter_unpack(data):
        if p[0] & 63 == BLOCK_TAG and p[8] & 1:
            groups[p[2:4]].append(p)
    results = []
    for task in verified["tasks"]:
        group, = (rows for rows in groups.values()
                  if len(rows) == task["grid"] and match(task, rows))
        if sorted(p[12] for p in group) != list(range(task["grid"])):
            raise ValueError("logical block IDs do not cover the launch")
        rows = [[p[12], p[9], p[25] - p[24], p[4], *p[14:24]] for p in group]
        results.append(dict(task=task["task"], phase=task["phase"], rows=sorted(rows)))
    return dict(case=case, metric=metric, events=sample["aiv_profiling_events"],
                raw_sha256=hashlib.sha256(data).hexdigest(), results=results,
                columns=["logical_block", "physical_core", "duration_ns", "total_cycles",
                         *[f"counter_{i}" for i in range(10)]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("case", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metric", choices=("PipeUtilization", "ResourceConflictRatio", "L2Cache", "Memory"),
                        default="PipeUtilization")
    args = parser.parse_args()
    record = extract(args.root, args.case, args.metric)
    with args.output.open("x") as stream:
        json.dump(record, stream, separators=(",", ":"))
