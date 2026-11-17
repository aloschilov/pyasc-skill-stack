"""Extract one restored unary kernel's task and physical-core timings from msprof 9.2."""

import argparse
import csv
import hashlib
import json
import sqlite3
import subprocess
from decimal import Decimal
from pathlib import Path


def export(root, case, kernel):
    receipt = json.loads((root / f"ok-{case}.json").read_text())
    phases = [phase for _, phase in receipt["checked"]]
    if phases != ["preflight"] + ["warmup"] * 3 + ["measured"] * 5:
        raise ValueError("unexpected restored-call sequence")
    profile, = (root / f"profile-{case}").glob("PROF*")
    db, = profile.glob("device_0/sqlite/metric_summary.db")
    summaries = sorted(profile.rglob("op_summary*.csv"))
    timelines = sorted(profile.rglob("msprof*.json"))
    with summaries[-1].open(newline="") as stream:
        tasks = [r for r in csv.DictReader(stream) if r["Op Name"].strip() == kernel]
    tasks.sort(key=lambda r: Decimal(r["Task Start Time(us)"].strip()))
    if len(tasks) != len(phases):
        raise ValueError(f"matched {len(tasks)} tasks for {len(phases)} invocations")
    events = json.loads(timelines[-1].read_text(), parse_float=Decimal)
    tracks = {x["pid"] for x in events if x.get("name") == "process_name"
              and x.get("args", {}).get("name") == "Block Detail"}
    if len(tracks) != 1:
        raise ValueError(f"unexpected Block Detail tracks: {tracks}")
    binaries = list(root.glob(f"d-{case}/binary.o"))
    binary, = binaries
    listing = subprocess.check_output(["readelf", "-x", ".text", binary])
    data = []
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        for r, phase in zip(tasks, phases):
            stream, task = int(r["Stream ID"]), int(r["Task ID"])
            start = Decimal(r["Task Start Time(us)"].strip())
            blocks = conn.execute(
                "select core_id, start_time, duration from V6BlockPmu "
                "where stream_id=? and task_id=? order by core_id, start_time",
                (stream, task),
            ).fetchall()
            detail = []
            for x in events:
                if x.get("pid") not in tracks or x.get("ph") != "X":
                    continue
                args = x["args"]
                if (args.get("Physic Stream Id"), args.get("Task Id")) == (stream, task):
                    detail.append([args["Core Id"], str(Decimal(x["ts"]) - start), str(x["dur"])])
            if not blocks or not detail:
                raise ValueError(f"missing timing data for {stream}/{task}")
            data.append(dict(phase=phase, stream=stream, task=task, start_us=str(start),
                             task_us=r["Task Duration(us)"].strip(), grid=int(r["Block Num"]),
                             pmu=blocks, timeline=sorted(detail)))
    return dict(receipt=receipt, text_dump_sha256=hashlib.sha256(listing).hexdigest(),
                pmu_fields=["physical_core", "raw_start", "duration_us"],
                timeline_fields=["physical_core", "offset_from_task_start_us", "display_duration_us"],
                tasks=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("case", type=int)
    parser.add_argument("kernel")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = export(args.root, args.case, args.kernel)
    with args.output.open("x") as output:
        json.dump(record, output, separators=(",", ":"))
