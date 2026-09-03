"""Definite-prompt builder for opencode/GLM-5.2 CANN Bench workers.

Renders self-contained prompts from templates/ + vendored task specs +
archived harness reports. A worker gets everything it needs in one message:
spec files, the pyasc constraints digest, a proven reference module, and
(for tuning) measured per-case timings.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

WORKERS_DIR = Path(__file__).resolve().parent
CANNBENCH_DIR = WORKERS_DIR.parent
TASKS_DIR = CANNBENCH_DIR / "tasks"
SUBMISSION_PKG = CANNBENCH_DIR / "submission" / "cann_bench"
EVIDENCE_DIR = CANNBENCH_DIR.parent.parent / "evidence" / "cannbench"
TEMPLATES = WORKERS_DIR / "templates"


def _render(template_name: str, tokens: dict) -> str:
    text = (TEMPLATES / template_name).read_text()
    for key, value in tokens.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def _strip_license_header(text: str) -> str:
    lines = text.splitlines()
    out, in_header = [], False
    for line in lines:
        if line.startswith("# ---------------"):
            in_header = not in_header
            continue
        if in_header or line.startswith("# Copyright"):
            continue
        out.append(line)
    return "\n".join(out).strip()


def cases_summary(op: str) -> tuple[str, int]:
    cases = yaml.safe_load((TASKS_DIR / op / "cases.yaml").read_text())["cases"]
    rows = ["| case | shapes | dtype | value_range | attrs |",
            "|---|---|---|---|---|"]
    for c in cases:
        rows.append(
            f"| {c['case_id']} | {c['input_shape']} | {c['dtype']} "
            f"| {c.get('value_range')} | {c.get('attrs') or '-'} |")
    return "\n".join(rows), len(cases)


def timings_table(op: str) -> tuple[str, str, str]:
    """Per-case timing table, avg speedup, and score summary from the
    archived final-eval report for *op*."""
    report = json.loads((EVIDENCE_DIR / f"{op}_final_eval.json").read_text())
    o = report["operators"][0]
    cases = yaml.safe_load((TASKS_DIR / op / "cases.yaml").read_text())["cases"]
    rel_path = o.get("rel_path", "")
    level = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
    if not level:
        proto = yaml.safe_load((TASKS_DIR / op / "proto.yaml").read_text())
        difficulty = str(proto["operator"].get("difficulty", "L1")).lower()
        level = difficulty.replace("l", "level", 1)
    meta = {f"{level}/{op}_{c['case_id']}": c for c in cases}
    rows = ["| case | shape | dtype | elapsed_us | baseline_us | t_hw_us | speedup |",
            "|---|---|---|---|---|---|---|"]
    for c in o["cases"]:
        m = meta.get(c["case_id"], {})
        rows.append(
            f"| {c['case_id'].split('/')[-1]} | {m.get('input_shape')} "
            f"| {m.get('dtype')} | {c.get('elapsed_us')} "
            f"| {c.get('baseline_perf_us')} | {c.get('t_hw_us')} "
            f"| {round(c.get('speedup') or 0, 3)}x |")
    score_summary = (
        f"harness score {round(o['score'], 2)}/100: compile "
        f"{o['compile_runtime_score']}/20, accuracy {o['function_score']}/30, "
        f"performance {round(o['performance_score'], 2)}/50")
    return "\n".join(rows), f"{round(o['avg_speedup'], 3)}", score_summary


def build_generation_prompt(op: str, callable_name: str, module_name: str,
                            guidance: str) -> str:
    op_dir = TASKS_DIR / op
    summary, n_cases = cases_summary(op)
    proto = yaml.safe_load((op_dir / "proto.yaml").read_text())
    return _render("generation.md", {
        "OP_NAME": proto["operator"]["name"],
        "CALLABLE": callable_name,
        "N_CASES": n_cases,
        "DESC": _strip_license_header((op_dir / "desc.md").read_text()),
        "PROTO": _strip_license_header((op_dir / "proto.yaml").read_text()),
        "GOLDEN": _strip_license_header((op_dir / "golden.py").read_text()),
        "CASES_SUMMARY": summary,
        # The canonical module was authored for the pre-rename asc2 snapshot.
        # The current v2 API is source-compatible under the asctile package
        # name, so translate this structural reference before showing it to a
        # worker. Task files and the pinned source remain authoritative.
        "REFERENCE_MODULE": (SUBMISSION_PKG / "sigmoid.py").read_text().replace(
            "asc2", "asctile"
        ),
        "CONSTRAINTS": (TEMPLATES / "constraints.md").read_text(),
        "GUIDANCE": guidance,
    })


def build_tuning_prompt(op: str, callable_name: str, module_name: str,
                        extra_levers: str = "") -> str:
    _, n_cases = cases_summary(op)
    table, avg_speedup, score_summary = timings_table(op)
    return _render("perf_tuning.md", {
        "OP_NAME": op,
        "CALLABLE": callable_name,
        "N_CASES": n_cases,
        "AVG_SPEEDUP": avg_speedup,
        "SCORE_SUMMARY": score_summary,
        "CURRENT_SOURCE": (SUBMISSION_PKG / f"{module_name}.py").read_text(),
        "TIMINGS_TABLE": table,
        "EXTRA_LEVERS": extra_levers,
        "CONSTRAINTS": (TEMPLATES / "constraints.md").read_text(),
    })


def build_feedback(digest: dict) -> str:
    """Render an eval-result digest into a follow-up worker message."""
    lines = ["The harness evaluated your candidate.py on the NPU. Result:"]
    if digest.get("hard_failure"):
        lines.append("")
        lines.append("It FAILED before producing a report. Harness log tail:")
        lines.append("```")
        lines.append(digest["log_tail"])
        lines.append("```")
    else:
        lines.append(
            f"- score {digest['score']}/100 (compile {digest['compile']}/20, "
            f"accuracy {digest['accuracy']}/30, perf {digest['perf']}/50)")
        lines.append(
            f"- {digest['passed']}/{digest['total']} cases passed, "
            f"avg speedup {digest['avg_speedup']}x")
        if digest.get("failed_cases"):
            lines.append("")
            lines.append("Failed cases:")
            for fc in digest["failed_cases"]:
                lines.append(f"- {fc['case_id']}: {fc['error']}")
        if digest.get("timings"):
            lines.append("")
            lines.append("Per-case timings (elapsed_us / baseline_us / speedup):")
            for t in digest["timings"]:
                lines.append(
                    f"- {t['case_id']}: {t['elapsed_us']} / "
                    f"{t['baseline_us']} / {t['speedup']}x")
    if digest.get("instruction"):
        lines.append("")
        lines.append(digest["instruction"])
    else:
        lines.append("")
        lines.append(
            "Improve candidate.py accordingly and overwrite it in place. "
            "Keep the public callable name/signature identical. Remember: "
            "correctness first — a failed case costs more than any speedup "
            "gains.")
    return "\n".join(lines)
