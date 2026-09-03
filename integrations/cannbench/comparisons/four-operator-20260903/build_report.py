#!/usr/bin/env python3
"""Build Markdown/HTML/PDF report from immutable local and remote evidence."""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
COMMIT = "030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d"
OPS = ("gelu", "rms_norm", "softmax", "transpose")
ARMS = ("handwritten", "no_skills", "with_skills")
LABELS = {
    "handwritten": "Handwritten target-derived",
    "no_skills": "Generated without skills",
    "with_skills": "Generated with skills",
}
JOBS_URL = "https://cannbench.com/workspace/jobs"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "canceled", "rejected", "timed_out", "timeout"}


def job_url(job_id: str) -> str:
    return f"{JOBS_URL}/{job_id}"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def remote_row(index: int, arm: str) -> dict:
    payload = read_json(ROOT / "remote_runs" / f"{index:02d}-{arm}-job.json")
    if not payload:
        submission = read_json(ROOT / "remote_runs" / f"{index:02d}-{arm}-submission.json")
        if not submission:
            return {"arm": arm, "status": "pending"}
        response = submission.get("response") or {}
        job = response.get("job") or {}
        return {
            "arm": arm,
            "status": job.get("status", "submitted"),
            "job_id": job.get("id") or submission.get("job_id"),
            "passed": None,
            "total": None,
            "score": None,
            "gmean": None,
            "anti_cheat": None,
            "operators": [],
        }
    job = payload.get("job", payload)
    results = job.get("results") or {}
    summary = results.get("summary") or {}
    return {
        "arm": arm,
        "status": job.get("status", "unknown"),
        "job_id": job.get("id"),
        "passed": summary.get("passed_cases", job.get("passed_cases")),
        "total": summary.get("total_cases", job.get("total_cases")),
        "score": summary.get("overall_score", job.get("result_score")),
        "gmean": summary.get("geometric_mean_speedup"),
        "anti_cheat": summary.get("anti_cheat_failed_cases"),
        "operators": results.get("operators") or [],
    }


def skill_summary() -> tuple[int, int, list[str]]:
    required = 0
    verified = 0
    sessions = []
    for op in OPS:
        data = read_json(ROOT / "generated" / "with_skills" / op / "provenance.json") or {}
        for phase in data.get("phases", []):
            req = set(phase.get("required_skills") or [])
            got = set(phase.get("verified_loaded_skills") or [])
            required += len(req)
            verified += len(req & got)
            if phase.get("session_id"):
                sessions.append(phase["session_id"])
    return required, verified, sessions


def generation_rows() -> list[list[str]]:
    rows = []
    for arm in ("no_skills", "with_skills"):
        for op in OPS:
            data = read_json(ROOT / "generated" / arm / op / "provenance.json") or {}
            phases = data.get("phases") or []
            models = ", ".join(str(p.get("model") or "—") for p in phases) or "pending"
            sessions = ", ".join(str(p.get("session_id") or "—") for p in phases) or "pending"
            calls = sum(len(p.get("skill_calls") or []) for p in phases)
            required = sum(len(p.get("required_skills") or []) for p in phases)
            verified = sum(len(set(p.get("required_skills") or []) &
                               set(p.get("verified_loaded_skills") or [])) for p in phases)
            rows.append([
                LABELS[arm], op, str(data.get("candidate_sha256") or "pending")[:16],
                models, sessions, f"{verified}/{required}", str(calls),
            ])
    return rows


def fmt(value, digits=6):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def bar_svg(rows: list[dict], key: str, title: str, max_value: float) -> str:
    colors = ("#8b5cf6", "#64748b", "#10b981")
    width, height = 760, 80 + 68 * len(rows)
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
              f'<text x="10" y="28" class="chart-title">{html.escape(title)}</text>']
    for i, (row, color) in enumerate(zip(rows, colors)):
        y = 52 + i * 68
        value = row.get(key)
        number = float(value or 0)
        bar = max(0, min(520, 520 * number / max_value)) if max_value else 0
        pieces += [
            f'<text x="10" y="{y + 23}" class="chart-label">{html.escape(LABELS[row["arm"]])}</text>',
            f'<rect x="205" y="{y}" width="520" height="30" rx="6" fill="#e2e8f0"/>',
            f'<rect x="205" y="{y}" width="{bar:.2f}" height="30" rx="6" fill="{color}"/>',
            f'<text x="735" y="{y + 21}" text-anchor="end" class="chart-value">{html.escape(fmt(value))}</text>',
        ]
    pieces.append("</svg>")
    return "".join(pieces)


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    local = read_json(ROOT / "local_validation_summary.json") or {"results": []}
    remote = [remote_row(i, arm) for i, arm in enumerate(ARMS, 1)]
    manifest = read_json(ROOT / "MANIFEST.json") or {}
    required, verified, sessions = skill_summary()
    provenance_rows = generation_rows()

    md = [
        "# CANNBench: four-operator three-arm comparison",
        "",
        "Date: 2026-09-03 (Europe/Moscow). Benchmark: `official-tasks` v1.1.1, private submissions, 950PR.",
        "",
        "## Scope and provenance",
        "",
        f"All arms use the same current pyasc `v2` snapshot `{COMMIT}` and the same four CANNBench operators: " + ", ".join(f"`{op}`" for op in OPS) + ".",
        "",
        "- handwritten target-derived: exact JIT/device definitions extracted from `python/test/asctile/target/test_<op>.py`; only host signature/allocation/launch transport is generated or assembled deterministically;",
        "- generated without skills: OpenCode workers have no skill paths and any skill call fails provenance acceptance;",
        "- generated with skills: the same generation/review structure, with native skill calls required and verified from OpenCode event traces.",
        "- controlled exceptions: skill-guided Softmax is a review/tuning pass over the 20/20 no-skill baseline and produced the same hash; both generated Transpose implementation workers exhausted their context before writing, so both comparison arms use the same target-derived, deterministically remediated rank-5/int64 candidate and differ only in their recorded review path. No Transpose delta may therefore be attributed to skills.",
        "",
        f"Verified required skill loads: **{verified}/{required}**. Accepted skill-arm sessions: " + (", ".join(f"`{s}`" for s in sessions) if sessions else "pending") + ".",
        "",
        "CANNBench receives only the built wheel and cannot infer generation history. Arm identity is therefore established by candidate hashes, source-definition hashes, prompts, model sessions, and native skill-call traces retained beside this report.",
        "",
        "### Generated-arm provenance",
        "",
        "| Arm | Operator | Candidate SHA-256 | Phase models | Session IDs | Verified/required skill loads | Skill calls |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in provenance_rows:
        md.append("| " + " | ".join(f"`{v}`" if i in {1, 2, 4} else v for i, v in enumerate(row)) + " |")
    md += [
        "",
        "## Local pinned-v2 compile gate",
        "",
        "| Arm | Operator | Dispatch | Compile | Status |",
        "|---|---|---:|---:|---|",
    ]
    for row in local.get("results", []):
        md.append(f"| {LABELS.get(row['variant'], row['variant'])} | `{row['operator']}` | {fmt(row.get('dispatch_passed'))}/{fmt(row.get('total_cases'))} | {fmt(row.get('compile_passed'))}/{fmt(row.get('total_cases'))} | {row.get('status', '—')} |")
    if not local.get("results"):
        md.append("| — | — | — | — | pending |")
    md += [
        "",
        "The QEMU gate checks host dispatch plus pyasc AST/codegen, lowering, AscendC translation, and 950PR UB limits. It does not execute numeric code or measure silicon latency.",
        "Worker prose that called a local compile result `verified-cannbench` is rejected by this report; only a completed remote job receives that label.",
        "",
        "## CANNBench results",
        "",
        "Run pages:",
        "",
    ]
    completed_links = [
        f"- {LABELS[row['arm']]}: [`{row['job_id']}`]({job_url(row['job_id'])})"
        for row in remote if row.get("job_id")
    ]
    md += completed_links or [
        f"- current series is waiting for submission credits; [CANNBench jobs workspace]({JOBS_URL}) will expose each run after submission."
    ]
    md += [
        "",
        "| Arm | Job | Cases | Score | GMean speedup | Anti-cheat | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in remote:
        cases = "—" if row.get("total") is None else f"{row.get('passed')}/{row.get('total')}"
        job_id = row.get("job_id")
        job_cell = f"[`{job_id}`]({job_url(job_id)})" if job_id else f"[pending]({JOBS_URL})"
        md.append(f"| {LABELS[row['arm']]} | {job_cell} | {cases} | {fmt(row.get('score'))} | {fmt(row.get('gmean'))} | {fmt(row.get('anti_cheat'))} | {row['status']} |")
    md += ["", "### Per-operator measurements", "", "| Arm | Operator | Cases | Average speedup | Score |", "|---|---|---:|---:|---:|"]
    any_ops = False
    for row in remote:
        for op in row.get("operators", []):
            any_ops = True
            md.append(f"| {LABELS[row['arm']]} | {op.get('operator')} | {op.get('passed_cases')}/{op.get('total_cases')} | {fmt(op.get('avg_speedup'))} | {fmt(op.get('score'))} |")
    if not any_ops:
        md.append("| — | — | — | — | pending |")
    md += [
        "",
        "## Interpretation rules",
        "",
        "Correctness is a hard gate: speedups from a partial pass must not be compared as if they represented complete solutions. The target-derived arm measures how far upstream development targets cover the broader CANNBench contract; it is not silently repaired into a fourth implementation. Generated arms are compared only after preserving their skill/no-skill provenance boundary.",
        "",
        "## Artifacts",
        "",
        "- `MANIFEST.json`: runtime and candidate hashes;",
        "- `generated/*/*/evidence`: prompts, events, accepted/rejected attempts, model sessions and skill traces;",
        "- `handwritten/*/provenance.json`: exact upstream source paths and definition hashes;",
        "- `*/local_validation`: QEMU gate reports;",
        "- `remote_runs`: CANNBench job payloads and logs;",
        "- `submissions/*.zip`: immutable submitted source archives.",
        "",
    ]
    md_text = "\n".join(md)
    (REPORTS / "four-operator-comparison.md").write_text(md_text, encoding="utf-8")

    def table_from_markdown(lines: list[str]) -> str:
        header = [c.strip() for c in lines[0].strip("|").split("|")]
        body = []
        for line in lines[2:]:
            body.append([c.strip().replace("`", "") for c in line.strip("|").split("|")])
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in header) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in body) + "</tbody></table>"

    # Render from structured data to keep the PDF self-contained and chart-rich.
    remote_table = [[LABELS[r["arm"]], r.get("job_id") or "pending",
                     "—" if r.get("total") is None else f'{r.get("passed")}/{r.get("total")}',
                     fmt(r.get("score")), fmt(r.get("gmean")), fmt(r.get("anti_cheat")), r["status"]] for r in remote]
    local_table = [[LABELS.get(r["variant"], r["variant"]), r["operator"],
                    f'{fmt(r.get("dispatch_passed"))}/{fmt(r.get("total_cases"))}',
                    f'{fmt(r.get("compile_passed"))}/{fmt(r.get("total_cases"))}', r.get("status", "—")] for r in local.get("results", [])]
    def html_table(headers, rows):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(v))}</th>" for v in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"

    chart_rows = [dict(r, passed=r.get("passed") or 0) for r in remote]
    run_links_html = "".join(
        f'<li>{html.escape(LABELS[r["arm"]])}: <a href="{html.escape(job_url(r["job_id"]))}"><code>{html.escape(r["job_id"])}</code></a></li>'
        for r in remote if r.get("job_id")
    )
    if not run_links_html:
        run_links_html = f'<li>Waiting for submission credits. Open the <a href="{JOBS_URL}">CANNBench jobs workspace</a> to follow the series after submission.</li>'
    html_text = f'''<!doctype html><html><head><meta charset="utf-8"><title>CANNBench four-operator comparison</title><style>
@page {{ size: A4; margin: 15mm; }}
body {{ font: 14px/1.48 Inter, Arial, sans-serif; color:#172033; max-width:1000px; margin:32px auto; padding:0 24px; }}
h1 {{ font-size:30px; margin-bottom:4px; }} h2 {{ margin-top:28px; color:#263a67; }}
.lede {{ color:#526078; }} .pin {{ background:#eef2ff; border-left:5px solid #6366f1; padding:14px 18px; border-radius:8px; }}
table {{ width:100%; border-collapse:collapse; margin:14px 0 22px; font-size:12px; }} th {{ background:#e8eef9; text-align:left; }} th,td {{ border:1px solid #cbd5e1; padding:7px; }}
svg {{ width:100%; height:auto; margin:10px 0; }} .chart-title {{ font-size:18px; font-weight:700; }} .chart-label,.chart-value {{ font-size:12px; }}
code {{ font-family:ui-monospace, monospace; }} li {{ margin:5px 0; }} .pending {{ color:#a16207; }}
</style></head><body>
<h1>CANNBench: four operators, three arms</h1><p class="lede">2026-09-03 · official-tasks v1.1.1 · private 950PR runs</p>
<div class="pin"><strong>Runtime pin:</strong> compiler-team/pyasc <code>v2@{COMMIT}</code><br><strong>Operators:</strong> {', '.join(OPS)}<br><strong>Verified skill loads:</strong> {verified}/{required}</div>
<h2>Experimental design</h2><p>Handwritten target-derived kernels are copied from <code>python/test/asctile/target</code> with only generated or deterministic host transport adaptation. The no-skill arm exposes no skill path and rejects skill calls. The with-skills arm requires native OpenCode skill calls in every accepted phase. CANNBench sees only the resulting wheel, so repository evidence establishes generation provenance.</p><p><strong>Controlled exceptions:</strong> skill-guided Softmax reviewed the no-skill 20/20 baseline and retained the same hash. Transpose generation repeatedly exhausted model context before writing; both comparison arms therefore use the same target-derived candidate with deterministic rank-5/int64 remediation and differ only in their recorded review path. A Transpose performance delta cannot be attributed to skills.</p>
<h3>Generated-arm provenance</h3>{html_table(['Arm','Operator','Candidate hash','Phase models','Session IDs','Verified/required','Skill calls'], provenance_rows)}
<h2>Local compile gate</h2>{html_table(['Arm','Operator','Dispatch','Compile','Status'], local_table or [['—','—','—','—','pending']])}<p>The QEMU gate checks dispatch, pyasc lowering, AscendC translation and UB limits, but not numerical execution or latency. Any worker prose that called this <code>verified-cannbench</code> is rejected; only a completed remote job receives that label.</p>
<h2>CANNBench results</h2><p><strong>Run pages</strong></p><ul>{run_links_html}</ul>{html_table(['Arm','Job','Cases','Score','GMean','Anti-cheat','Status'], remote_table)}
{bar_svg(chart_rows, 'passed', 'Passed cases (maximum 80)', 80)}
{bar_svg(remote, 'gmean', 'Geometric-mean speedup', max([float(r.get('gmean') or 0) for r in remote] + [1.0]))}
<h2>Interpretation</h2><p>Correctness is a hard gate: partial-pass speedups are not comparable with complete solutions. The target-derived arm measures upstream target coverage of the wider CANNBench contract; it is not silently rewritten. Generated arms retain separate skill/no-skill provenance.</p>
<h2>Evidence</h2><ul><li><code>MANIFEST.json</code> — hashes and runtime pin</li><li><code>generated/*/*/evidence</code> — worker events, sessions and skill traces</li><li><code>handwritten/*/provenance.json</code> — source definition hashes</li><li><code>*/local_validation</code> — QEMU reports</li><li><code>remote_runs</code> — jobs and logs</li><li><code>submissions/*.zip</code> — immutable archives</li></ul>
</body></html>'''
    html_path = REPORTS / "four-operator-comparison.html"
    pdf_path = REPORTS / "four-operator-comparison.pdf"
    html_path.write_text(html_text, encoding="utf-8")
    browser = next((p for p in ("/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/snap/bin/chromium") if Path(p).exists()), None)
    if browser:
        subprocess.run([browser, "--headless", "--no-sandbox", "--disable-gpu", f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=False, timeout=120)
    print(json.dumps({"markdown": str(REPORTS / 'four-operator-comparison.md'), "html": str(html_path), "pdf": str(pdf_path) if pdf_path.exists() else None, "remote_complete": all(r['status'] in TERMINAL_STATUSES for r in remote)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
