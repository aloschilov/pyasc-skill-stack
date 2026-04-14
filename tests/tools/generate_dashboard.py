#!/usr/bin/env python3
"""Generate an interactive HTML dashboard from capabilities.yaml and evidence/*.json.

Reads the capabilities matrix and any evidence artifacts, then produces a
self-contained index.html with embedded data, interactive table, filters,
and expand/collapse evidence details.

Usage:
    python generate_dashboard.py [--output-dir _site]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CAPABILITIES_FILE = REPO_ROOT / "capabilities.yaml"
EVIDENCE_DIR = REPO_ROOT / "evidence"


def _load_yaml(path: Path) -> dict:
    if yaml is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    import subprocess
    result = subprocess.run(
        ["python3", "-c",
         f"import yaml,json; print(json.dumps(yaml.safe_load(open('{path}'))))"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        return json.loads(result.stdout)
    sys.stderr.write("ERROR: PyYAML is required. pip install pyyaml\n")
    sys.exit(2)


def _load_evidence(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_data(cap: dict) -> dict:
    classes = cap.get("classes", {})
    operations = cap.get("operations", [])

    counts = {
        "golden": {"confirmed": 0, "golden_only": 0, "pending": 0,
                    "claimed": 0, "untested": 0, "blocked": 0},
        "generative": {"confirmed": 0, "pending": 0, "claimed": 0,
                        "untested": 0, "blocked": 0},
    }

    all_dtypes: list[str] = []
    rows: list[dict] = []

    for op in operations:
        op_name = op.get("name", "?")
        op_class = op.get("class", "?")
        asc2_api = op.get("asc2_api", "")

        for cell in op.get("cells", []):
            dtype = cell.get("dtype", "?")
            if dtype not in all_dtypes:
                all_dtypes.append(dtype)

            gs = cell.get("golden_status", "untested")
            gen_s = cell.get("generative_status", "untested")
            counts["golden"][gs] = counts["golden"].get(gs, 0) + 1
            counts["generative"][gen_s] = counts["generative"].get(gen_s, 0) + 1

            golden_ev = None
            gen_ev = None
            ge_ref = cell.get("golden_evidence")
            if ge_ref:
                golden_ev = _load_evidence(REPO_ROOT / ge_ref)
            gn_ref = cell.get("generative_evidence")
            if gn_ref:
                gen_ev = _load_evidence(REPO_ROOT / gn_ref)

            row: dict = {
                "op": op_name,
                "class": op_class,
                "asc2_api": asc2_api,
                "dtype": dtype,
                "golden_status": gs,
                "generative_status": gen_s,
            }

            if golden_ev:
                row["golden_evidence"] = {
                    "date": golden_ev.get("date", ""),
                    "score": golden_ev.get("score", {}).get("value", 0),
                    "verification_mode": golden_ev.get("verification", {}).get("mode", ""),
                    "verification_status": golden_ev.get("verification", {}).get("status", ""),
                    "shapes": golden_ev.get("verification", {}).get("shapes_verified", []),
                    "notes": golden_ev.get("notes", ""),
                    "kernel_path": golden_ev.get("kernel_path", ""),
                }

            if gen_ev:
                row["generative_evidence"] = {
                    "date": gen_ev.get("date", ""),
                    "score": gen_ev.get("score", {}).get("value", 0),
                    "verification_mode": gen_ev.get("verification", {}).get("mode", ""),
                    "verification_status": gen_ev.get("verification", {}).get("status", ""),
                    "shapes": gen_ev.get("verification", {}).get("shapes_verified", []),
                    "notes": gen_ev.get("notes", ""),
                    "kernel_path": gen_ev.get("kernel_path", ""),
                    "agent_platform": gen_ev.get("agent", {}).get("platform", ""),
                    "agent_completed": gen_ev.get("agent", {}).get("completed", False),
                    "artifacts": gen_ev.get("agent", {}).get("artifacts_found", []),
                }

            rows.append(row)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classes": classes,
        "counts": counts,
        "dtypes": all_dtypes,
        "total_cells": len(rows),
        "rows": rows,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pyasc-skill-stack Capabilities</title>
<style>
:root {
  --bg: #ffffff;
  --bg-alt: #f6f8fa;
  --fg: #1f2328;
  --fg-muted: #656d76;
  --border: #d0d7de;
  --confirmed: #1a7f37;
  --confirmed-bg: #dafbe1;
  --golden-only: #0969da;
  --golden-only-bg: #ddf4ff;
  --pending: #9a6700;
  --pending-bg: #fff8c5;
  --claimed: #bf8700;
  --claimed-bg: #fff1cc;
  --untested: #656d76;
  --untested-bg: #eaeef2;
  --blocked: #cf222e;
  --blocked-bg: #ffebe9;
  --radius: 6px;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117;
    --bg-alt: #161b22;
    --fg: #e6edf3;
    --fg-muted: #8b949e;
    --border: #30363d;
    --confirmed: #3fb950;
    --confirmed-bg: #0d2818;
    --golden-only: #58a6ff;
    --golden-only-bg: #0c2d6b;
    --pending: #d29922;
    --pending-bg: #3b2300;
    --claimed: #d29922;
    --claimed-bg: #3b2300;
    --untested: #8b949e;
    --untested-bg: #21262d;
    --blocked: #f85149;
    --blocked-bg: #3d1214;
    --shadow: 0 1px 3px rgba(0,0,0,0.3);
  }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.5;
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}
h1 { font-size: 24px; margin-bottom: 8px; }
.subtitle { color: var(--fg-muted); margin-bottom: 24px; font-size: 14px; }
.summary {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 24px;
}
.summary-card {
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 16px;
  min-width: 120px;
  box-shadow: var(--shadow);
}
.summary-card .label { font-size: 11px; text-transform: uppercase; color: var(--fg-muted); letter-spacing: 0.5px; }
.summary-card .value { font-size: 28px; font-weight: 600; }
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 16px;
  align-items: center;
}
.controls label { font-size: 13px; color: var(--fg-muted); }
.controls select {
  padding: 4px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg);
  color: var(--fg);
  font-size: 13px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}
thead th {
  background: var(--bg-alt);
  border-bottom: 2px solid var(--border);
  padding: 8px 12px;
  text-align: left;
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  color: var(--fg-muted);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}
thead th:hover { color: var(--fg); }
thead th .sort-arrow { font-size: 10px; margin-left: 4px; }
tbody tr { border-bottom: 1px solid var(--border); }
tbody tr:hover { background: var(--bg-alt); }
tbody td { padding: 6px 12px; vertical-align: middle; }
td.op-name { font-weight: 600; }
td.api-col { font-family: "SFMono-Regular", Consolas, monospace; font-size: 13px; color: var(--fg-muted); }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
  cursor: default;
  white-space: nowrap;
}
.badge-confirmed { background: var(--confirmed-bg); color: var(--confirmed); }
.badge-golden_only { background: var(--golden-only-bg); color: var(--golden-only); }
.badge-pending { background: var(--pending-bg); color: var(--pending); }
.badge-claimed { background: var(--claimed-bg); color: var(--claimed); }
.badge-untested { background: var(--untested-bg); color: var(--untested); }
.badge-blocked { background: var(--blocked-bg); color: var(--blocked); }
.badge.clickable { cursor: pointer; text-decoration: underline; text-decoration-style: dotted; }
.detail-panel {
  display: none;
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 16px;
  margin: 4px 0 8px;
  font-size: 13px;
  line-height: 1.6;
}
.detail-panel.open { display: block; }
.detail-panel dt { font-weight: 600; display: inline; }
.detail-panel dt::after { content: ": "; }
.detail-panel dd { display: inline; margin: 0; }
.detail-panel dd::after { content: ""; display: block; }
.class-header td {
  background: var(--bg-alt);
  font-weight: 600;
  font-size: 13px;
  padding: 10px 12px;
  color: var(--fg-muted);
  border-bottom: 2px solid var(--border);
}
footer {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--fg-muted);
}
footer a { color: var(--fg-muted); }
</style>
</head>
<body>

<h1>pyasc-skill-stack Capabilities</h1>
<p class="subtitle">Auto-generated from <code>capabilities.yaml</code> and <code>evidence/*.json</code></p>

<div class="summary" id="summary"></div>

<div class="controls">
  <label>Class:
    <select id="filter-class">
      <option value="all">All</option>
    </select>
  </label>
  <label>Status:
    <select id="filter-status">
      <option value="all">All</option>
      <option value="confirmed">Confirmed</option>
      <option value="golden_only">Golden only</option>
      <option value="pending">Pending</option>
      <option value="claimed">Claimed</option>
      <option value="untested">Untested</option>
      <option value="blocked">Blocked</option>
    </select>
  </label>
  <label>Dimension:
    <select id="filter-dimension">
      <option value="any">Any (golden or gen)</option>
      <option value="golden">Golden only</option>
      <option value="generative">Generative only</option>
    </select>
  </label>
</div>

<table>
  <thead>
    <tr id="thead-row"></tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>

<footer>
  Generated <span id="gen-time"></span> &mdash;
  <a href="https://github.com/aloschilov/pyasc-skill-stack">pyasc-skill-stack</a>
</footer>

<script>
const DATA = __DATA_PLACEHOLDER__;

function init() {
  document.getElementById("gen-time").textContent = DATA.generated_at;
  renderSummary();
  populateClassFilter();
  renderTable();
  document.getElementById("filter-class").addEventListener("change", renderTable);
  document.getElementById("filter-status").addEventListener("change", renderTable);
  document.getElementById("filter-dimension").addEventListener("change", renderTable);
}

function renderSummary() {
  const el = document.getElementById("summary");
  const gc = DATA.counts.golden;
  const gn = DATA.counts.generative;
  const cards = [
    { label: "Total cells", value: DATA.total_cells },
    { label: "Golden confirmed", value: gc.confirmed || 0 },
    { label: "Golden only", value: gc.golden_only || 0 },
    { label: "Gen confirmed", value: gn.confirmed || 0 },
    { label: "Gen pending", value: gn.pending || 0 },
    { label: "Claimed", value: gc.claimed || 0 },
    { label: "Untested", value: (gc.untested || 0) + (gn.untested || 0) },
  ];
  el.innerHTML = cards.map(c =>
    `<div class="summary-card"><div class="label">${c.label}</div><div class="value">${c.value}</div></div>`
  ).join("");
}

function populateClassFilter() {
  const sel = document.getElementById("filter-class");
  const classes = Object.keys(DATA.classes);
  classes.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c.replace(/_/g, " ");
    sel.appendChild(opt);
  });
}

function makeBadge(status, evidence, kind) {
  const label = status.replace(/_/g, " ");
  const hasEvidence = !!evidence;
  const cls = hasEvidence ? "badge badge-" + status + " clickable" : "badge badge-" + status;
  const dataAttr = hasEvidence ? ` data-kind="${kind}" data-detail='${JSON.stringify(evidence).replace(/'/g, "&#39;")}'` : "";
  return `<span class="${cls}"${dataAttr} onclick="toggleDetail(this)">${label}</span>`;
}

function toggleDetail(el) {
  const detailStr = el.getAttribute("data-detail");
  if (!detailStr) return;
  let panel = el.parentElement.querySelector(".detail-panel");
  if (panel) {
    panel.classList.toggle("open");
    return;
  }
  const d = JSON.parse(detailStr);
  const kind = el.getAttribute("data-kind");
  let html = "<dl>";
  if (d.date) html += `<dt>Date</dt><dd>${d.date}</dd>`;
  if (d.score !== undefined) html += `<dt>Score</dt><dd>${d.score}/10</dd>`;
  if (d.verification_mode) html += `<dt>Verification</dt><dd>${d.verification_mode} (${d.verification_status})</dd>`;
  if (d.shapes && d.shapes.length) html += `<dt>Shapes</dt><dd>${JSON.stringify(d.shapes)}</dd>`;
  if (d.kernel_path) html += `<dt>Kernel</dt><dd><code>${d.kernel_path}</code></dd>`;
  if (kind === "generative") {
    if (d.agent_platform) html += `<dt>Agent</dt><dd>${d.agent_platform}${d.agent_completed ? " (completed)" : " (incomplete)"}</dd>`;
    if (d.artifacts && d.artifacts.length) html += `<dt>Artifacts</dt><dd>${d.artifacts.join(", ")}</dd>`;
  }
  if (d.notes) html += `<dt>Notes</dt><dd>${d.notes}</dd>`;
  html += "</dl>";
  panel = document.createElement("div");
  panel.className = "detail-panel open";
  panel.innerHTML = html;
  el.parentElement.appendChild(panel);
}

let sortCol = null;
let sortAsc = true;

function sortBy(col) {
  if (sortCol === col) { sortAsc = !sortAsc; }
  else { sortCol = col; sortAsc = true; }
  renderTable();
}

function renderTable() {
  const filterClass = document.getElementById("filter-class").value;
  const filterStatus = document.getElementById("filter-status").value;
  const filterDim = document.getElementById("filter-dimension").value;

  let rows = DATA.rows.filter(r => {
    if (filterClass !== "all" && r["class"] !== filterClass) return false;
    if (filterStatus !== "all") {
      if (filterDim === "golden") return r.golden_status === filterStatus;
      if (filterDim === "generative") return r.generative_status === filterStatus;
      return r.golden_status === filterStatus || r.generative_status === filterStatus;
    }
    return true;
  });

  if (sortCol) {
    rows.sort((a, b) => {
      let va = a[sortCol] || "";
      let vb = b[sortCol] || "";
      if (typeof va === "string") va = va.toLowerCase();
      if (typeof vb === "string") vb = vb.toLowerCase();
      if (va < vb) return sortAsc ? -1 : 1;
      if (va > vb) return sortAsc ? 1 : -1;
      return 0;
    });
  }

  const cols = [
    { key: "op", label: "Operation" },
    { key: "class", label: "Class" },
    { key: "dtype", label: "dtype" },
    { key: "asc2_api", label: "asc2 API" },
    { key: "golden_status", label: "Golden" },
    { key: "generative_status", label: "Generative" },
  ];

  const thead = document.getElementById("thead-row");
  thead.innerHTML = cols.map(c => {
    const arrow = sortCol === c.key ? (sortAsc ? " &#9650;" : " &#9660;") : "";
    return `<th onclick="sortBy('${c.key}')">${c.label}<span class="sort-arrow">${arrow}</span></th>`;
  }).join("");

  const tbody = document.getElementById("tbody");
  let html = "";
  let lastClass = null;

  for (const r of rows) {
    if (r["class"] !== lastClass) {
      lastClass = r["class"];
      const classInfo = DATA.classes[lastClass] || {};
      const desc = classInfo.description || "";
      const pattern = classInfo.pattern || "";
      html += `<tr class="class-header"><td colspan="${cols.length}">${lastClass.replace(/_/g, " ")} &mdash; <code>${pattern}</code> &mdash; ${desc}</td></tr>`;
    }
    html += "<tr>";
    html += `<td class="op-name">${r.op}</td>`;
    html += `<td>${r["class"].replace(/_/g, " ")}</td>`;
    html += `<td>${r.dtype}</td>`;
    html += `<td class="api-col">${r.asc2_api}</td>`;
    html += `<td>${makeBadge(r.golden_status, r.golden_evidence, "golden")}</td>`;
    html += `<td>${makeBadge(r.generative_status, r.generative_evidence, "generative")}</td>`;
    html += "</tr>";
  }

  tbody.innerHTML = html;
}

init();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate capabilities dashboard HTML.")
    parser.add_argument("--output-dir", default="_site", help="Output directory (default: _site)")
    args = parser.parse_args()

    if not CAPABILITIES_FILE.exists():
        print(f"ERROR: {CAPABILITIES_FILE} not found", file=sys.stderr)
        sys.exit(1)

    cap = _load_yaml(CAPABILITIES_FILE)
    data = build_data(cap)

    data_json = json.dumps(data, indent=None, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    print(f"Dashboard written to {out_dir / 'index.html'}")
    print(f"  {data['total_cells']} cells, {len(data['rows'])} rows")
    golden_confirmed = data["counts"]["golden"].get("confirmed", 0)
    gen_confirmed = data["counts"]["generative"].get("confirmed", 0)
    print(f"  Golden confirmed: {golden_confirmed}, Generative confirmed: {gen_confirmed}")


if __name__ == "__main__":
    main()
