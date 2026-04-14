#!/usr/bin/env python3
"""Generate an interactive HTML dashboard from capabilities.yaml (v3) and evidence/*.json.

Reads the tier-based capabilities matrix and any evidence artifacts, then
produces a self-contained index.html with embedded data, tier progress bars,
representative_of tags, prompt display, and expand/collapse evidence details.

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
    tiers = cap.get("tiers", {})
    operations = cap.get("operations", [])

    tier_progress: dict[str, dict] = {}
    for t_name, t_info in tiers.items():
        tier_progress[t_name] = {
            "level": t_info.get("level", 0),
            "description": t_info.get("description", ""),
            "note": t_info.get("note", ""),
            "golden_confirmed": 0,
            "generative_confirmed": 0,
            "total_cells": 0,
        }

    all_dtypes: list[str] = []
    rows: list[dict] = []

    for op in operations:
        op_name = op.get("name", "?")
        op_tier = op.get("tier", "?")
        asc2_api = op.get("asc2_api", "")
        representative_of = op.get("representative_of", [])
        note = op.get("note", "")

        for cell in op.get("cells", []):
            dtype = cell.get("dtype", "?")
            if dtype not in all_dtypes:
                all_dtypes.append(dtype)

            gs = cell.get("golden_status", "untested")
            gen_s = cell.get("generative_status", "untested")
            prompt = cell.get("prompt", "")

            if op_tier in tier_progress:
                tier_progress[op_tier]["total_cells"] += 1
                if gs == "confirmed":
                    tier_progress[op_tier]["golden_confirmed"] += 1
                if gen_s == "confirmed":
                    tier_progress[op_tier]["generative_confirmed"] += 1

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
                "tier": op_tier,
                "asc2_api": asc2_api,
                "dtype": dtype,
                "golden_status": gs,
                "generative_status": gen_s,
                "representative_of": representative_of,
                "note": note,
                "prompt": prompt,
            }

            if golden_ev:
                gv_status = golden_ev.get("verification", {}).get("status", "")
                row["golden_evidence"] = {
                    "date": golden_ev.get("date", ""),
                    "score": golden_ev.get("score", {}).get("value", 0),
                    "verification_mode": golden_ev.get("verification", {}).get("mode", ""),
                    "verification_status": gv_status,
                    "runtime_verified": gv_status == "pass",
                    "shapes": golden_ev.get("verification", {}).get("shapes_verified", []),
                    "notes": golden_ev.get("notes", ""),
                    "kernel_path": golden_ev.get("kernel_path", ""),
                }

            if gen_ev:
                v_status = gen_ev.get("verification", {}).get("status", "")
                row["generative_evidence"] = {
                    "date": gen_ev.get("date", ""),
                    "score": gen_ev.get("score", {}).get("value", 0),
                    "verification_mode": gen_ev.get("verification", {}).get("mode", ""),
                    "verification_status": v_status,
                    "runtime_verified": v_status == "pass",
                    "shapes": gen_ev.get("verification", {}).get("shapes_verified", []),
                    "notes": gen_ev.get("notes", ""),
                    "kernel_path": gen_ev.get("kernel_path", ""),
                    "agent_platform": gen_ev.get("agent", {}).get("platform", ""),
                    "agent_completed": gen_ev.get("agent", {}).get("completed", False),
                    "artifacts": gen_ev.get("agent", {}).get("artifacts_found", []),
                    "semantic_check": gen_ev.get("semantic_check", {}),
                }

            rows.append(row)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tiers": tier_progress,
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
  --accent: #0969da;
  --bar-bg: #eaeef2;
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
    --accent: #58a6ff;
    --bar-bg: #21262d;
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
h1 { font-size: 24px; margin-bottom: 4px; }
.subtitle { color: var(--fg-muted); margin-bottom: 24px; font-size: 14px; }
.tier-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.tier-card {
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 16px;
  box-shadow: var(--shadow);
}
.tier-card .tier-title {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 2px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.tier-card .tier-level {
  display: inline-block;
  background: var(--accent);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  padding: 1px 6px;
  border-radius: 10px;
}
.tier-card .tier-desc { font-size: 12px; color: var(--fg-muted); margin-bottom: 8px; }
.progress-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
  font-size: 12px;
}
.progress-row .prog-label { min-width: 80px; color: var(--fg-muted); }
.progress-bar {
  flex: 1;
  height: 8px;
  background: var(--bar-bg);
  border-radius: 4px;
  overflow: hidden;
}
.progress-bar .fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
}
.progress-bar .fill-golden { background: var(--golden-only); }
.progress-bar .fill-gen { background: var(--confirmed); }
.progress-row .prog-count { min-width: 30px; text-align: right; font-weight: 600; }
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
tbody td { padding: 6px 12px; vertical-align: top; }
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
.rep-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 4px;
}
.rep-tag {
  display: inline-block;
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 10px;
  background: var(--bg-alt);
  border: 1px solid var(--border);
  color: var(--fg-muted);
}
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
.detail-panel .prompt-text {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px 10px;
  font-size: 12px;
  margin: 4px 0;
  font-style: italic;
  color: var(--fg-muted);
}
.tier-header td {
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
<p class="subtitle">Auto-generated from <code>capabilities.yaml</code> (v3) and <code>evidence/*.json</code></p>

<div class="tier-cards" id="tier-cards"></div>

<div class="controls">
  <label>Tier:
    <select id="filter-tier">
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

const TIER_ORDER = Object.entries(DATA.tiers)
  .sort((a, b) => a[1].level - b[1].level)
  .map(e => e[0]);

function init() {
  document.getElementById("gen-time").textContent = DATA.generated_at;
  renderTierCards();
  populateTierFilter();
  renderTable();
  document.getElementById("filter-tier").addEventListener("change", renderTable);
  document.getElementById("filter-status").addEventListener("change", renderTable);
  document.getElementById("filter-dimension").addEventListener("change", renderTable);
}

function renderTierCards() {
  const el = document.getElementById("tier-cards");
  let html = "";
  for (const tName of TIER_ORDER) {
    const t = DATA.tiers[tName];
    const goldenPct = t.total_cells ? Math.round(100 * t.golden_confirmed / t.total_cells) : 0;
    const genPct = t.total_cells ? Math.round(100 * t.generative_confirmed / t.total_cells) : 0;
    html += `<div class="tier-card">
      <div class="tier-title"><span class="tier-level">${t.level}</span> ${tName.replace(/_/g, " ")}</div>
      <div class="tier-desc">${t.description}</div>
      <div class="progress-row">
        <span class="prog-label">Golden</span>
        <div class="progress-bar"><div class="fill fill-golden" style="width:${goldenPct}%"></div></div>
        <span class="prog-count">${t.golden_confirmed}/${t.total_cells}</span>
      </div>
      <div class="progress-row">
        <span class="prog-label">Generative</span>
        <div class="progress-bar"><div class="fill fill-gen" style="width:${genPct}%"></div></div>
        <span class="prog-count">${t.generative_confirmed}/${t.total_cells}</span>
      </div>
    </div>`;
  }
  el.innerHTML = html;
}

function populateTierFilter() {
  const sel = document.getElementById("filter-tier");
  TIER_ORDER.forEach(t => {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = `Tier ${DATA.tiers[t].level}: ${t.replace(/_/g, " ")}`;
    sel.appendChild(opt);
  });
}

function makeBadge(status, evidence, kind, prompt) {
  const label = status.replace(/_/g, " ");
  const hasEvidence = !!evidence;
  const hasPrompt = !!prompt;
  const isClickable = hasEvidence || hasPrompt;
  const cls = isClickable ? "badge badge-" + status + " clickable" : "badge badge-" + status;
  let dataAttr = "";
  if (hasEvidence) dataAttr += ` data-detail='${JSON.stringify(evidence).replace(/'/g, "&#39;")}'`;
  if (hasPrompt) dataAttr += ` data-prompt='${prompt.replace(/'/g, "&#39;")}'`;
  dataAttr += ` data-kind="${kind}"`;
  let rtIcon = "";
  if (hasEvidence && status === "confirmed") {
    if (evidence.runtime_verified) {
      rtIcon = ' <span title="Runtime verified on simulator" style="font-size:11px;color:var(--confirmed);">&#9745;</span>';
    } else {
      rtIcon = ' <span title="Static only (runtime skipped)" style="font-size:11px;color:var(--pending);">&#9744;</span>';
    }
  }
  return `<span class="${cls}"${dataAttr} onclick="toggleDetail(this)">${label}</span>${rtIcon}`;
}

function toggleDetail(el) {
  let panel = el.parentElement.querySelector(".detail-panel");
  if (panel) {
    panel.classList.toggle("open");
    return;
  }
  const detailStr = el.getAttribute("data-detail");
  const promptStr = el.getAttribute("data-prompt");
  const kind = el.getAttribute("data-kind");
  if (!detailStr && !promptStr) return;

  let html = "<dl>";
  if (promptStr) {
    html += `<dt>Prompt</dt><dd><div class="prompt-text">${promptStr}</div></dd>`;
  }
  if (detailStr) {
    const d = JSON.parse(detailStr);
    if (d.date) html += `<dt>Date</dt><dd>${d.date}</dd>`;
    if (d.score !== undefined) html += `<dt>Score</dt><dd>${d.score}/10</dd>`;
    if (d.verification_mode) {
      const rtLabel = d.runtime_verified ? "&#9745; runtime pass" : (d.verification_status === "skip" ? "&#9744; runtime skipped" : d.verification_status);
      html += `<dt>Verification</dt><dd>${d.verification_mode} &mdash; ${rtLabel}</dd>`;
    }
    if (d.shapes && d.shapes.length) html += `<dt>Shapes</dt><dd>${JSON.stringify(d.shapes)}</dd>`;
    if (d.kernel_path) html += `<dt>Kernel</dt><dd><code>${d.kernel_path}</code></dd>`;
    if (kind === "generative") {
      if (d.agent_platform) html += `<dt>Agent</dt><dd>${d.agent_platform}${d.agent_completed ? " (completed)" : " (incomplete)"}</dd>`;
      if (d.artifacts && d.artifacts.length) html += `<dt>Artifacts</dt><dd>${d.artifacts.join(", ")}</dd>`;
    }
    if (d.notes) html += `<dt>Notes</dt><dd>${d.notes}</dd>`;
  }
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
  const filterTier = document.getElementById("filter-tier").value;
  const filterStatus = document.getElementById("filter-status").value;
  const filterDim = document.getElementById("filter-dimension").value;

  let rows = DATA.rows.filter(r => {
    if (filterTier !== "all" && r.tier !== filterTier) return false;
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
    { key: "tier", label: "Tier" },
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
  let lastTier = null;

  for (const r of rows) {
    if (r.tier !== lastTier) {
      lastTier = r.tier;
      const t = DATA.tiers[lastTier] || {};
      html += `<tr class="tier-header"><td colspan="${cols.length}">Tier ${t.level || "?"}: ${lastTier.replace(/_/g, " ")} &mdash; ${t.description || ""}</td></tr>`;
    }
    const repTags = (r.representative_of && r.representative_of.length)
      ? `<div class="rep-tags">${r.representative_of.map(x => `<span class="rep-tag">${x}</span>`).join("")}</div>`
      : "";
    html += "<tr>";
    html += `<td class="op-name">${r.op}${repTags}</td>`;
    html += `<td>${r.tier.replace(/_/g, " ")}</td>`;
    html += `<td>${r.dtype}</td>`;
    html += `<td class="api-col">${r.asc2_api}</td>`;
    html += `<td>${makeBadge(r.golden_status, r.golden_evidence, "golden", r.prompt)}</td>`;
    html += `<td>${makeBadge(r.generative_status, r.generative_evidence, "generative", r.prompt)}</td>`;
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

    for t_name in sorted(data["tiers"], key=lambda k: data["tiers"][k]["level"]):
        t = data["tiers"][t_name]
        print(f"  Tier {t['level']} ({t_name}): golden {t['golden_confirmed']}/{t['total_cells']}, "
              f"gen {t['generative_confirmed']}/{t['total_cells']}")


if __name__ == "__main__":
    main()
