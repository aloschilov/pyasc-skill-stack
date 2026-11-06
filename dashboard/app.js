'use strict';
const $ = id => document.getElementById(id);
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function link(label, url) {
  const node = el('a', label);
  const parsed = new URL(url);
  if (parsed.protocol !== 'https:' || parsed.hostname !== 'github.com') throw new Error('Unsafe link');
  node.href = url; return node;
}
function metric(label, value) {
  const node = el('div', undefined, 'metric'); node.append(el('strong', value), el('span', label)); return node;
}
function render(data) {
  const selected = $('filter').value;
  const records = data.records.filter(r => selected === 'all' || r.category === selected || r.availability === selected);
  $('count').textContent = `${records.length} retained records`;
  $('records').replaceChildren();
  for (const r of records) {
    const card = el('article', undefined, 'card');
    const top = el('div', undefined, 'card-top');
    const stateLabel = r.status === 'complete' ? (['paired','models'].includes(r.kind) ? 'all trials accounted' : 'qualified scope') : r.status;
    top.append(el('span', r.level || r.kind, 'eyebrow'), el('span', r.availability === 'available' ? `${stateLabel} · ${r.freshness}` : 'unavailable', 'badge'));
    card.append(top, el('h2', r.id));
    if (r.availability === 'available') {
      const metrics = el('div', undefined, 'metrics');
      metrics.append(metric('qualified / required', `${r.qualified} / ${r.required}`), metric('repetitions', r.repetitions),
        metric(r.level === 'npu-numerical-and-timing' ? 'device job, seconds' : r.category === 'upstream-fixture' ? 'fixture phases, seconds' : 'experiment, seconds', Number(r.elapsed_seconds).toFixed(2)));
      card.append(metrics, el('p', Object.entries(r.counts).map(([k,v]) => `${k}: ${v}`).join(' · '), 'counts'));
      const observed = r.observed_models.length ? r.observed_models.join(', ') : (r.models.length ? 'See retained model attribution' : 'No model generation');
      card.append(el('p', observed, 'model'), el('p', `${r.manual_run ? 'Manual fixture run' : 'CI-linked result'} · ${r.finished_at}`, 'date'));
      if (r.baseline_time_ratio !== null && r.baseline_time_ratio !== undefined) {
        card.append(el('p', `Baseline / candidate time: ${r.baseline_time_ratio.toPrecision(5)}× · Service score: ${Number(r.service_score).toFixed(4)}`, 'timing'));
      }
      for (const [name, effect] of Object.entries(r.comparisons)) {
        card.append(el('p', `${name}: ${(100*effect.qualification_delta).toFixed(1)} percentage points; conditional 95% interval [${effect.interval.map(n => (100*n).toFixed(1)).join(', ')}], ${effect.pairs} pairs.`, 'comparison'));
      }
      const details = el('details'); details.append(el('summary', 'Scope, settings and immutable provenance'));
      const safe = {...r}; delete safe.comparisons;
      details.append(el('pre', JSON.stringify(safe, null, 2))); card.append(details);
    } else card.append(el('p', r.error || 'Evidence unavailable. No acceptance claim is displayed.'));
    const links = el('div', undefined, 'links');
    links.append(link('Retained archive', r.release_url), link('Download', r.archive_url));
    if (r.ci_url) links.append(link('CI identity', r.ci_url));
    card.append(links, el('code', `SHA256 ${r.archive_sha256}`, 'checksum')); $('records').append(card);
  }
}
function freshness(data) {
  const hours = (Date.now()-Date.parse(data.built_at))/3600000;
  const missing = data.records.filter(r => r.availability !== 'available').length;
  $('freshness').textContent = `${hours > 24 ? 'STALE PAGE — ' : ''}Evidence checked ${data.built_at}. ${missing} unavailable records. ${hours > 24 ? 'Rebuild before relying on current availability.' : 'This is a retained snapshot, not a live device monitor.'}`;
  $('freshness').className = hours > 24 || missing ? 'attention' : '';
}
fetch('data.json', {cache:'no-store'}).then(r => {if (!r.ok) throw new Error('Missing data'); return r.json();}).then(data => {
  if (data.schema_version !== 1 || !Array.isArray(data.records) || !Number.isFinite(Date.parse(data.built_at))) throw new Error('Invalid schema');
  render(data); freshness(data);
  $('filter').addEventListener('change', () => render(data));
  setInterval(() => freshness(data), 60000);
}).catch(() => {
  $('records').replaceChildren(); $('freshness').textContent = 'EVIDENCE UNAVAILABLE — the public summary could not be loaded or validated. No result is accepted here.';
  $('freshness').className = 'attention';
});
