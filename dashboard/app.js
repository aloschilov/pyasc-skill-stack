'use strict';
const $ = id => document.getElementById(id);
const titles = {paired:'Paired skills & context', models:'Model comparison', review:'Reviewed generation', target:'Upstream fixture subset', hardware:'Generated NPU candidate'};
const levels = {'simulator-numerical':'Simulator numerics', 'npu-numerical-and-timing':'NPU numerics + timing'};
const failures = ['candidate-failed', 'budget-exhausted', 'failed', 'timeout', 'timed-out'];
const controls = ['search', 'filter', 'model', 'level', 'outcome'];
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function safeURL(url) {
  const parsed = new URL(url);
  if (parsed.protocol !== 'https:' || parsed.hostname !== 'github.com' || parsed.username || parsed.password) throw new Error('Unsafe link');
  return url;
}
function link(label, url) {
  const node = el('a', label); node.href = safeURL(url); return node;
}
function available(r) { return r.availability === 'available'; }
function models(r) {
  // Prefixes distinguish observed identity from an unverified requested identity.
  return [...(r.observed_models || []).map(m => `Observed: ${m}`), ...(r.models || []).map(m => `Requested: ${m.requested}`)];
}
function countsText(r) { return Object.entries(r.counts).map(([k,v]) => `${k}: ${v}`).join(' · '); }
function unit(r) { return r.kind === 'target' ? 'fixture catalog cases' : r.kind === 'hardware' ? 'device cases' : ['paired','models'].includes(r.kind) ? 'trials' : 'candidate'; }
function stateLabel(r) {
  if (!available(r)) return 'Unavailable';
  if (r.status !== 'complete') return r.status === 'partial' ? 'Partial scope' : 'Incomplete trials';
  return ['paired','models'].includes(r.kind) ? 'All trials accounted' : 'Qualified scope';
}
function matches(r) {
  const query = $('search').value.trim().toLowerCase();
  const searchable = [r.id, r.benchmark || '', ...(r.tasks || []).flatMap(t => [t.id, t.module || '', ...(t.cases || [])])].join(' ').toLowerCase();
  const outcome = $('outcome').value;
  const outcomeMatch = outcome === 'all' || (outcome === 'unavailable' && !available(r)) ||
    (outcome === 'stale' && r.freshness === 'stale') ||
    (outcome === 'partial' && ['partial','incomplete'].includes(r.status)) ||
    (outcome === 'failures' && failures.some(k => (r.counts || {})[k] > 0)) ||
    (['qualified','excluded'].includes(outcome) && (r.counts || {})[outcome] > 0);
  return searchable.includes(query) && ($('filter').value === 'all' || r.category === $('filter').value) &&
    ($('model').value === 'all' || models(r).includes($('model').value)) &&
    ($('level').value === 'all' || r.level === $('level').value) && outcomeMatch;
}
function renderOverview(data) {
  const records = data.records;
  const groups = [
    ['Retained archives', records.length, `${records.filter(available).length} available · ${records.filter(r => !available(r)).length} unavailable`],
    ['Generated / simulator', records.filter(r => available(r) && r.category === 'fresh-generated' && r.level === 'simulator-numerical').length, 'retained records · separate trial denominators'],
    ['Upstream fixtures', records.filter(r => available(r) && r.category === 'upstream-fixture').length, 'retained records · catalog ≠ tested coverage'],
    ['Generated / NPU', records.filter(r => available(r) && r.level === 'npu-numerical-and-timing').length, 'retained records · numerics + device timing']
  ];
  $('overview').replaceChildren(...groups.map(([label, value, note]) => {
    const card = el('article', undefined, 'summary-card'); card.append(el('h2', label), el('strong', value), el('p', note)); return card;
  }));
}
function evidenceLinks(r) {
  const links = el('div', undefined, 'links');
  links.append(link('Retained archive', r.release_url), link('Download', r.archive_url));
  if (r.ci_url) links.append(link('CI identity', r.ci_url));
  return links;
}
function provenance(r, trigger) {
  const dialog = el('dialog', undefined, 'provenance-dialog');
  dialog.setAttribute('aria-label', `Scope and provenance: ${r.id}`);
  const close = el('button', 'Close', 'close'); close.type = 'button'; close.addEventListener('click', () => dialog.close());
  dialog.append(close, el('h2', titles[r.kind]), el('p', r.id, 'record-id'),
    el('p', 'Exact task IDs, counts, source revisions, settings and hashes from the public allowlisted summary.', 'subtle'),
    evidenceLinks(r), el('code', `Archive SHA256 ${r.archive_sha256}`, 'checksum'), el('pre', JSON.stringify(r, null, 2)));
  const source = dialog.querySelector('pre'); source.tabIndex = 0; source.setAttribute('aria-label', 'Public scope and provenance JSON');
  dialog.addEventListener('close', () => { dialog.remove(); trigger.focus(); });
  document.body.append(dialog); dialog.showModal(); close.focus();
}
function kernelViewer(r, trigger) {
  const sources = r.kernel_sources;
  const dialog = el('dialog', undefined, 'provenance-dialog');
  dialog.setAttribute('aria-label', `Generated kernels: ${r.id}`);
  const close = el('button', 'Close', 'close'); close.type = 'button'; close.addEventListener('click', () => dialog.close());
  dialog.append(close, el('h2', 'Generated kernel source'), el('p', r.id, 'record-id'),
    el('p', 'Read-only Python from the retained archive, hash-verified at build time. Failed candidates remain labelled failed. Viewing source does not execute or independently qualify it.', 'subtle'));
  const label = el('label', 'Candidate / archived outcome'); const select = el('select'); label.append(select);
  for (const [index, file] of sources.files.entries()) {
    const option = el('option', `${file.id} · ${file.outcome}`); option.value = String(index); select.append(option);
  }
  const member = el('p', undefined, 'subtle'); const hash = el('code', undefined, 'checksum');
  const source = el('pre'); source.tabIndex = 0; source.setAttribute('aria-label', 'Generated Python source');
  const show = () => {
    const file = sources.files[Number(select.value)];
    member.textContent = `Archive member: ${file.member}`; hash.textContent = `Source SHA256 ${file.sha256}`; source.textContent = file.text;
  };
  select.addEventListener('change', show);
  dialog.append(label, member, hash, source);
  if (sources.missing.length) {
    const details = el('details'); details.append(el('summary', `${sources.missing.length} attempts without viewable source`));
    for (const item of sources.missing) details.append(el('p', `${item.id}: ${item.reason}`, 'subtle'));
    dialog.append(details);
  }
  dialog.append(evidenceLinks(r)); show();
  dialog.addEventListener('close', () => { dialog.remove(); trigger.focus(); });
  document.body.append(dialog); dialog.showModal(); close.focus();
}

function scopeText(r) {
  const tasks = r.tasks || [];
  // Shorten verbose pytest node IDs for scanning; full identifiers remain in provenance.
  const names = [...new Set(tasks.map(t => t.module || t.id.split('::')[0]))];
  return names.slice(0, 3).join(' · ') + (names.length > 3 ? ` · +${names.length-3} more` : '');
}
function renderRow(r) {
  const row = el('tr'); row.dataset.recordId = r.id;
  const scope = el('td'); scope.append(el('h3', titles[r.kind]), el('div', r.id, 'record-id'));
  if (available(r)) scope.append(el('p', r.category === 'upstream-fixture' ? 'Upstream fixtures · manual run' : 'Generated candidates', 'scope'), el('p', scopeText(r), 'scope'));
  const attribution = el('td');
  if (available(r)) {
    const observed = r.observed_models;
    for (const m of observed) attribution.append(el('span', m, 'model-name'));
    if (observed.length) attribution.append(el('span', 'Observed identity', 'subtle'));
    else if (r.models.length) {
      for (const m of r.models) attribution.append(el('span', m.requested, 'model-name'));
      attribution.append(el('span', 'Requested · observed identity not published', 'subtle'));
    } else attribution.append(el('span', 'No model generation', 'subtle'));
  } else attribution.append(el('span', 'Unknown', 'subtle'));
  const verification = el('td');
  verification.append(el('strong', levels[r.level] || 'Not verified'));
  const badges = el('div', undefined, 'badges');
  badges.append(el('span', stateLabel(r), `badge ${!available(r) ? 'bad' : r.status !== 'complete' ? 'warn' : 'blue'}`));
  if (available(r)) badges.append(el('span', r.freshness === 'stale' ? 'Stale result' : 'Recent result', `badge ${r.freshness === 'stale' ? 'warn' : 'good'}`));
  verification.append(badges);
  if (r.finished_at) verification.append(el('div', r.finished_at.slice(0,10), 'subtle'));
  const coverage = el('td', undefined, 'coverage');
  if (available(r)) {
    coverage.append(el('strong', `${r.qualified} / ${r.required}`), el('small', `qualified / required ${unit(r)}`));
    const bar = el('progress'); bar.max = r.required; bar.value = r.qualified;
    bar.setAttribute('aria-label', `${r.id}: ${r.qualified} qualified of ${r.required} required ${unit(r)}`);
    coverage.append(bar, el('p', countsText(r), 'counts'));
    if (r.kind === 'target') coverage.append(el('p', 'Excluded cases were not numerically tested.', 'subtle'));
  } else coverage.append(el('p', 'No acceptance claim'), el('p', r.error || 'Evidence could not be retrieved or validated.', 'subtle'));
  const evidence = el('td'); const button = el('button', 'Scope & provenance'); button.type = 'button';
  button.setAttribute('aria-label', `Scope and provenance for ${r.id}`);
  button.addEventListener('click', () => provenance(r, button));
  evidence.append(button);
  if (available(r) && r.category === 'fresh-generated') {
    const sources = r.kernel_sources;
    if (sources && sources.files.length) {
      const view = el('button', `View kernels (${sources.files.length})`, 'source-button'); view.type = 'button';
      view.setAttribute('aria-label', `View generated kernels for ${r.id}`);
      view.addEventListener('click', () => kernelViewer(r, view)); evidence.append(view);
      if (sources.missing.length) evidence.append(el('p', `${sources.missing.length} attempts: source unavailable`, 'subtle'));
    } else evidence.append(el('p', 'Kernel source unavailable in this summary.', 'subtle'));
  }
  evidence.append(el('p'), evidenceLinks(r));
  row.append(scope, attribution, verification, coverage, evidence); return row;
}
function svgNode(tag, attrs) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k,v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}
function intervalChart(effect) {
  // Fixed [-100,+100] pp domain for every contrast; zero and negative bounds stay visible.
  const x = value => 12 + (value + 1) * 138;
  const chart = svgNode('svg', {viewBox:'0 0 300 40', preserveAspectRatio:'none', 'aria-hidden':'true', focusable:'false'});
  chart.append(svgNode('line', {x1:150,x2:150,y1:0,y2:40,class:'chart-zero'}),
    svgNode('line', {x1:x(effect.interval[0]),x2:x(effect.interval[1]),y1:20,y2:20,class:'chart-range'}));
  for (const bound of effect.interval) chart.append(svgNode('line', {x1:x(bound),x2:x(bound),y1:14,y2:26,class:'chart-range'}));
  chart.append(svgNode('circle', {cx:x(effect.qualification_delta),cy:20,r:5,class:'chart-point'}));
  return chart;
}
function pp(value) { return `${value > 0 ? '+' : ''}${(value*100).toFixed(1)}`; }
function renderComparisons(records) {
  const panels = [];
  for (const r of records.filter(r => ['paired','models'].includes(r.kind))) {
    const panel = el('article', undefined, 'panel'); panel.append(el('h3', titles[r.kind]), el('div', r.id, 'record-id'));
    if (!available(r) || !Object.keys(r.comparisons).length) {
      panel.append(el('p', 'No validated paired estimate is available for this record. Incomplete trials do not support an effect estimate.', 'empty'));
    } else {
      panel.append(el('p', `${r.qualified}/${r.required} qualified trials · ${r.repetitions} repetitions`, 'subtle'),
        el('p', countsText(r), 'counts'), el('p', `Observed: ${r.observed_models.join(', ') || 'not published'}`, 'subtle'));
      for (const [name,effect] of Object.entries(r.comparisons)) {
        const contrast = el('div', undefined, 'contrast');
        contrast.append(el('strong', name), el('span', `${pp(effect.qualification_delta)} pp · ${effect.pairs} pairs`, 'effect-value'), intervalChart(effect),
          el('span', `Conditional 95% interval [${pp(effect.interval[0])}, ${pp(effect.interval[1])}] pp`, 'interval-label'));
        panel.append(contrast);
      }
      const axis = el('div', undefined, 'axis'); axis.append(el('span', '−100 pp'), el('span', '0 · no difference'), el('span', '+100 pp')); panel.append(axis);
      panel.append(el('p', 'Cohort totals above; arm-level rates are not provided in this summary. See the archive for protocol definitions and paired observations.', 'subtle'));
    }
    panel.append(evidenceLinks(r)); panels.push(panel);
  }
  $('comparisons').replaceChildren(...(panels.length ? panels : [el('p', 'No retained paired comparisons match these filters.', 'empty')]));
}
function renderTiming(records) {
  const panels = records.filter(r => available(r) && r.baseline_time_ratio != null).map(r => {
    const panel = el('article', undefined, 'panel'); panel.append(el('h3', `${titles[r.kind]} · ${r.qualified}/${r.required} device cases`), el('div', r.id, 'record-id'));
    const metrics = el('div', undefined, 'timing-metrics');
    const timing = el('div'); timing.append(el('span', 'Baseline / candidate time · geometric mean'), el('strong', `${r.baseline_time_ratio.toPrecision(5)}×`),
      el('p', r.baseline_time_ratio < 1 ? 'Below 1×: candidate slower than baseline.' : 'Above 1×: candidate faster; 1×: equal time.'));
    const service = el('div'); service.append(el('span', 'CANNBench service score'), el('strong', r.service_score === null ? 'Unavailable' : r.service_score.toFixed(4)), el('p', 'Service-defined score; not a timing speedup.'));
    metrics.append(timing, service); panel.append(metrics, el('p', `Device job: ${r.elapsed_seconds.toFixed(2)} s · ${stateLabel(r)} · ${r.freshness} result`, 'subtle'), evidenceLinks(r)); return panel;
  });
  $('timing').replaceChildren(...(panels.length ? panels : [el('p', 'No retained NPU timing evidence matches these filters. Simulator qualification does not supply a device timing ratio.', 'empty')]));
}
function render(data) {
  const records = data.records.filter(matches);
  $('count').textContent = `${records.length} of ${data.records.length} retained records`;
  $('records').replaceChildren(...records.map(renderRow)); $('empty').hidden = records.length > 0;
  renderComparisons(records); renderTiming(records);
}
const dailyLabels = {
  'unavailable':'Insufficient verified observations', 'incompatible':'Execution conditions differ',
  'same-skills':'Same skill bytes', 'tie':'Same first-attempt outcomes', 'mixed':'Gains and losses',
  'improved':'More first-attempt qualifications', 'regressed':'Fewer first-attempt qualifications'
};
function renderDaily(daily) {
  const panel = $('daily-quality');
  if (!daily) { panel.textContent = 'No verified daily comparison in this snapshot.'; return; }
  const dayNow = new Intl.DateTimeFormat('en-CA', {timeZone:'Europe/Moscow', year:'numeric', month:'2-digit', day:'2-digit'}).format(new Date());
  const historical = dayNow !== daily.dates[1];
  panel.replaceChildren(el('h3', historical ? 'Today’s comparison is unavailable' : dailyLabels[daily.status]), el('p',
    `${daily.dates.join(' → ')} · Europe/Moscow · ${historical ? 'Historical comparison: '+dailyLabels[daily.status] : 'Current comparison'}`));
  daily.runs.forEach((r,i) => {
    const line = el('p');
    line.append(el('strong', daily.dates[i]+' · '));
    if (r && r.cells) {
      const values = Object.values(r.cells), passed = values.filter(v => v === 'passed').length;
      line.append(el('span', `${passed}/${values.length} first attempts passed CaModel · ${r.model} · `));
    } else { line.append(el('span',(daily.reasons?.[i] || 'Qualification unavailable')+' · ')); }
    if (r) line.append(link(r.revision.slice(0,8), `https://github.com/aloschilov/pyasc-skill-stack/actions/runs/${r.run_id}/attempts/${r.attempt}`));
    panel.append(line);
  });
  if (daily.runs.every(r => r && r.cells)) {
    panel.append(el('p','Cohort: '+Object.keys(daily.runs[1].cells).join(', ')));
  }
  if (daily.gained?.length) panel.append(el('p','Gained: '+daily.gained.join(', ')));
  if (daily.lost?.length) panel.append(el('p','Lost: '+daily.lost.join(', ')));
  const hw = daily.hardware;
  if (hw?.status === 'measured') {
    const line = el('p', `Matched CANNBench first attempts · previous/current device time ${hw.previous_over_current.toFixed(3)}× across ${hw.count} cases · skill inputs ${hw.skills_changed ? 'changed' : 'unchanged'} · service score ${hw.scores[0].toFixed(4)} → ${hw.scores[1].toFixed(4)}. `);
    line.append(link('Previous',hw.sources[0]), document.createTextNode(' · '),link('Current',hw.sources[1])); panel.append(line);
  } else { panel.append(el('p','CANNBench speed change: no matched pair of qualified first attempts for these dates.')); }
}
function renderTransfer(study) {
  const panel=$('skill-transfer');
  if (study?.status !== 'available') { panel.textContent='Held-out comparison unavailable.'; return; }
  const cases=Object.keys(study.cases.A), total=cases.length;
  const passes=arm=>Object.values(study.cases[arm]).filter(v=>v==='passed').length;
  panel.replaceChildren(el('h3','Held-out CANNBench tasks · tuning guide revision'),
    el('p',`${study.model}: ${passes('A')}/${total} → ${passes('B')}/${total} first attempts passed CaModel (${study.arms.A.slice(0,7)} → ${study.arms.B.slice(0,7)}).`),
    el('p',cases.map(k=>`${k}: ${study.cases.A[k]} → ${study.cases.B[k]}`).join(' · ')),
    link('Fixed prompts, six generation sessions and numerical controls',study.url));
}
function renderSwiGLU(study) {
  const panel = $('swiglu-study');
  if (study?.status !== 'available') { panel.textContent = 'SwiGLU measurements unavailable.'; return; }
  panel.replaceChildren(el('h3','CANNBench · one OpenCode edit at eight blocks'),
    el('p', `All 20 cases passed. Six wide cases improved ${study.parent_over_wide.toFixed(3)}× (geometric mean). Service scores: parent ${study.scores.parent.toFixed(4)}, retiled ${study.scores.wide.toFixed(4)}, reference ${study.scores.reference.toFixed(4)}.`),
    link('Hardware reports and generation provenance',study.release),
    el('h3','Shape dispatch · retained hardware replay'),
    el('p', `Archive / generated dispatch time: ${study.archive_over_shape.toFixed(5)}× across 20 cases on the same PyASC extension and CPython 3.12. Generation included geometry feedback, API repair and an existing BF16 template.`),
    link('Shape selection, sources and per-launch samples',study.shape_evidence));
  const details = el('details'); details.append(el('summary','Compare all 20 cases'));
  const table = el('table'); table.append(el('caption','Device time in microseconds. The final two columns use the separate same-runtime replay.'));
  const head=table.createTHead().insertRow();
  for (const title of ['Case','Parent','Retiled','PyASC reference','Shape dispatch','Archived replay']) {
    const th=el('th',title); th.scope='col'; head.append(th);
  }
  const body=table.createTBody();
  for (const r of study.cases) {
    const row=body.insertRow(); const th=el('th',String(r.case)); th.scope='row'; row.append(th);
    for (const k of ['parent','wide','reference','shape_us','replay_us']) row.insertCell().textContent=r[k].toFixed(3);
  }
  const wrap=el('div',undefined,'table-wrap'); wrap.tabIndex=0; wrap.setAttribute('role','region');
  wrap.setAttribute('aria-label','SwiGLU case timings'); wrap.append(table); details.append(wrap); panel.append(details);
}
function freshness(data) {
  const hours = (Date.now()-Date.parse(data.built_at))/3600000;
  const missing = data.records.filter(r => !available(r)).length;
  $('freshness').textContent = `${hours > 24 ? 'STALE PAGE — ' : ''}Evidence checked ${data.built_at}. ${missing} unavailable records. Result freshness threshold: ${data.max_age_hours} hours. ${hours > 24 ? 'Rebuild before relying on current availability.' : 'Retained snapshot · not a live device monitor.'}`;
  $('freshness').className = hours > 24 || missing ? 'attention' : '';
}
function validate(data) {
  // The builder owns archive validation. Check the rendered shape before displaying any result.
  const require = ok => { if (!ok) throw new Error('Invalid public summary'); };
  require(data.schema_version === 1 && Array.isArray(data.records) && Number.isFinite(Date.parse(data.built_at)) && Number.isInteger(data.max_age_hours) && data.max_age_hours > 0);
  if (data.daily) {
    const d=data.daily;
    require(Object.hasOwn(dailyLabels,d.status) && d.timezone === 'Europe/Moscow' && Array.isArray(d.dates) && d.dates.length===2 && d.dates.every(v => /^\d{4}-\d{2}-\d{2}$/.test(v)) && Array.isArray(d.runs) && d.runs.length===2);
    require(Date.parse(d.dates[1]) - Date.parse(d.dates[0]) === 86400000);
    if (!['unavailable','incompatible'].includes(d.status)) require(d.runs.every(r => r?.cells));
    for (const r of d.runs) if (r) {
      require(Number.isSafeInteger(r.run_id) && r.run_id>0 && Number.isSafeInteger(r.attempt) && r.attempt>0 && /^[0-9a-f]{40}$/.test(r.revision));
      if (r.cells) require(Object.keys(r.cells).length>0 && Object.values(r.cells).every(v => ['passed','failed'].includes(v)));
    }
    if (d.hardware?.status==='measured') {
      require(d.hardware.count === 20 && typeof d.hardware.skills_changed === 'boolean' && d.hardware.sources.length === 2 && Number.isFinite(d.hardware.previous_over_current) && d.hardware.previous_over_current>0 && d.hardware.scores.length===2 && d.hardware.scores.every(Number.isFinite));
      d.hardware.sources.forEach(safeURL);
    }
  }
  if (data.transfer?.status === 'available') {
    const t = data.transfer; safeURL(t.url);
    require(typeof t.model === 'string' && ['A','B'].every(a => /^[a-f0-9]{40}$/.test(t.arms[a])));
    const cases = Object.keys(t.cases.A);
    require(cases.length === 3 && Object.keys(t.cases.B).length === cases.length &&
      cases.every(k => ['A','B'].every(a => ['passed','failed'].includes(t.cases[a][k]))));
  }
  if (data.swiglu?.status === 'available') {
    const s=data.swiglu; safeURL(s.release); safeURL(s.shape_evidence);
    require(s.cases.length===20 && s.cases.every((r,i)=>r.case===i+1 && ['parent','wide','reference','shape_us','replay_us'].every(k=>Number.isFinite(r[k]) && r[k]>0)));
    require(['wide','parent','reference'].every(k=>Number.isFinite(s.scores[k])) && [s.parent_over_wide,s.archive_over_shape].every(v=>Number.isFinite(v)&&v>0));
  }
  const ids = new Set();
  for (const r of data.records) {
    require(typeof r.id === 'string' && !ids.has(r.id) && Object.hasOwn(titles, r.kind)); ids.add(r.id);
    require(['available','unavailable'].includes(r.availability) && typeof r.archive_sha256 === 'string' && /^[0-9a-f]{64}$/.test(r.archive_sha256));
    safeURL(r.release_url); safeURL(r.archive_url); if (r.ci_url) safeURL(r.ci_url);
    if (!available(r)) continue;
    if (r.kernel_sources !== undefined) {
      const sources = r.kernel_sources;
      require(sources && ['available','partial','unavailable','not-applicable'].includes(sources.status) && Array.isArray(sources.files) && sources.files.length <= 2000 && Array.isArray(sources.missing));
      require(!['unavailable','not-applicable'].includes(sources.status) || sources.files.length === 0);
      for (const file of sources.files) require(typeof file.id === 'string' && ['qualified','failed'].includes(file.outcome) && typeof file.member === 'string' && typeof file.sha256 === 'string' && /^[0-9a-f]{64}$/.test(file.sha256) && typeof file.text === 'string' && file.text.length > 0 && file.text.length <= 131072);
      for (const item of sources.missing) require(typeof item.id === 'string' && typeof item.reason === 'string');
    }
    require(Object.hasOwn(levels, r.level) && ['fresh-generated','upstream-fixture'].includes(r.category) && ['complete','partial','incomplete'].includes(r.status) && ['recent','stale'].includes(r.freshness));
    require(Number.isInteger(r.required) && r.required > 0 && Number.isInteger(r.qualified) && r.qualified >= 0 && r.qualified <= r.required);
    require(r.counts && Object.values(r.counts).every(n => Number.isInteger(n) && n >= 0) && Object.values(r.counts).reduce((a,b) => a+b,0) === r.required && (r.counts.qualified || 0) === r.qualified);
    require(Array.isArray(r.tasks) && r.tasks.every(t => typeof t.id === 'string' && (t.cases === undefined || Array.isArray(t.cases))));
    require(Array.isArray(r.models) && r.models.every(m => typeof m.requested === 'string') && Array.isArray(r.observed_models) && r.observed_models.every(m => typeof m === 'string'));
    require(Number.isFinite(r.elapsed_seconds) && r.elapsed_seconds >= 0 && Number.isInteger(r.repetitions) && r.repetitions > 0 && Number.isFinite(Date.parse(r.finished_at)));
    require(r.comparisons && typeof r.comparisons === 'object' && !Array.isArray(r.comparisons));
    for (const effect of Object.values(r.comparisons)) {
      require(r.status === 'complete' && Number.isInteger(effect.pairs) && effect.pairs > 0 && Number.isFinite(effect.qualification_delta) && Math.abs(effect.qualification_delta) <= 1 && Array.isArray(effect.interval) && effect.interval.length === 2 && effect.interval.every(n => Number.isFinite(n) && Math.abs(n) <= 1) && effect.interval[0] <= effect.qualification_delta && effect.interval[1] >= effect.qualification_delta);
    }
    require(r.baseline_time_ratio == null || r.level === 'npu-numerical-and-timing' && Number.isFinite(r.baseline_time_ratio) && r.baseline_time_ratio > 0);
    require(r.service_score == null || Number.isFinite(r.service_score));
  }
}
function unavailable() {
  for (const id of ['overview','records','comparisons','timing','count','daily-quality','skill-transfer','swiglu-study']) $(id).replaceChildren();
  $('empty').hidden = true;
  $('freshness').textContent = 'EVIDENCE UNAVAILABLE — the public summary could not be loaded or validated. No result is accepted here.';
  $('freshness').className = 'attention';
  for (const control of $('filters').elements) control.disabled = true;
}
fetch('data.json', {cache:'no-store'}).then(r => {if (!r.ok) throw new Error('Missing data'); return r.json();}).then(data => {
  validate(data);
  for (const name of [...new Set(data.records.flatMap(models))].sort()) {
    const option = el('option', name); option.value = name; $('model').append(option);
  }
  renderDaily(data.daily); renderTransfer(data.transfer); renderSwiGLU(data.swiglu); renderOverview(data); render(data); freshness(data);
  for (const id of controls) $(id).addEventListener(id === 'search' ? 'input' : 'change', () => render(data));
  $('filters').addEventListener('submit', e => e.preventDefault());
  $('filters').addEventListener('reset', () => { for (const id of controls) $(id).value = id === 'search' ? '' : 'all'; render(data); });
  setInterval(() => { freshness(data); renderDaily(data.daily); }, 60000);
}).catch(unavailable);
