// Synthetic UI fixtures only. No model, archive download, compiler or device access.
const {test, before, after, beforeEach, afterEach} = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
const {default: AxeBuilder} = require('@axe-core/playwright');
const root = path.resolve(__dirname, '../../dashboard');
let browser, server, origin, page, context;
function record(kind, extra = {}) {
  return {id:`synthetic-${kind}`, kind, availability:'available', freshness:'recent',
    release_url:'https://github.com/example/repo/releases/tag/test', archive_url:'https://github.com/example/repo/releases/download/test/test.tar.gz',
    archive_sha256:'a'.repeat(64), ci_url:kind === 'target' ? null : 'https://github.com/example/repo/actions/runs/1',
    category:'fresh-generated', level:'simulator-numerical', status:'complete', required:4, qualified:2,
    counts:{qualified:2,'candidate-failed':1,'budget-exhausted':1}, repetitions:2, elapsed_seconds:10,
    tasks:[{id:'add-tail',module:'test_add.py',cases:['tail']}],
    models:[{requested:'provider/model-a',provider_revision:null}], observed_models:['model-a'],
    finished_at:'2026-11-06T12:00:00Z', stack_revision:'b'.repeat(40), skill_hashes:{'skills/test/SKILL.md':'c'.repeat(64)},
    comparisons:{}, baseline_time_ratio:null, service_score:null, ...extra};
}
function data() {
  return {schema_version:1,built_at:new Date().toISOString(),max_age_hours:192,catalog_sha256:'d'.repeat(64),records:[
    record('paired',{comparisons:{'P6-P3':{pairs:2,qualification_delta:.5,interval:[-.5,1]}}}),
    record('models',{observed_models:['model-a','model-b'],models:[{requested:'provider/model-a'},{requested:'provider/model-b'}],
      comparisons:{'P6:B-A':{pairs:2,qualification_delta:-.5,interval:[-1,.5]}}}),
    record('review',{required:1,qualified:1,counts:{qualified:1},repetitions:1}),
    record('target',{category:'upstream-fixture',required:100,qualified:2,counts:{qualified:2,excluded:98},status:'partial',manual_run:true,models:[],observed_models:[]}),
    record('hardware',{level:'npu-numerical-and-timing',required:3,qualified:3,counts:{qualified:3},tasks:[{id:'swi_glu_1'}],observed_models:[],baseline_time_ratio:.25,service_score:32})
  ]};
}
async function load(payload = data(), status = 200) {
  await page.route('**/data.json', route => route.fulfill({status,contentType:'application/json',body:JSON.stringify(payload)}));
  await page.goto(origin);
  await page.waitForFunction(() => !document.querySelector('#freshness').textContent.startsWith('Loading'));
}
async function ids() {return page.locator('#records tr').evaluateAll(rows => rows.map(r => r.dataset.recordId));}
async function accessibility() {
  const report = await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
  assert.deepEqual(report.violations.map(v => ({id:v.id,impact:v.impact,nodes:v.nodes.map(n=>n.target)})), []);
}
before(async () => {
  server = http.createServer((req,res) => {
    const filename = req.url === '/' ? 'index.html' : req.url.slice(1);
    if (!['index.html','style.css','app.js'].includes(filename)) {res.writeHead(404);res.end();return;}
    res.setHeader('Content-Type', {'index.html':'text/html','style.css':'text/css','app.js':'text/javascript'}[filename]);
    res.end(fs.readFileSync(path.join(root,filename)));
  });
  await new Promise(resolve => server.listen(0,'127.0.0.1',resolve)); origin = `http://127.0.0.1:${server.address().port}/`;
  browser = await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHANNEL ? {channel:process.env.PLAYWRIGHT_CHANNEL} : {})});
});
after(async () => {await browser?.close(); await new Promise(resolve => server.close(resolve));});
beforeEach(async () => {context = await browser.newContext({viewport:{width:1440,height:1000}}); page = await context.newPage();});
afterEach(async () => {await context.close();});

test('counts, exclusions, uncertainty and timing stay separate', async () => {
  const payload = data(); await load(payload);
  assert.equal(await page.locator('#records tr').count(),5);
  for (const r of payload.records) {
    const row = page.locator(`[data-record-id="${r.id}"]`);
    assert.equal(await row.locator('.coverage strong').innerText(),`${r.qualified} / ${r.required}`);
    assert.equal(await row.locator('progress').getAttribute('value'),String(r.qualified));
    for (const [outcome,n] of Object.entries(r.counts)) assert.ok((await row.innerText()).includes(`${outcome}: ${n}`));
  }
  assert.match(await page.locator('#comparisons').innerText(),/Conditional 95% interval \[−?-?50.0, \+100.0\] pp/);
  const charts = page.locator('.contrast svg');
  assert.equal(await charts.nth(0).locator('circle').getAttribute('cx'),'219');
  assert.equal(await charts.nth(1).locator('circle').getAttribute('cx'),'81');
  assert.match(await page.locator('#timing').innerText(),/0.25000×/);
  assert.match(await page.locator('#timing').innerText(),/32.0000/);
  assert.match(await page.locator('#timing').innerText(),/candidate slower/);
  assert.match(await page.locator('#records').innerText(),/Excluded cases were not numerically tested/);
});

test('combined task, model, origin, verification and outcome filters retain cohort denominators', async () => {
  await load();
  await page.locator('#search').fill('ADD'); await page.locator('#filter').selectOption('fresh-generated');
  await page.locator('#level').selectOption('simulator-numerical'); await page.locator('#outcome').selectOption('failures');
  await page.locator('#model').selectOption('Observed: model-b');
  assert.deepEqual(await ids(),['synthetic-models']);
  assert.equal(await page.locator('.coverage strong').innerText(),'2 / 4');
  assert.equal(await page.locator('#comparisons .panel').count(),1);
  assert.match(await page.locator('#timing').innerText(),/No retained NPU/);
  // Overview always describes the whole catalog, never a filtered pass rate.
  assert.equal(await page.locator('.summary-card strong').first().innerText(),'5');
  await page.locator('#search').fill('no-such-operator');
  assert.equal(await page.locator('#records tr').count(),0); assert.ok(await page.locator('#empty').isVisible());
  await page.getByRole('button',{name:'Reset filters'}).focus(); await page.keyboard.press('Enter');
  assert.equal(await page.locator('#records tr').count(),5);
  for (const [filter,expected] of [['excluded',['synthetic-target']],['partial',['synthetic-target']],['qualified',data().records.map(r=>r.id)]]) {
    await page.locator('#outcome').selectOption(filter); assert.deepEqual(await ids(),expected);
  }
});

test('keyboard filters and provenance dialog retain exact hashes and return focus', async () => {
  await load(); await page.locator('#search').focus(); await page.keyboard.press('Tab');
  assert.equal(await page.locator(':focus').getAttribute('id'),'filter');
  await page.keyboard.press('g'); await page.keyboard.press('Enter'); await page.keyboard.press('Tab');
  assert.equal(await page.locator('#filter').inputValue(),'fresh-generated'); assert.equal((await ids()).length,4);
  const button = page.getByRole('button',{name:'Scope and provenance for synthetic-paired'});
  await button.focus(); await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog'); assert.ok(await dialog.isVisible());
  assert.ok((await dialog.innerText()).includes('c'.repeat(64))); assert.ok((await dialog.innerText()).includes('b'.repeat(40)));
  const displayed = JSON.parse(await dialog.locator('pre').innerText()); assert.deepEqual(displayed,data().records[0]);
  await accessibility(); await page.keyboard.press('Escape'); assert.equal(await page.getByRole('dialog').count(),0);
  assert.equal(await page.locator(':focus').getAttribute('aria-label'),'Scope and provenance for synthetic-paired');
});

test('stale page, stale result, unavailable record and incomplete effects have explicit states', async () => {
  const payload = data(); payload.built_at='2020-01-01T00:00:00Z'; payload.records[0].freshness='stale';
  payload.records[1].status='incomplete'; payload.records[1].comparisons={};
  payload.records[2] = {id:'unavailable',kind:'review',availability:'unavailable',freshness:'unknown',archive_sha256:'a'.repeat(64),
    release_url:payload.records[2].release_url,archive_url:payload.records[2].archive_url,error:'Evidence could not be retrieved or validated'};
  await load(payload); assert.match(await page.locator('#freshness').innerText(),/STALE PAGE.*1 unavailable/);
  assert.match(await page.locator('#comparisons').innerText(),/No validated paired estimate/);
  await page.locator('#outcome').selectOption('stale'); assert.deepEqual(await ids(),['synthetic-paired']);
  await page.locator('#outcome').selectOption('unavailable'); assert.deepEqual(await ids(),['unavailable']);
  assert.equal(await page.locator('#records progress').count(),0); assert.match(await page.locator('#records').innerText(),/No acceptance claim/);
  await accessibility();
});

test('empty catalog and unavailable-only catalog never fabricate plots or coverage', async () => {
  const payload = data(); payload.records=[]; await load(payload);
  assert.equal(await page.locator('#records tr').count(),0); assert.ok(await page.locator('#empty').isVisible());
  assert.equal(await page.locator('#comparisons svg').count(),0); assert.match(await page.locator('#timing').innerText(),/No retained NPU/);
  const r=record('paired'); payload.records=[{id:r.id,kind:r.kind,availability:'unavailable',archive_sha256:r.archive_sha256,release_url:r.release_url,archive_url:r.archive_url}];
  await load(payload); assert.equal(await page.locator('#records tr').count(),1); assert.equal(await page.locator('progress').count(),0);
  assert.match(await page.locator('#comparisons').innerText(),/No validated paired estimate/);
});

test('load errors, malformed counts, unsafe links and partial effect estimates fail closed', async () => {
  const badCounts=data();badCounts.records[0].qualified=999;
  const badLink=data();badLink.records[4].release_url='javascript:alert(1)';
  const partial=data();partial.records[0].status='incomplete';
  for (const [payload,status] of [[{},503],[{schema_version:2},200],[badCounts,200],[badLink,200],[partial,200]]) {
    await load(payload,status); assert.match(await page.locator('#freshness').innerText(),/EVIDENCE UNAVAILABLE/);
    for (const id of ['records','overview','comparisons','timing']) assert.equal(await page.locator(`#${id}`).innerText(),'');
    assert.ok(await page.locator('#search').isDisabled());
  }
});

test('external strings render as text under the retained CSP', async () => {
  const payload=data(); const injection='<img src=x onerror="window.injected=true">';
  payload.records[0].tasks[0].module=injection; payload.records[0].observed_models=[injection];
  payload.records[0].comparisons[injection]=payload.records[0].comparisons['P6-P3'];
  await load(payload); assert.ok((await page.locator('#records').innerText()).includes(injection));
  assert.equal(await page.locator('img').count(),0); assert.equal(await page.evaluate(()=>window.injected),undefined);
  assert.match(await page.locator('meta[http-equiv="Content-Security-Policy"]').getAttribute('content'),/script-src 'self'; style-src 'self'/);
});

test('desktop and narrow layouts have accessible contrast, no document overflow and scrollable matrix', async () => {
  await load();
  for (const width of [1440,768,390,320]) {
    await page.setViewportSize({width,height:900});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`overflow at ${width}`);
    await accessibility();
    if (width < 960) {
      const table = page.getByRole('region',{name:'Results matrix, scroll horizontally on narrow screens'});
      await table.focus(); await page.keyboard.press('ArrowRight');
      await page.waitForFunction(()=>document.querySelector('.table-wrap').scrollLeft>0);
      assert.ok(await page.locator('.scroll-hint').isVisible());
    }
  }
});

test('generated kernel viewer is hash-labelled, escaped, keyboard accessible and keeps missing attempts visible', async () => {
  const payload=data();
  const code='raise RuntimeError("DISPLAY ONLY")\n# <img src=x onerror="window.injected=true">\n';
  payload.records[0].kernel_sources={status:'partial',files:[
    {id:'trial-a/attempt-1',outcome:'qualified',member:'run/trial-a/add/attempt-1/candidate.py',sha256:'e'.repeat(64),text:code},
    {id:'trial-b/attempt-1',outcome:'failed',member:'run/trial-b/add/attempt-1/candidate.py',sha256:'f'.repeat(64),text:'# Failed source\n'}
  ],missing:[{id:'trial-c/attempt-1',reason:'No hash-bound candidate retained for this attempt'}]};
  await load(payload);
  const button=page.getByRole('button',{name:'View generated kernels for synthetic-paired'});
  await button.focus();await page.keyboard.press('Enter');
  const dialog=page.getByRole('dialog');
  assert.equal(await dialog.locator('pre').textContent(),code);
  assert.match(await dialog.innerText(),/Source SHA256 e{64}/);
  assert.equal(await page.locator('img').count(),0);assert.equal(await page.evaluate(()=>window.injected),undefined);
  await dialog.getByLabel('Candidate / archived outcome').selectOption('1');
  assert.equal(await dialog.locator('pre').textContent(),'# Failed source\n');
  await dialog.locator('summary').click();assert.match(await dialog.innerText(),/trial-c\/attempt-1: No hash-bound candidate/);
  await accessibility();await page.keyboard.press('Escape');
  assert.equal(await page.locator(':focus').getAttribute('aria-label'),'View generated kernels for synthetic-paired');
  assert.match(await page.locator('[data-record-id="synthetic-review"]').innerText(),/Kernel source unavailable/);
});

test('malformed kernel source projection fails closed', async () => {
  const payload=data();payload.records[0].kernel_sources={status:'available',files:[{id:'x',outcome:'qualified',member:'x.py',sha256:'bad',text:'print(1)'}],missing:[]};
  await load(payload);assert.match(await page.locator('#freshness').innerText(),/EVIDENCE UNAVAILABLE/);
  assert.equal(await page.getByRole('button',{name:/View generated kernels/}).count(),0);
});
