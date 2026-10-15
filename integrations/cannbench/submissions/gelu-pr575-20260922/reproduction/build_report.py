"""Build the publication from the submitted ZIP and allowlisted evidence (stdlib only)."""
from pathlib import Path
import ast,csv,hashlib,io,json,math,re,zipfile
ROOT=Path(__file__).resolve().parent.parent
load=lambda n:json.loads((ROOT/n).read_text())
def save(n,v):(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
package=load('evidence/package.json')
assert sha((ROOT/'submission.zip').read_bytes())==package['archive_sha256']
with zipfile.ZipFile(ROOT/'submission.zip') as z:
    for n,h in package['archive_members'].items():assert sha(z.read(n))==h,n
    source=z.read('cann_bench/gelu.py')
    wheel=next(n for n in z.namelist() if n.endswith('.whl'))
    with zipfile.ZipFile(io.BytesIO(z.read(wheel))) as w:assert w.read('cann_bench/gelu.py')==source
assert sha(source)==package['source_sha256']
(ROOT/'asctile/gelu.py').write_bytes(source)
lines=source.decode().splitlines(keepends=True)
functions={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
gen=load('evidence/generation.json');quals=[load(f'evidence/qualification-cap{c}.json') for c in (64,72)]
assert gen['status']=='passed' and gen['compile_passed']==20
specs={s['specialization_id']:s for s in gen['specializations']}
assert len(specs)==6
for i,s in enumerate(gen['specializations']):
    for q in quals:
        prior=q['specializations'][i]
        for k in ('compile_options','constexprs','memory_consumed','ascendc_bytes'):assert s[k]==prior[k],k
    cpp=ROOT/'ascendc'/f"{s['specialization_id']}.cpp"
    assert sha(cpp.read_bytes())==s['ascendc_sha256']
    fn=functions[s['specialization_key']['kernel']]
    dtype=s['specialization_key']['arg_types']['input_ptr'].split(':')[-1]
    mode='tanh' if s['constexprs']['approximate'].endswith('(True)') else 'none'
    route=dtype+'_'+mode
    body=''.join(lines[min([fn.lineno]+[d.lineno for d in fn.decorator_list])-1:fn.end_lineno])
    (ROOT/'asctile'/f'{route}.py').write_text('# Exact device body extracted from submitted asctile/gelu.py.\n# Specialization: '+json.dumps(s['specialization_key'],sort_keys=True)+'\nimport asctile\n\n'+body)
    s['asctile_file']=f'asctile/{route}.py';s['ascendc_file']=f'ascendc/{s["specialization_id"]}.cpp'
case_specs={c['case_id']:specs[c['specialization_ids'][0]] for c in gen['case_results']}
hw=load('evidence/hardware-results.json');prev=load('evidence/previous-hardware-results.json');job=load('evidence/submission.json')
cases={int(c['case_id'].rsplit('_',1)[-1]):c for c in hw['operators'][0]['cases']}
old={int(c['case_id'].rsplit('_',1)[-1]):c for c in prev['results']['operators'][0]['cases']}
rows=[]
for t in load('evidence/tilings.json'):
    i=t['case_id'];s=case_specs[i];c=cases[i];opts=s['compile_options'];N=t['size'];L=t['block_length'];T=t['tile'];B=t['blocks'];last=N-(B-1)*L
    assert L==((N+71)//72+511)//512*512 and B==(N+L-1)//L
    assert s['specialization_key']['arg_types']['input_ptr'].endswith(':'+t['dtype'])
    assert s['constexprs']['tile_length']==f'ConstExpr[int]({T})'
    assert list(c['op_times']['device_kernels'])==[s['specialization_key']['kernel']]
    assert c['accuracy']['output_results'][0]['total_count']==N
    assert c['accuracy']['output_results'][0]['dtype']==t['dtype']
    assert all(q['case_results'][i-1]['cores']==[B] for q in quals)
    row=dict(case_id=i,shape='x'.join(map(str,t['shape'])),dtype=t['dtype'],mode=t['mode'],elements=N,ttk_key=t['key'],core_num=72,logical_blocks=B,block_length=L,last_block_elements=last,tile=T,regular_full_tiles=L//T,regular_tail_elements=L%T,last_full_tiles=last//T,last_tail_elements=last%T,unroll=2,UB_bytes=s['memory_consumed']['UB'],**opts,accuracy_passed=c['accuracy']['passed'],status=c['status'],elapsed_us=c['elapsed_us'],baseline_us=c['baseline_perf_us'],speedup=c['baseline_perf_us']/c['elapsed_us'],previous_elapsed_us=old[i]['elapsed_us'],previous_over_current=old[i]['elapsed_us']/c['elapsed_us'],asctile_code=s['asctile_file'],ascendc_code=s['ascendc_file'])
    rows.append(row)
save('report.json',rows)
with (ROOT/'report.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
gm=lambda xs:math.exp(sum(math.log(v) for v in xs)/len(xs))
metrics=dict(passed_cases=hw['passed_cases'],total_cases=hw['total_cases'],anti_cheat_failed_cases=hw['summary']['anti_cheat_failed_cases'],true_geometric_mean_speedup=gm([r['speedup'] for r in rows]),arithmetic_mean_speedup=sum(r['speedup'] for r in rows)/20,previous_true_geometric_mean_speedup=gm([c['baseline_perf_us']/c['elapsed_us'] for c in old.values()]),geometric_mean_previous_over_current=gm([r['previous_over_current'] for r in rows]))
save('evidence/metrics.json',metrics)
flags=gen['specializations'][0]['compile_options']
assert all(s['compile_options']==flags for s in specs.values())
text=f'''# GeLU PR575 submission — 20/20 passed

[Hardware job]({job['url']}) · [exact submitted archive](submission.zip) · [CSV](report.csv) · [JSON](report.json) · [complete submitted AscTile module](asctile/gelu.py)

CANNBench accepted all **20/20 cases**, with **0 anti-cheat failures**. Independently calculated geometric-mean speedup versus the benchmark reference is **{metrics['true_geometric_mean_speedup']:.6f}×** (greater than 1 means faster). The API field named `geometric_mean_speedup` is **{hw['summary']['geometric_mean_speedup']:.6f}×**; it equals the arithmetic mean of these 20 speedups, so it is not used as the geometric mean here. See [raw result payload](evidence/hardware-results.json) and [derived metrics](evidence/metrics.json).

This is the latest submitted PR575 geometry experiment, tag `{job['job_tag']}`, private 950PR job `{job['id']}`, benchmark `{job['benchmark_slug']}` v{job['benchmark_version']}. Publishing these requested files does not change job visibility on CANNBench. Hardware environment: `{hw['setup_info']['environment']['npu']}`, CANN `{hw['setup_info']['environment']['cann']}`, Python `{hw['setup_info']['environment']['python']}`. [Submission metadata](evidence/submission.json) includes timestamps and runner revisions; credential-bearing job fields are excluded.

## Source and generation provenance

- pyasc runtime pin: `{package['pin']}`; PR575 head: `543fa2d02a5f66cabb38ee1fd39abf5552bc339e` ([PR](https://gitcode.com/compiler-team/pyasc/pull/575)).
- Submitted source SHA256: `{package['source_sha256']}`.
- Exact submitted ZIP SHA256: `{package['archive_sha256']}`. The ZIP includes the evaluator wheel and build inputs; it is byte-identical to the uploaded archive.
- AscTile files are exact device-function extracts from the submitted source, with descriptive headers. Both `gelu_kernel` (guarded grouped loop) and `gelu_reference_kernel` are submitted candidate functions; the latter name does **not** designate CANNBench's reference implementation. The FP32 exact route retains the reviewed inline AscendC Erfc expression.
- Six standalone `.cpp` files were **regenerated on the VM** from the installed submission wheel on 2026-09-23, using the pinned pyasc runtime and existing qualification SDK. They are complete generated translation units, not hand-written approximations. All20 host dispatch/translation checks passed. Each row has an explicit specialization ID mapping; flags, constexprs, UB use and generated byte counts match the original cap64 and cap72 qualification.
- Generation uses mocked full resource count64 and platform `{gen['platform']}` with the real installed SDK metadata bridge. Hardware results report `Ascend950PR_9589`. These exports are **not claimed to be the runner's captured C++ or binary**, and translation is not an additional numerical hardware run. [Generation record](evidence/generation.json), [qualification](evidence/qualification.json), [exact command](evidence/generation-command.json), [export script](reproduction/export_codegen.py).

## Does tiling depend on shape size?

The submitted module uses TTK-derived constants; it does not run TTK during dispatch. For each dtype/mode, the local tile is fixed across the 20 recorded cases:

| dtype | exact (`none`) tile | `tanh` tile |
| --- | ---: | ---: |
| float16 | 4608 | 5376 |
| bfloat16 | 4608 | 5376 |
| float32 | 5440 | 8192 |

Tile sizes are element counts. The flattened element count N changes block length, logical block count, loop iterations and tails through the formulas below. Individual dimensions and rank do not select a different tile. For example, float32 exact uses tile 5440 for both 512x2049 (block length 14848, 71 blocks) and 2048x2048 (block length 58368, 72 blocks). This describes this submitted policy and the recorded cases, not a universal property of TTK.

Compared with [repaired R](../gelu-uniform-jit-repaired-r/README.md), this attempt replaces the multi-tile geometric heuristic and quotient/remainder tile distribution with one TTK-derived tile per dtype/mode and 512-element-aligned partitions using divisor 72. It adds full-resource guards and TTK launch logging. GeLU expressions, guarded-loop strategy and JIT settings are retained. See the [exact source diff](repaired-r-to-ttk.patch); the standalone module and archive retain their original bytes.

## Exact geometry and compiler settings

All dimensions below are **elements**, except UB bytes. `core_num=72` is the PR partition divisor. For N elements: `block_length = round_up(ceil(N/72),512)`, `logical_blocks = ceil(N/block_length)`, and the final block has `N-(logical_blocks-1)*block_length` elements. These logical grid sizes are preserved even at full reported physical count64. The host rejects unknown or observably reduced device/thread/stream limits; it does not change those limits. Actual runner resource counts are not available in the retained API log tails.

A regular block processes `block_length // tile` full tiles plus `block_length % tile` tail elements; the final block follows the same rule with `last_block_elements`. Masks use the exact remaining valid element count. CSV/JSON include all these quantities, PR tiling keys, every compiler-option value, and previous-run timings. Input length is a runtime int32, so cases sharing dtype/mode share one generated specialization. Unroll is2; guarded and ordinary loops retain their respective submitted bodies.

All six specializations have the following observed options (null means unspecified, not false). No additional Bisheng options were supplied; exports stop at AscendC translation and do not establish the runner's native compiler command.

| Compiler option | Value |
| --- | --- |
'''
for k,v in flags.items():text+=f'| `{k}` | `{json.dumps(v)}` |\n'
text+='\nCodegen options: `capture_exceptions=true`, `ir_multithreading=true`; standard pyasc custom builtins are recorded in generation metadata. Device type is AIV_ONLY, `vf_vec_len=256`.\n\n## Per-case report\n\nSpeedup = reference µs / candidate µs. All rows passed accuracy. Final two columns link to separate AscTile and generated AscendC files. `sync` means `insert_sync`; `VF` means `vf_fusion`.\n\n'
columns=[('Case','case_id'),('Shape','shape'),('dtype','dtype'),('Mode','mode'),('N','elements'),('core_num','core_num'),('Blocks','logical_blocks'),('Block length','block_length'),('Last block','last_block_elements'),('Tile','tile'),('UB bytes','UB_bytes'),('Unroll','unroll'),('reuse_alloc','reuse_alloc'),('static_alloc','static_alloc'),('VF','vf_fusion'),('sync','insert_sync'),('opt_level','opt_level'),('Candidate µs','elapsed_us'),('Reference µs','baseline_us'),('Speedup','speedup'),('Accuracy','accuracy_passed'),('AscTile code','asctile_code'),('AscendC code','ascendc_code')]
text+='| '+' | '.join(k for k,v in columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'
for r in rows:
    vals=[]
    for _,k in columns:
        v=r[k]
        if k in ('asctile_code','ascendc_code'):v=f'[{Path(v).name}]({v})'
        elif k=='speedup':v=f'{v:.6f}×'
        elif k=='accuracy_passed':v='PASS' if v else 'FAIL'
        elif v is None:v='null'
        vals.append(str(v))
    text+='| '+' | '.join(vals)+' |\n'
text+=f'''
## Comparison and interpretation

Previous repaired R job [`{prev['job_id']}`](https://cannbench.com/workspace/jobs/{prev['job_id']}) passed20/20 with true geometric mean **{metrics['previous_true_geometric_mean_speedup']:.6f}×**. This job's value is **{metrics['true_geometric_mean_speedup']:.6f}×**. The geometric mean of paired previous/current candidate latencies is **{metrics['geometric_mean_previous_over_current']:.6f}×** (above1 would mean this submission is faster). Per-case previous timings are in CSV/JSON; [previous result evidence](evidence/previous-hardware-results.json) retains the underlying values.

PR575 geometry did not close the performance gap to the benchmark reference. This single cross-job comparison does not exclude tiling as a general performance factor: the tile and logical-grid policy changed, runs were separate, and it is not a repeated controlled hardware timing study. No additional submission was made to prepare this report.

## Reproduction and verification

Run `python3 reproduction/build_report.py` from any directory to verify the exact ZIP/member/source hashes, map all20 cases to six specializations, check the original qualification matches, and rebuild this report, CSV and JSON. Run `python3 reproduction/verify_publication.py` to check every published file hash and all local report links. CANN translation must run on the VM; see [reproduction instructions](reproduction/README.md).
'''
(ROOT/'README.md').write_text(text)
print(json.dumps(metrics,indent=2))
