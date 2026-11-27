"""Offline fixed-shape receipt composition; no framework or Docker calls."""
from pathlib import Path
import copy
import hashlib
import json
import math
import shutil
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

RECIPE=Path(__file__).resolve().parents[3]/'examples/tuning/swiglu'
sys.path.insert(0,str(RECIPE))
import tool
from consumer_interface import COMPARE_ACTIONS
from non_tool_budget import CAPS
from selection import select

class ShapeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.work=self.base/'evaluation';self.work.mkdir();self.recipe=self.base/'recipe'
        self.fixtures=json.loads((RECIPE/'fixtures.json').read_text())['small-tail'];self.ids=[f['id']for f in self.fixtures]
        self.recipe.mkdir();(self.recipe/'engine/probe/checker').mkdir(parents=True)
        shutil.copyfile(RECIPE/'baseline.py',self.recipe/'baseline.py');shutil.copyfile(RECIPE/'timing-template.json',self.recipe/'timing-template.json')
        self.save(self.recipe/'engine/probe/fixtures.json',{'small-tail':self.fixtures})
        self.save(self.recipe/'engine/probe/qualification-plan.json',{'runtime':json.loads((RECIPE/'engine-lock.json').read_text())['runtime'],'checker_hashes':{'tests/tools/execute_kernel.py':'a'*64}})
        (self.recipe/'engine/probe/checker/execute_kernel.py').write_text('# Synthetic trusted checker fixture, never executed.\n')
        (self.recipe/'engine/probe/trusted-fixture-draft.py').write_text("if __name__ == '__main__':\n    main()")
        for name in (*tool.TIMING_HELPERS,'qualification_observe.py'):(self.recipe/'engine'/name).write_text('# Synthetic source closure, never executed.\n')
        self.save(self.recipe/'recipe-input-hashes.json',tool.tree_hashes(self.recipe))
        self.cpu=self.base/'cpu.json'
        self.save(self.cpu,{f['id']+'/'+str(seed):{'outcomes':{'native':True,'zero':False,'poison':False},'finite':True,'dtype':'float16','shape':f['shape'],'attrs':f['attrs'],'torch_version':'2.7.1+cpu','input_sha256':'1'*64,'native_sha256':'2'*64,'fp64_sha256':'3'*64}for f in self.fixtures for seed in f['seeds']})
        self.save(self.recipe/'session-config.json',{'attempt':str(self.base/'attempt'),'work':str(self.work),'cpu_receipt':str(self.cpu)})
        self.save(self.recipe/'admission.json',{'status':'ADMITTED','session_config_sha256':tool.sha(self.recipe/'session-config.json'),'recipe_input_sha256':tool.sha(self.recipe/'recipe-input-hashes.json'),'cpu_receipt_sha256':tool.sha(self.cpu)})
        rootpatch=patch.object(tool,'ROOT',self.recipe);rootpatch.start();self.addCleanup(rootpatch.stop)
        baseline=(RECIPE/'baseline.py').read_bytes();self.source_bytes={'baseline':baseline,'variant':baseline.replace(b'T = 128',b'T = 16')}
        self.sources={name:hashlib.sha256(data).hexdigest()for name,data in self.source_bytes.items()}
        for name,data in self.source_bytes.items():self.qualify_fixture(name,data)
    def save(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    def terminal(self):return {'exit_code':0,'original_client_reaped':True,'original_container_absent':True,'name_absent':True,'errors':[]}
    def qualify_fixture(self,name,data):
        scope=self.work/name;tool.prepare_qualification(scope,data,self.cpu);self.save(scope/'terminal.json',self.terminal())
        checks=[]
        for fixture in self.fixtures:
            for seed in fixture['seeds']:
                case=scope/'out'/(fixture['id']+'-'+str(seed));(case/'dumps').mkdir(parents=True)
                (case/'candidate.py').write_bytes(data);(case/'dumps/ascendc.cpp').write_text('// synthetic CPP\n');(case/'dumps/binary.o').write_bytes(b'synthetic ELF')
                (case/'dumps/ascir.mlir').write_text('asc.memory_consumed = {UB = 1536 : i64}')
                binding={'binary_sha256':tool.sha(case/'dumps/binary.o'),'compiler_memory_bytes':{'UB':1536}}
                self.save(case/'qualification-real-telemetry.json',{'phase':'real','status':'passed','actual_runtime_launches_observed':1,'binding':binding,'events':[{'kind':'resource_query','value':64}]})
                self.save(case/'qualification-suppressed-telemetry.json',{'phase':'suppressed','status':'raised','actual_runtime_launches_observed':0})
                checks.append({'fixture':fixture['id'],'seed':seed,'exit_code':0,'verification_level':'numerically_verified','launcher_calls':1,'evidence':{
                    'execution':{'exit_code':0,'report':{'mock_launch':False,'numerical_checks':[{}]*4},'log':''},
                    'control':{'exit_code':1,'report':{'mock_launch':True,'numerical_failure':True},'log':''}}})
        self.save(scope/'out/result.json',{'candidate_sha256':self.sources[name],'status':'passed','checks':checks})
        proof=tool.qualification(scope,self.sources[name])
        self.save(scope/'public-receipt.json',{'action':name,'status':'passed','qualification':proof})
    def one(self,position):
        action=COMPARE_ACTIONS[position];fixture=self.fixtures[position];fid=fixture['id'];scope=self.work/action
        tool.prepare_comparison(scope,self.work,self.cpu,fid)
        command=tool.command_spec(scope,['/bundle:/bundle:ro'],tool.TIMING_INNER,360,32)
        self.save(scope/'command.json',command);self.save(scope/'run-intent.json',{'argv':command['argv'],'kernel_launches_permitted':32})
        self.save(scope/'terminal.json',self.terminal());manifest=json.loads((scope/'bundle/manifest.json').read_text());schedule=manifest['schedule']
        ratio=math.exp(math.log(100)-math.log(97))
        result={'status':'passed','source_manifest_sha256':tool.sha(scope/'bundle/manifest.json'),
                'series':[{'series':dict(spec,session='baseline'if spec['arm']=='A'else'variant',index=i),
                           'report':{'status':'pass','launcher_calls':4,'numerical_checks':[{}]*16},
                           'samples':[100,101,100]if spec['arm']=='A'else[97,98,97],'median':100 if spec['arm']=='A'else 97}
                          for i,spec in enumerate(schedule)],
                'controls':[{'key':k+'/'+fid,'code':1,'report':{'mock_launch':True,'numerical_failure':True}}for k in self.sources],
                'analysis':[{'fixture':fid,'order_A_over_B':{'ABBA':ratio,'BAAB':ratio},'primary_A_over_B':math.sqrt(ratio*ratio),'order_direction_agrees':True,'no_shape_pooling':True}]}
        self.save(scope/'out/result.json',result);self.receipt(position)
        return scope,result
    def receipt(self,position):
        scope=self.work/COMPARE_ACTIONS[position]
        self.save(scope/'public-receipt.json',{'status':'passed','action':COMPARE_ACTIONS[position],'fixture_id':self.ids[position],
                    'result_sha256':tool.sha(scope/'out/result.json'),'manifest_sha256':tool.sha(scope/'bundle/manifest.json'),
                    'terminal_sha256':tool.sha(scope/'terminal.json'),'command_sha256':tool.sha(scope/'command.json'),
                    'bundle_index_sha256':tool.sha(scope/'bundle/input-hashes.json'),'parent_binding':tool.comparison_parent(self.work),
                    'sources':self.sources,'real_passes':32,'samples':24,'rejecting_controls':2})
    def test_all_three_exact_results_aggregate_without_pooling(self):
        for i in range(3):self.one(i)
        result,receipts=tool.collected_comparison(self.work,self.ids)
        self.assertEqual(result['counts'],{'real_passes':96,'samples':72,'rejecting_controls':6})
        self.assertEqual(list(receipts),list(COMPARE_ACTIONS));self.assertTrue(all(receipts.values()))
        self.assertEqual(select(result,self.ids)['selected'],'variant')
    def test_missing_third_keeps_baseline_and_incomplete(self):
        self.one(0);self.one(1)
        result,receipts=tool.collected_comparison(self.work,self.ids)
        self.assertEqual(result['status'],'incomplete');self.assertIsNone(receipts['compare-3'])
        self.assertEqual(select(result,self.ids)['selected'],'baseline')
    def test_failed_shape_retained_unknown_stops(self):
        self.one(0);scope,_=self.one(1)
        self.save(scope/'public-receipt.json',{'status':'failed'})
        result,receipts=tool.collected_comparison(self.work,self.ids)
        self.assertEqual(result['status'],'incomplete');self.assertIsNotNone(receipts['compare-2'])
        self.save(scope/'public-receipt.json',{'status':'unknown'})
        with self.assertRaisesRegex(AssertionError,'Unresolved'):tool.collected_comparison(self.work,self.ids)
    def test_reordered_duplicate_or_wrong_fixture_rejected(self):
        scope,result=self.one(0)
        mutations=[lambda r:r['series'].reverse(),lambda r:r['series'].__setitem__(1,copy.deepcopy(r['series'][0])),
                   lambda r:r['analysis'][0].update(fixture=self.ids[1])]
        for mutate in mutations:
            value=copy.deepcopy(result);mutate(value);self.save(scope/'out/result.json',value)
            with self.assertRaises(AssertionError):tool.comparison_result(scope,self.ids[0])
        self.save(scope/'out/result.json',result)
        with self.assertRaises(AssertionError):tool.comparison_result(scope,self.ids[1])
    def test_counts_negative_controls_and_finite_samples_required(self):
        scope,result=self.one(0)
        mutations=[lambda r:r['series'].pop(),lambda r:r['series'][0]['samples'].pop(),
                   lambda r:r['series'][0]['report'].update(launcher_calls=3),lambda r:r['controls'].pop(),
                   lambda r:r['controls'][0]['report'].update(mock_launch=False),
                   lambda r:r['series'][0]['samples'].__setitem__(0,float('nan'))]
        for mutate in mutations:
            value=copy.deepcopy(result);mutate(value);self.save(scope/'out/result.json',value)
            with self.assertRaises(AssertionError):tool.comparison_result(scope,self.ids[0])
    def test_changed_source_or_result_after_receipt_rejected(self):
        for i in range(3):self.one(i)
        path=self.work/'compare-2/out/result.json';original=path.read_bytes();path.write_bytes(original+b' ')
        with self.assertRaises(AssertionError):tool.collected_comparison(self.work,self.ids)
        path.write_bytes(original)
        (self.work/'variant/candidate/candidate.py').write_text('replacement')
        with self.assertRaises((AssertionError,ValueError)):tool.collected_comparison(self.work,self.ids)
    def test_fixture_order_and_token_budgets_unchanged(self):
        for i in range(3):self.one(i)
        with self.assertRaises(AssertionError):tool.collected_comparison(self.work,list(reversed(self.ids)))
        from consumer_interface import LIMITS
        self.assertEqual((LIMITS['rounds_per_attempt'],LIMITS['max_output_tokens'],LIMITS['output_tokens_per_response']),(12,200000,32768))
        self.assertEqual(sum(CAPS.values()),1750)
        self.assertLess(sum(CAPS.values())+900,LIMITS['generation_seconds_per_attempt'])
    def test_preparation_selects_one_shape_and_creates_no_intent(self):
        scope=self.work/'prepared-only'
        mounts,inner=tool.prepare_comparison(scope,self.work,self.cpu,self.ids[1])
        manifest=json.loads((scope/'bundle/manifest.json').read_text())
        self.assertEqual({r['fixture_id']for r in manifest['schedule']},{self.ids[1]})
        self.assertEqual(len(manifest['schedule']),8)
        self.assertEqual(sum(r['suppressed_before_warmup']for r in manifest['schedule']),2)
        self.assertEqual(inner[:4],['timeout','-k','5','330'])
        self.assertFalse((scope/'run-intent.json').exists())
    def test_contradictory_analysis_cannot_choose_variant(self):
        scope,result=self.one(0)
        for row in result['series']:row.update(samples=[100,101,100],median=100)
        self.save(scope/'out/result.json',result);self.receipt(0)
        with self.assertRaises(AssertionError):tool.collected_comparison(self.work,self.ids)
    def test_one_ulp_analysis_is_accepted_but_selection_uses_canonical(self):
        scope,result=self.one(0);expected=copy.deepcopy(result['analysis'])
        row=result['analysis'][0]
        for order in ('ABBA','BAAB'):row['order_A_over_B'][order]=math.nextafter(row['order_A_over_B'][order],math.inf)
        row['primary_A_over_B']=math.nextafter(row['primary_A_over_B'],math.inf)
        self.save(scope/'out/result.json',result);self.receipt(0)
        returned,_,_=tool.comparison_result(scope,self.ids[0])
        self.assertEqual(returned['analysis'],expected)
        self.assertNotEqual(returned['analysis'],result['analysis'])
        self.assertEqual(json.loads((scope/'out/result.json').read_text())['analysis'],result['analysis'])
    def test_ratio_tolerance_does_not_relax_exact_medians_or_boolean_keys(self):
        scope,result=self.one(0)
        for mutate in (lambda r:r['series'][0].update(median=math.nextafter(100,math.inf)),lambda r:r['analysis'][0].update(no_shape_pooling=1),lambda r:r['analysis'][0].update(extra=True)):
            changed=copy.deepcopy(result);mutate(changed);self.save(scope/'out/result.json',changed)
            with self.assertRaises(AssertionError):tool.comparison_result(scope,self.ids[0])
    def test_foreign_parent_runtime_and_qualification_rejected_even_if_reindexed(self):
        scope,result=self.one(0);manifest_path=scope/'bundle/manifest.json';original=manifest_path.read_bytes()
        for mutate in (lambda m:m['parent_binding'].update(attempt='/foreign'),lambda m:m['runtime'].update(image='foreign-runtime'),lambda m:m['secondary']['baseline'].update(result_sha256='0'*64)):
            manifest=json.loads(original);mutate(manifest);self.save(manifest_path,manifest)
            result['source_manifest_sha256']=tool.sha(manifest_path);self.save(scope/'out/result.json',result)
            self.save(scope/'bundle/input-hashes.json',tool.tree_hashes(scope/'bundle',exclude=('input-hashes.json',)))
            self.receipt(0)
            with self.assertRaises(AssertionError):tool.collected_comparison(self.work,self.ids)
        manifest_path.write_bytes(original)
    def test_checker_fixture_or_helper_changed_self_index_does_not_authorize(self):
        scope,result=self.one(0)
        for relative in ('probe/checker/execute_kernel.py','probe/fixtures.json','series_control.py'):
            path=scope/'bundle'/relative;original=path.read_bytes();path.write_bytes(original+b' ')
            self.save(scope/'bundle/input-hashes.json',tool.tree_hashes(scope/'bundle',exclude=('input-hashes.json',)));self.receipt(0)
            with self.assertRaises(AssertionError):tool.collected_comparison(self.work,self.ids)
            path.write_bytes(original)
    def test_missing_or_changed_terminal_rejected(self):
        scope,_=self.one(0);path=scope/'terminal.json';original=path.read_bytes();path.unlink()
        with self.assertRaises(FileNotFoundError):tool.collected_comparison(self.work,self.ids)
        for field,value in (('exit_code',1),('original_client_reaped',False),('original_container_absent',False),('name_absent',False),('errors',['failed'])):
            data=json.loads(original);data[field]=value;self.save(path,data);self.receipt(0)
            with self.assertRaises(AssertionError):tool.collected_comparison(self.work,self.ids)
    def test_changed_qualification_and_session_rejected(self):
        self.one(0);path=self.work/'baseline/out/result.json';original=path.read_bytes();path.write_bytes(original+b' ')
        with self.assertRaises(AssertionError):tool.collected_comparison(self.work,self.ids)
        path.write_bytes(original)
        path=self.recipe/'session-config.json';data=json.loads(path.read_text());data['attempt']='/foreign';self.save(path,data)
        with self.assertRaises(AssertionError):tool.collected_comparison(self.work,self.ids)

if __name__=='__main__':unittest.main()
