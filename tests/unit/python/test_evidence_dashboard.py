from copy import deepcopy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
import evidence_dashboard as dash


def raw_json(value): return json.dumps(value).encode()


def archive(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for name, raw in files.items():
            if not isinstance(raw, bytes): raw = raw_json(raw)
            item = tarfile.TarInfo(name); item.size = len(raw); tar.addfile(item, io.BytesIO(raw))
    return stream.getvalue()


class Dashboard(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        self.entry = {'id':'review-1','kind':'review','release':'qualification-test','asset':'public.tar.gz','sha256':'a'*64,'run_id':1}
        self.spec = {'schema_version':1,'repository':dash.REPOSITORY,'max_age_hours':192,'sources':[self.entry]}
        self.catalog = self.root/'sources.json'; self.write_catalog()
        self.now = datetime(2026,11,6,tzinfo=timezone.utc)

    def write_catalog(self): self.catalog.write_bytes(raw_json(self.spec))

    def reviewed(self, accepted=True, requests=None):
        common = {'model':{'requested':'gateway/test','expected_response':'test-1','profile':'test'},
            'runtime':{'backend':'Model','source_revision':'a'*40,'image':'image@sha256:'+'a'*64},
            'source_sha256':{'skills/example/SKILL.md':'a'*64},'cells':[{'id':'add','module':'test_add.py','cases':['tail']} ]}
        plan={'kind':'review-generation-plan','schema_version':1,'common':common,'hardware_invoked':False,'checkout_revision':'b'*40}
        plan['plan_sha256']=dash.digest(json.dumps(plan,sort_keys=True,separators=(',',':')).encode())
        report={'status':'accepted' if accepted else 'failed','validated_acceptance':accepted,'plan_sha256':plan['plan_sha256'],
            'candidate_sha256':'c'*64,'observed_model':'test-1','elapsed_seconds':12.0,'independent_model':False,'repair_attempts':0}
        inner=archive({'run/plan.json':plan,'run/report.json':report,'run/state.json':{'finished_at':'2026-11-05T10:00:00Z'},
            'run/cell/usage.json':{'requests':requests if requests is not None else [{'usage':{'observed_model':'test-1'}}]},
            'run/private/credential.json':{'secret':'NEVER-PUBLISH'},
            'run/arbitrary.py':b"raise RuntimeError('NEVER EXECUTE')"})
        verification={'run':1,'head':'b'*40,'archive_sha256':dash.digest(inner),'report':report,
            'output_input_checks':3,'execution_launches':1,'suppressed_launch_control_exit':1}
        files={'generation.tar.gz':inner,'ci-run.json':raw_json({'headSha':'b'*40,'status':'completed'}),
            'verification.json':raw_json(verification)}
        outer={'result/'+n:v for n,v in files.items()};outer['result/files.json']={n:dash.digest(v) for n,v in files.items()}
        raw=archive(outer);self.entry['sha256']=dash.digest(raw);self.write_catalog();return raw

    def test_success_is_public_projection_not_archive_execution(self):
        raw=self.reviewed();result=dash.build(self.catalog,self.root/'out',loader=lambda _:raw,now=self.now)
        record=result['records'][0]
        self.assertEqual(record['availability'],'available');self.assertEqual(record['qualified'],1)
        self.assertEqual(record['level'],'simulator-numerical');self.assertIsNone(record['service_score'])
        text=(self.root/'out/data.json').read_text();self.assertNotIn('NEVER-PUBLISH',text);self.assertNotIn('arbitrary.py',text)

    def source_archive(self, raw=b"raise RuntimeError('DISPLAY ONLY')\n", sha=None, path='cycle-1/writer/candidate.py', twin=None):
        sha = sha or dash.digest(raw)
        plan = {'checkout_revision':'b'*40}
        plan['plan_sha256'] = dash.digest(json.dumps(plan,sort_keys=True,separators=(',',':')).encode())
        files = {'run/plan.json':plan, 'run/report.json':{'candidate_sha256':sha},
            'run/state.json':{'accepted_candidate_sha256':sha,'cycles':[{'status':'accepted','candidate_sha256':sha,'candidate':path}]},
            'run/'+path:raw, 'run/cycle-1/verification/candidate.py':raw if twin is None else twin,
            'run/private/credential.json':{'secret':'NEVER-PUBLISH'}, 'run/arbitrary.py':b'PRIVATE'}
        return dash.Archive(archive({'result/generation.tar.gz':archive(files)}))

    def test_kernel_source_is_exact_hash_bound_text_without_archive_execution(self):
        result=dash.kernel_sources(self.source_archive(),{'kind':'review'})
        self.assertEqual(result['status'],'available')
        self.assertEqual(result['files'][0]['text'],"raise RuntimeError('DISPLAY ONLY')\n")
        self.assertEqual(result['files'][0]['sha256'],dash.digest(result['files'][0]['text'].encode()))
        self.assertNotIn('PRIVATE',json.dumps(result))
        raw=self.reviewed();row=dash.build(self.catalog,self.root/'legacy-source',loader=lambda _:raw,now=self.now)['records'][0]
        self.assertEqual(row['availability'],'available');self.assertEqual(row['qualified'],1)
        self.assertEqual(row['kernel_sources']['status'],'unavailable')

    def test_kernel_source_rejects_changed_twin_private_path_binary_and_oversize(self):
        for kwargs in ({'sha':'a'*64}, {'twin':b'changed'}, {'path':'private/candidate.py'},
                       {'raw':b'\xff'}, {'raw':b'a\x00b'}, {'raw':b'x'*(dash.MAX_KERNEL_SOURCE+1)}):
            with self.subTest(kwargs=list(kwargs)):
                result=dash.kernel_sources(self.source_archive(**kwargs),{'kind':'review'})
                self.assertEqual(result['status'],'unavailable');self.assertEqual(result['files'],[])
                self.assertNotIn('PRIVATE',json.dumps(result))

    def test_trial_sources_keep_failed_candidates_and_explicit_missing_attempts(self):
        def seal(plan):
            plan['plan_sha256']=dash.digest(json.dumps(plan,sort_keys=True,separators=(',',':')).encode());return plan
        plan=seal({'checkout_revision':'b'*40,'trials':[{'id':'trial','cell':'add'}]})
        child=seal({'checkout_revision':'b'*40})
        raw=b'# Failed candidate, display only\n';sha=dash.digest(raw)
        state={'plan_sha256':child['plan_sha256'],'cells':[{'id':'add','attempts':[
            {'number':1,'status':'failed','candidate_sha256':sha},{'number':2,'status':'failed'}]}]}
        files={'run/plan.json':plan,'run/report.json':{},'run/trial/plan.json':child,'run/trial/state.json':state,
            'run/trial/add/attempt-1/candidate.py':raw,'run/trial/add/attempt-1/project/candidate.py':raw}
        def project():return dash.kernel_sources(dash.Archive(archive({'result/generation.tar.gz':archive(files)})),{'kind':'models'})
        result=project();self.assertEqual(result['status'],'partial')
        self.assertEqual(result['files'][0]['outcome'],'failed');self.assertEqual(result['missing'][0]['id'],'trial/attempt-2')
        state['plan_sha256']='a'*64
        self.assertEqual(project()['files'],[])

    def test_fixture_source_is_not_generated_and_unknown_hardware_path_is_unavailable(self):
        self.assertEqual(dash.kernel_sources(None,{'kind':'target'})['status'],'not-applicable')
        a=dash.Archive(archive({'evidence/generation.json':{'original_candidate':'private/candidate.py','candidate_sha256':'a'*64}}))
        self.assertEqual(dash.kernel_sources(a,{'kind':'hardware'})['status'],'unavailable')

    def test_wire_settings_validate_the_original_request_cap(self):
        sys.path.insert(0, str(ROOT/'tests/tools'))
        from generation_provider import request_settings
        wire=request_settings({'max_tokens':24000,'reasoning_effort':'high','thinking':{'type':'enabled'}})
        request={'max_tokens':24000,'request_settings':wire,'usage':{'observed_model':'test-1'}}
        raw=self.reviewed(requests=[request])
        row=dash.build(self.catalog,self.root/'valid-wire',loader=lambda _:raw,now=self.now)['records'][0]
        self.assertEqual(row['availability'],'available');self.assertEqual(row['wire_settings'],[wire])
        for cap in (None, 1, True):
            changed=dict(request,max_tokens=cap);raw=self.reviewed(requests=[changed])
            row=dash.build(self.catalog,self.root/str(cap),loader=lambda _:raw,now=self.now)['records'][0]
            self.assertEqual(row['availability'],'unavailable');self.assertNotIn('wire_settings',row)

    def test_missing_malformed_and_checksum_failure_are_visible_without_acceptance(self):
        good=self.reviewed()
        for index,loader in enumerate((lambda _:b'invalid',lambda _:good+b'changed',lambda _:(_ for _ in ()).throw(OSError('PRIVATE DETAIL')))):
            result=dash.build(self.catalog,self.root/str(index),loader=loader,now=self.now)
            self.assertEqual(result['records'][0]['availability'],'unavailable')
            self.assertNotIn('qualified',result['records'][0]);self.assertNotIn('PRIVATE DETAIL',json.dumps(result))

    def test_failed_review_does_not_become_numerically_accepted(self):
        raw=self.reviewed(False);record=dash.build(self.catalog,self.root/'out',loader=lambda _:raw,now=self.now)['records'][0]
        self.assertEqual(record['availability'],'unavailable');self.assertNotIn('qualified',record)

    def test_stale_result_separate_from_availability(self):
        raw=self.reviewed();later=datetime(2026,12,6,tzinfo=timezone.utc)
        row=dash.build(self.catalog,self.root/'out',loader=lambda _:raw,now=later)['records'][0]
        self.assertEqual(row['freshness'],'stale');self.assertEqual(row['availability'],'available')

    def test_safe_rebuild_does_not_overwrite_previous_snapshot(self):
        raw=self.reviewed();first=dash.build(self.catalog,self.root/'old',loader=lambda _:raw,now=self.now)
        old=(self.root/'old/data.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'new external'):dash.build(self.catalog,self.root/'old',loader=lambda _:raw,now=self.now)
        second=dash.build(self.catalog,self.root/'new',loader=lambda _:b'broken',now=self.now)
        self.assertEqual((self.root/'old/data.json').read_bytes(),old)
        self.assertEqual(first['records'][0]['availability'],'available');self.assertEqual(second['records'][0]['availability'],'unavailable')

    def test_catalog_rejects_arbitrary_urls_duplicate_ids_and_private_fields(self):
        for field,value in [('release','../../secrets'),('sha256','invalid'),('kind','compile-accepted')]:
            spec=deepcopy(self.spec);spec['sources'][0][field]=value;self.catalog.write_bytes(raw_json(spec))
            with self.assertRaises(ValueError):dash.catalog(self.catalog)
        self.spec['sources'].append(deepcopy(self.entry));self.write_catalog()
        with self.assertRaises(ValueError):dash.catalog(self.catalog)

    def test_archive_rejects_traversal_links_duplicate_members_and_bombs(self):
        with self.assertRaises(ValueError):dash.Archive(archive({'../escape':b'bad'}))
        for duplicate in (False,True):
            out=io.BytesIO()
            with tarfile.open(fileobj=out,mode='w:gz') as tar:
                member=tarfile.TarInfo('file');member.type=tarfile.SYMTYPE if not duplicate else tarfile.REGTYPE;member.linkname='/tmp/escape'
                tar.addfile(member)
                if duplicate:tar.addfile(member)
            with self.assertRaises(ValueError):dash.Archive(out.getvalue())
        with patch.object(dash,'MAX_EXPANDED',128),self.assertRaises(ValueError):dash.Archive(archive({'large':b'x'*2048}))

    def test_json_rejects_duplicate_keys_nonfinite_and_oversized(self):
        for value in (b'{"a":1,"a":2}',b'{"a":NaN}'):
            with self.assertRaises(ValueError):dash.decode(value)
        with patch.object(dash,'MAX_JSON',2),self.assertRaises(ValueError):dash.decode(b'{} ')

    def test_frontend_uses_text_nodes_and_rejects_unexpected_links(self):
        js=(ROOT/'dashboard/app.js').read_text();self.assertIn('textContent',js)
        self.assertNotIn('innerHTML',js);self.assertNotIn('eval(',js)
        self.assertIn("parsed.hostname !== 'github.com'",js);self.assertIn('STALE PAGE',js)

    def test_rejected_projection_does_not_leave_partial_acceptance_fields(self):
        raw=self.reviewed()
        with patch.object(dash,'timestamp',side_effect=ValueError('bad')):
            row=dash.build(self.catalog,self.root/'out',loader=lambda _:raw,now=self.now)['records'][0]
        self.assertEqual(row['availability'],'unavailable');self.assertNotIn('qualified',row)

    def target_archive(self, complete=False, duplicate=False):
        ids=['test_add.py::tail','test_sum.py::small']
        plan={'schema_version':1,'kind':'target-suite-plan','selected':ids[:1],'case_ids':ids,'checkout_revision':'b'*40,
            'excluded_reason':'Initial bounded subset'}
        plan['plan_sha256']=dash.digest(json.dumps(plan,sort_keys=True,separators=(',',':')).encode())
        report={'schema_version':1,'kind':'target-numerical-coverage','plan_sha256':plan['plan_sha256'],
            'outcomes':{ids[0]:{'status':'qualified'},ids[1]:{'status':'excluded'}},'counts':{'qualified':1,'excluded':1},
            'required_cases':2,'full_suite_qualified':complete,'runtime_image':'image@sha256:'+'a'*64,'upstream_revision':'a'*40}
        if duplicate:plan['case_ids']=[ids[0],ids[0]]
        verification={'selected_qualified_cases':1,'checked_at':'2026-11-05T10:00:00Z','total_phase_seconds':2.0,
            'official_assertions':1,'actual_model_launches':1,'rejecting_controls':1}
        files={'run/plan.json':raw_json(plan),'run/coverage.json':raw_json(report),
            'verification-'+plan['checkout_revision'][:7]+'.json':raw_json(verification)}
        files['qualification-target-subset-files.json']=raw_json({n:dash.digest(v) for n,v in files.items()})
        return dash.Archive(archive(files))

    def test_partial_fixture_census_stays_partial_and_not_model_generation(self):
        result=dash.target(self.target_archive(),{})
        self.assertEqual(result['category'],'upstream-fixture');self.assertEqual(result['required'],2)
        self.assertEqual(result['qualified'],1);self.assertEqual(result['counts']['excluded'],1)
        self.assertEqual(result['status'],'partial');self.assertEqual(result['models'],[])

    def test_contradictory_full_suite_and_missing_inventory_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Contradictory'):dash.target(self.target_archive(complete=True),{})
        with self.assertRaises(ValueError):dash.target(self.target_archive(duplicate=True),{})

    def test_hardware_timing_ratio_is_not_service_score_and_case_gaps_fail(self):
        candidate=b'candidate'
        origin={'origin_run_id':'1','repository':dash.REPOSITORY,'skill_stack_commit':'b'*40,
            'pyasc':{'pyasc_commit':'a'*40,'runtime_bundle_sha256':'a'*64,'compile_image_id':'sha256:'+'a'*64,'profile':'test'}}
        gen={'candidate_sha256':dash.digest(candidate),'original_candidate':'candidate.py','model':'gateway/test',
            'provider_model_revision':None,'model_configuration_sha256':'a'*64,'skill_hashes':{}}
        cases=[{'case_id':'add-1','status':'success','baseline_perf_us':2.0,'elapsed_us':4.0}]
        report={'schema_version':1,'source':'cannbench-hardware','gate_passed':True,'errors':[],
            'operators':[{'cases':cases}],'required_cases':1,'correct_cases':1,'measured_cases':1,'true_geometric_mean_speedup':.5,
            'benchmark_slug':'official','benchmark_version':'2.0','hardware':'950pr',
            'finished_at':'2026-11-05T10:00:02','claimed_at':'2026-11-05T10:00:00','queued_at':'2026-11-05T09:00:00'}
        acceptance={'passed':True,'origin':origin,'generation':gen,'hardware':deepcopy(report),'service_score':33.0}
        files={'evidence/acceptance.json':acceptance,'evidence/run.json':origin,'evidence/package/report.json':report,
            'evidence/generation.json':gen,'evidence/candidate.py':candidate,
            'evidence/model-config.json':{'model':'gateway/test','provider':{'gateway':{'models':{'test':{}}}}}}
        gen['model_configuration_sha256']=dash.digest(raw_json(files['evidence/model-config.json']))
        result=dash.hardware(dash.Archive(archive(files)),{'run_id':1})
        self.assertEqual(result['baseline_time_ratio'],.5);self.assertEqual(result['service_score'],33.0)
        self.assertEqual(result['queue_seconds'],3600);self.assertEqual(result['elapsed_seconds'],2)
        report['required_cases']=2
        with self.assertRaisesRegex(ValueError,'case coverage'):dash.hardware(dash.Archive(archive(files)),{'run_id':1})

    def test_profile_projection_keeps_archived_settings_and_rejects_private_options(self):
        model={'options':{'reasoningEffort':'low','thinking':{'type':'enabled'}},'limit':{'output':8192}}
        config={'model':'gateway/test','provider':{'gateway':{'models':{'test':model}}}}
        result=dash.profile_settings(config);self.assertEqual(result['options']['reasoningEffort'],'low')
        model['options']['api_key']='PRIVATE'
        with self.assertRaises(ValueError):dash.profile_settings(config)

    def test_public_counts_reject_boolean_denominator_unknown_status_and_contradiction(self):
        raw=self.reviewed();valid=dash.generated(dash.Archive(raw),self.entry);dash.validate_projection(valid)
        for change in ({'required':True},{'counts':{'qualified':2}},{'counts':{'PRIVATE':1}},{'elapsed_seconds':float('inf')},
                       {'level':'compile-only'},{'stack_revision':'main'}):
            value=dict(valid,**change)
            with self.subTest(change=change),self.assertRaises(ValueError):dash.validate_projection(value)

    def test_truncated_compressed_record_is_unavailable_without_losing_healthy_record(self):
        healthy=self.reviewed(); broken=healthy[:-8]
        self.spec['sources'].append(dict(self.entry,id='truncated',sha256=dash.digest(broken)))
        self.write_catalog(); raw=iter([healthy,broken])
        rows=dash.build(self.catalog,self.root/'out',loader=lambda _:next(raw),now=self.now)['records']
        self.assertEqual([r['availability'] for r in rows],['available','unavailable'])
        self.assertNotIn('qualified',rows[1])

    def paired(self, status='complete', delta=1.0, interval=None):
        reviewed=dash.Archive(self.reviewed())
        inner=dash.Archive(reviewed.raw('result/generation.tar.gz',dash.MAX_ARCHIVE))
        plan=inner.json('run/plan.json'); plan.pop('plan_sha256')
        plan.update(kind='paired-generation-plan',trials=[{'id':'a'},{'id':'b'}],spec={'repetitions':1})
        plan['plan_sha256']=dash.digest(json.dumps(plan,sort_keys=True,separators=(',',':')).encode())
        comparison={'valid_pairs':1 if status=='complete' else 0,'denominator':1,
            'coverage_gaps':[] if status=='complete' else ['b'],'qualification_delta':delta,
            'paired_rows':[{'qualification_delta':1.0}],
            'uncertainty':{'method':'Hoeffding 95%, independent paired trials, D in [-1,1]',
                'interval':[-1,1] if interval is None else interval}}
        report={'plan_sha256':plan['plan_sha256'],'status':status,'trial_denominator':2,
            'observations':{'a':{'status':'valid','outcome':'qualified'},
                'b':{'status':'valid' if status=='complete' else 'unresolved','outcome':'candidate-failed' if status=='complete' else 'unresolved'}},
            'comparisons':{'B-A':comparison}}
        files={n:inner.raw(n,dash.MAX_ARCHIVE) for n in inner.members}
        files.update({'run/plan.json':raw_json(plan),'run/report.json':raw_json(report)})
        bundle=archive(files);v=reviewed.json('result/verification.json')
        v.update(archive_sha256=dash.digest(bundle),outcomes=dict(dash.Counter(o['outcome'] for o in report['observations'].values())),
            elapsed_seconds=12.0,accepted_numerical_checks=3,accepted_execution_launches=1,rejecting_controls=1)
        outer={'generation.tar.gz':bundle,'ci-run.json':reviewed.raw('result/ci-run.json'),'verification.json':raw_json(v)}
        outer['files.json']=raw_json({n:dash.digest(v) for n,v in outer.items()})
        return dash.Archive(archive({'result/'+n:v for n,v in outer.items()}))

    def test_paired_projection_suppresses_incomplete_effect_and_rejects_invented_precision(self):
        entry=dict(self.entry,kind='paired')
        self.assertEqual(dash.generated(self.paired(),entry)['comparisons']['B-A']['interval'],[-1,1])
        result=dash.generated(self.paired(status='incomplete',delta=None),entry)
        self.assertEqual(result['status'],'incomplete');self.assertEqual(result['comparisons'],{})
        with self.assertRaisesRegex(ValueError,'Partial comparison'):
            dash.generated(self.paired(status='incomplete'),entry)
        with self.assertRaisesRegex(ValueError,'Invalid uncertainty'):
            dash.generated(self.paired(interval=[.9,1]),entry)


if __name__=='__main__':unittest.main()
