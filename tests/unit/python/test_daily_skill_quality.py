from datetime import date
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
import daily_skill_quality as quality
import evidence_dashboard as dashboard
import swiglu_evidence


def run(n, stamp):
    return {'id':n,'run_attempt':1,'head_sha':str(n)*40,'created_at':stamp,'status':'completed',
            'event':'schedule','path':quality.WORKFLOW,'head_branch':'main'}


def observation(r):
    return {'run_id':r['id'], 'attempt':1, 'revision':r['head_sha'], 'archive_sha256':str(r['id'])*64,
            'protocol':'a'*64, 'model':'glm', 'skills':'b'*64, 'cells':{'add':'passed','reduce':'failed'},
            'sessions':{'s'+str(r['id'])}}


class DailyQuality(unittest.TestCase):
    def test_adjacent_moscow_days_even_on_same_utc_date(self):
        older = run(1,'2026-11-10T00:42:00Z')
        newer = run(2,'2026-11-10T22:22:00Z')
        result = quality.report(date(2026,11,11), lambda *_:[older,newer], observation)
        self.assertEqual(result['status'],'same-skills')
        self.assertEqual(result['dates'],['2026-11-10','2026-11-11'])
        self.assertNotIn('sessions', result['runs'][0])

    def test_missing_duplicate_and_failed_download_keep_gap(self):
        older = run(1,'2026-11-10T12:00:00Z')
        newer = run(2,'2026-11-11T12:00:00Z')
        manual = dict(newer,event='workflow_dispatch')
        for rows in ([older], [older,manual], [older,newer,newer], [older,dict(newer,status='in_progress')]):
            self.assertEqual(quality.report(date(2026,11,11),lambda *_:rows,observation)['status'],'unavailable')
        def broken(_): raise OSError()
        self.assertEqual(quality.report(date(2026,11,11),lambda *_:[older,newer],broken)['status'],'unavailable')

    def test_scope_and_directions(self):
        a=observation(run(1,'')); b=observation(run(2,''))
        b['skills']='c'*64
        self.assertEqual(quality.compare(a,b)['status'],'tie')
        b['cells']['reduce']='passed'
        self.assertEqual(quality.compare(a,b)['status'],'improved')
        b['cells']['add']='failed'
        self.assertEqual(quality.compare(a,b)['status'],'mixed')
        b['cells']['reduce']='failed'
        self.assertEqual(quality.compare(a,b)['lost'],['add'])
        self.assertEqual(quality.compare(a,b)['status'],'regressed')
        self.assertEqual(quality.compare(a,dict(b,skills=a['skills']))['status'],'same-skills')
        for changed in ({'protocol':'new-wheel'},{'sessions':a['sessions']},{'archive_sha256':a['archive_sha256']}):
            self.assertEqual(quality.compare(a,dict(b,**changed))['status'],'incompatible')

    def test_hardware_compares_times_and_scores_only_for_matched_publications(self):
        dates=['2026-11-10','2026-11-11']
        def row(i,t):
            return {'kind':'hardware','availability':'available','ci_url':str(i),'release_url':'link'+str(i),
                    'first_hardware':{'generated_at':dates[i]+'T10:00:00Z','protocol':'same','skills':str(i)},
                    'tasks':[{'id':f'level1/swi_glu_{c}','baseline_us':1,'elapsed_us':t} for c in range(1,21)], 'service_score':4+i}
        a,b=row(0,2),row(1,1)
        result=quality.hardware_pair(dates,[a,b])
        self.assertEqual((result['previous_over_current'],result['scores']), (2,[4,5]))
        self.assertEqual(quality.hardware_pair(dates,[a,b,b])['status'],'unavailable')
        self.assertEqual(quality.hardware_pair(dates,[a,dict(b,first_hardware=None)])['status'],'unavailable')
        self.assertEqual(quality.hardware_pair(dates,[a,dict(b,tasks=b['tasks'][1:])])['status'],'incompatible')
        b['first_hardware']['protocol']='different runtime'
        self.assertEqual(quality.hardware_pair(dates,[a,b])['status'],'incompatible')

    def test_first_hardware_pair_normalizes_paths_and_excludes_repair(self):
        class Files:
            def __init__(self, root):
                self.members = {
                    'evidence/model-config.json': {'skills':{'paths':[root+'/context/skills']}},
                    'evidence/package/job.json': {'job':{'id':'job', 'results':{
                        'result_stage':'final', 'anti_cheat_failed_cases':0, 'perf_metric':'time',
                        'score_weight_profile':{'mode':'default'}, 'setup_info':{'environment':{'cann':'9.2'}}}}},
                    'evidence/package/task.json': b'fixed task',
                    'evidence/generation/attempt-1/prompt.txt': ('Read '+root+'/context/task').encode(),
                }
            def raw(self, name):
                return self.members[name]
            def json(self, name):
                return self.members[name]

        def projection(archive, skill, *, attempt=1):
            prompt = 'generation/attempt-1/prompt.txt'
            generation = {'attempt':attempt, 'skill_hashes':{'guide':skill}, 'model':'glm',
                          'provider_model_revision':'unknown', 'opencode_version':'1.19.0',
                          'files':{prompt:dashboard.digest(archive.raw('evidence/'+prompt))}}
            origin = {'source_hashes':{'integrations/cannbench/ci/operator.py':'fixed'},
                      'context_hashes':{'skills/guide/SKILL.md':skill, 'API.md':'fixed'},
                      'pyasc':{'version':'pinned'}, 'created_at':'2026-11-10T12:00:00Z'}
            return dashboard.first_hardware_protocol(archive, origin, generation, {
                'job_id':'job','hardware':'950pr','benchmark_version':'2.0.0'})

        a, b = Files('/a'), Files('/b')
        left, right = projection(a,'old'), projection(b,'new')
        self.assertEqual(left['protocol'], right['protocol'])
        self.assertNotEqual(left['skills'], right['skills'])
        b.members['evidence/generation/attempt-1/prompt.txt'] = b'Other request'
        self.assertNotEqual(left['protocol'], projection(b,'new')['protocol'])
        self.assertIsNone(projection(b,'new',attempt=2))
        b.members['evidence/repair-lineage.json'] = {}
        self.assertIsNone(projection(b,'new'))

    def test_device_times_reject_missing_duplicate_and_failed_cases(self):
        rows=[{'case':i,'correct':True,'median_us':2,'samples_us':[1,2,3]} for i in range(1,21)]
        self.assertEqual(len(swiglu_evidence.times(rows)),20)
        for invalid in (rows[:-1], rows+[rows[0]], rows[:-1]+[dict(rows[-1],correct=False)],
                        rows[:-1]+[dict(rows[-1],median_us=3)]):
            with self.assertRaises(ValueError):
                swiglu_evidence.times(invalid)

    def test_service_ratio_uses_only_generated_compute_and_same_job_times(self):
        import hashlib
        cases = [{'id':f'level1/swi_glu_{i}', 'elapsed_us':10 if i==12 else 2,
                  'baseline_us':1, 'kernel':'_swiglu_pair_groups' if i==12 else 'rows'} for i in range(1,21)]
        report = {'schema':1, 'evaluation_run':'1234', 'origin_run':'1233','hardware':'950pr',
                  'benchmark_version':'2.0.0','job_id':'job_abc','submission_id':'sub_abc',
                  'candidate_sha256':'a'*64,'package_sha256':'b'*64,'cases':cases,'service_score':17,
                  'generated_ratio':0.5,'revision':'d'*40,'model':'gateway/test',
                  'finished_at':'2026-11-10T08:00:00Z'}

        def project():
            raw = json.dumps(report).encode()
            return swiglu_evidence.service_projection(lambda _:raw, {
                'release':'qualification-swiglu-performance-1234','sha256':hashlib.sha256(raw).hexdigest()})

        self.assertEqual(project()['ratio'],0.5)
        for field,value in [('kernel','_swiglu_pair_groups'),('id','level1/swi_glu_2'),('baseline_us',0)]:
            with self.subTest(field=field), patch.dict(cases[0],{field:value}), self.assertRaises(ValueError):
                project()
        report['generated_ratio']=0.893
        with self.assertRaises(ValueError):
            project()

    def test_current_publisher_uses_recipe_name_in_tag(self):
        r=dict(run(4,'2026-11-11T00:00:00Z'),event='workflow_dispatch',path='.github/workflows/ci.yml')
        release={'tag_name':'qualification-swi_glu-4','target_commitish':r['head_sha'],'draft':False,
                 'assets':[{'name':'run-4-qualification.tar.gz','size':120,'digest':'sha256:'+'b'*64}]}
        with patch.object(quality,'runs_for_dates',return_value=[r]), patch.object(quality,'gh',return_value=json.dumps(release)):
            found=dashboard.recent_hardware(['2026-11-10','2026-11-11'])
        self.assertEqual(found[0]['release'],release['tag_name'])
