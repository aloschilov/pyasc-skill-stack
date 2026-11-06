from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import test_target_inventory
import test_target_case

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
import target_suite as suite


class Suite(unittest.TestCase):
    def setUp(self):
        fixture = test_target_inventory.Inventory(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.fixture, self.root = fixture, fixture.root
        env = suite.read(self.root/'reference/images/ENV.json')
        env.update(version='1.0'); env['dev_image'].update(torch='2.7.1+cpu', pytest='8.4.2')
        fixture.write('reference/images/ENV.json', env)
        fixture.write(suite.CONTRACT, suite.derive(fixture.ids, self.root))
        for name in suite.HELPERS[:-1]:
            p = self.root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((ROOT/name).read_bytes())
        self.collection = self.root.parent/'collection'; self.collection.mkdir()
        suite.save(self.collection/'nodeids.json', fixture.ids)
        suite.save(self.collection/'result.json', {'status':'collected','cleanup_confirmed':True,'inputs_unchanged':True,
            'runtime_image':env['dev_image']['ref'],'source_sha256':suite.sources(self.root),'case_count':len(fixture.ids),
            'artifact_sha256':{'nodeids.json':suite.digest((self.collection/'nodeids.json').read_bytes())}})
        self.plan = suite.build_plan(self.collection, shard_size=1, root=self.root)
        self.run = self.root.parent/'run'

    def phase(self, directory, plan, source, nodeid, mode):
        q = test_target_case.Qualification(); q.setUp(); value = deepcopy(q.real if mode == 'real' else q.control)
        value.update(nodeid=nodeid, collected=['reference/asctile/target/'+nodeid],
            source_sha256=suite.sources(source), runtime={'pyasc':'1.0','torch':'2.7.1+cpu','pytest':'8.4.2','launcher_sha256':'a'*64})
        suite.save(directory/(mode+'.json'),value)
        command = {'status':'completed','exit_code':0 if mode=='real' else 1,'cleanup_confirmed':True,
                   'command':['docker','run',plan['inventory']['runtime_image']]}
        suite.save(directory/(mode+'.command.json'),command)
        return command

    def run_one(self, number=0):
        with patch.object(suite,'phase',side_effect=self.phase) as phase:
            result = suite.run_shard(self.plan,self.run,number,root=self.root)
            return result, phase.call_count

    def test_complete_inventory_and_bounded_shards_keep_every_case(self):
        suite.check_plan(self.plan,self.root)
        self.assertEqual([n for s in self.plan['shards'] for n in s],sorted(self.fixture.ids))
        for args in ({'selected':[]},{'selected':self.fixture.ids[:1]}, {'selected':['test_other.py::new'],'reason':'subset'},
                     {'shard_size':16,'seconds':300,'shard_seconds':900}, {'seconds':True}):
            with self.subTest(args=args), self.assertRaises(ValueError): suite.build_plan(self.collection,root=self.root,**args)
        subset = suite.build_plan(self.collection,selected=self.fixture.ids[:1],reason='Bounded initial coverage',root=self.root)
        self.assertEqual(len(subset['case_ids']),3);self.assertEqual(len(subset['selected']),1)

    def test_resume_only_unattempted_cases_and_full_coverage_needs_all(self):
        report,calls = self.run_one()
        self.assertEqual(calls,2);self.assertEqual(report['counts'],{'qualified':1,'not-run':2})
        self.assertFalse(report['full_suite_qualified']);self.assertFalse(report['all_selected_qualified'])
        again,calls = self.run_one();self.assertEqual(calls,0);self.assertEqual(again,report)
        self.run_one(1);result,_ = self.run_one(2)
        self.assertEqual(result['counts'],{'qualified':3});self.assertTrue(result['full_suite_qualified'])

    def test_missing_duplicate_identity_artifact_and_runtime_are_rejected(self):
        self.run_one(); node=self.plan['shards'][0][0];attempt=self.run/'cases'/suite.case_key(node)
        result=suite.read(attempt/'result.json')
        changed=deepcopy(result);changed['nodeid']=self.plan['shards'][1][0];suite.save(attempt/'result.json',changed)
        with self.assertRaisesRegex(ValueError,'identity'):suite.report(self.run)
        suite.save(attempt/'result.json',result)
        raw=(attempt/'real.json').read_bytes();(attempt/'real.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact drift'):suite.report(self.run)
        (attempt/'real.json').write_bytes(raw)
        bad=suite.read(attempt/'real.json');bad['runtime']['pyasc']='other';suite.save(attempt/'real.json',bad)
        result['artifact_sha256']['real.json']=suite.digest((attempt/'real.json').read_bytes());suite.save(attempt/'result.json',result)
        with self.assertRaisesRegex(ValueError,'runtime mismatch'):suite.report(self.run)
        (attempt/'real.json').write_bytes(raw);result['artifact_sha256']['real.json']=suite.digest(raw);suite.save(attempt/'result.json',result)
        (self.run/'cases'/'duplicate').mkdir()
        with self.assertRaisesRegex(ValueError,'Unexpected/duplicate'):suite.report(self.run)

    def test_unresolved_case_blocks_resume_but_preserves_census(self):
        self.run_one();attempt=next((self.run/'cases').iterdir());result=suite.read(attempt/'result.json')
        result['status']='running';suite.save(attempt/'result.json',result)
        with patch.object(suite,'phase',side_effect=AssertionError('Must not launch')):
            with self.assertRaisesRegex(ValueError,'Unresolved previous case'):suite.run_shard(self.plan,self.run,1,root=self.root)
        report=suite.report(self.run)
        self.assertEqual(report['counts'],{'unresolved':1,'not-run':2});self.assertFalse(report['full_suite_qualified'])

    def test_unavailable_or_timed_out_case_is_a_gap_not_a_pass(self):
        def timeout(directory,plan,source,nodeid,mode):
            value={'status':'timed-out','cleanup_confirmed':True}
            suite.save(directory/(mode+'.command.json'),value);return value
        with patch.object(suite,'phase',side_effect=timeout):result=suite.run_shard(self.plan,self.run,0,root=self.root)
        self.assertEqual(result['counts'],{'timed-out':1,'not-run':2})
        self.assertFalse(result['full_suite_qualified'])

    def test_selected_subset_pass_does_not_claim_full_coverage(self):
        self.plan=suite.build_plan(self.collection,selected=self.fixture.ids[:1],reason='Initial bounded subset',root=self.root)
        result,_=self.run_one()
        self.assertTrue(result['all_selected_qualified']);self.assertFalse(result['full_suite_qualified'])
        self.assertEqual(result['counts'],{'excluded':2,'qualified':1})

    def test_package_parents_are_exported_and_collection_errors_stop_shard(self):
        self.plan=suite.build_plan(self.collection,shard_size=3,root=self.root)
        self.assertIn('reference/__init__.py',self.plan['source_sha256'])
        def bad_collection(directory,plan,source,nodeid,mode):
            self.assertTrue((source/'reference/__init__.py').is_file())
            suite.save(directory/(mode+'.json'),{'phases':{},'collected':[],'exit_code':4})
            cmd={'status':'completed','exit_code':4,'cleanup_confirmed':True}
            suite.save(directory/(mode+'.command.json'),cmd);return cmd
        with patch.object(suite,'phase',side_effect=bad_collection) as mocked:
            result=suite.run_shard(self.plan,self.run,0,root=self.root)
        self.assertEqual(mocked.call_count,1)
        self.assertEqual(result['counts'],{'blocked':1,'not-run':2})


if __name__ == '__main__':unittest.main()
