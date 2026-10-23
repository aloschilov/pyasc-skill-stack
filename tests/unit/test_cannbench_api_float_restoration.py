import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'integrations/cannbench/workers'))
from reconcile_case_source import reconcile_api_float_types, load_verified_overlay
sha = lambda b: hashlib.sha256(b).hexdigest()

class ApiFloatRestorationTests(unittest.TestCase):
    def setUp(self):
        self.case = {'case_id': 1, 'attrs': {'epsilon': 1e-5}, 'input_shape': [[3, 5]], 'value_range': [-1, 1]}
        self.api = json.dumps({'cases_yaml': [self.case]}).encode()
        self.captured = json.dumps({'cases': [self.case]}).encode()

    def restore(self, captured=None, api=None):
        captured = self.captured if captured is None else captured
        api = self.api if api is None else api
        return reconcile_api_float_types(captured, api, 1, expected_captured_sha256=sha(captured), expected_api_sha256=sha(api))

    def test_only_yaml_misparsed_float_type_is_restored(self):
        self.assertIsInstance(yaml.safe_load(self.captured)['cases'][0]['attrs']['epsilon'], str)
        restored, evidence = self.restore()
        self.assertEqual(restored, self.case)
        self.assertIsInstance(restored['attrs']['epsilon'], float)
        self.assertEqual(evidence['changes'][0]['path'], ['attrs','epsilon'])
        self.assertEqual(yaml.safe_load(yaml.safe_dump({'cases':[restored]}))['cases'][0], self.case)

    def test_other_drift_and_coercions_rejected(self):
        for field, value in [('input_shape', [[3,4]]), ('value_range', [-2,2]), ('case_id', True)]:
            c=copy.deepcopy(self.case);c[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.restore(captured=json.dumps({'cases':[c]}).encode())
        for epsilon in ('0.00001', '1.00e-05', None, True):
            c=copy.deepcopy(self.case);c['attrs']['epsilon']=epsilon
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                self.restore(captured=json.dumps({'cases':[c]}).encode())

    def test_raw_api_string_is_not_reinterpreted(self):
        c=copy.deepcopy(self.case);c['attrs']['epsilon']='1e-05'
        restored,evidence=self.restore(api=json.dumps({'cases_yaml':[c]}).encode())
        self.assertIsInstance(restored['attrs']['epsilon'],str)
        self.assertEqual(evidence['changes'],[])

    def test_archive_loader_rechecks_raw_api_and_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=root/'catalog/op';task.mkdir(parents=True);bundle=root/'overlay';(bundle/'op').mkdir(parents=True)
            for parent in (task,bundle/'op'):
                (parent/'task.raw.json').write_bytes(self.api)
            (task/'cases.yaml').write_bytes(self.captured);(bundle/'op/captured.cases.yaml').write_bytes(self.captured)
            restored,evidence=self.restore();out=yaml.safe_dump({'cases':[restored]}).encode();(bundle/'op/case1.yaml').write_bytes(out)
            record=dict(evidence,operator='op',overlay='op/case1.yaml',overlay_sha256=sha(out))
            manifest=json.dumps({'records':[record]}).encode();(bundle/'MANIFEST.json').write_bytes(manifest)
            self.assertEqual(load_verified_overlay(bundle,sha(manifest),task,1)[0],self.case)
            (task/'task.raw.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'raw API'):
                load_verified_overlay(bundle,sha(manifest),task,1)

    def test_composed_repair_rejects_unrelated_upstream_changes(self):
        from reconcile_case_source import reconcile_api_then_nonfinite
        case=copy.deepcopy(self.case);case['value_range']=[None,None]
        api=json.dumps({'cases_yaml':[case]}).encode();captured=json.dumps({'cases':[case]}).encode()
        upstream=copy.deepcopy(case);upstream['value_range']=[float('-inf'),float('inf')]
        def run(source):
            return reconcile_api_then_nonfinite(captured,api,source,1,
                expected_captured_sha256=sha(captured),expected_api_sha256=sha(api),
                expected_source_sha256=sha(source),source_commit='a'*40)
        restored,evidence=run(yaml.safe_dump({'cases':[upstream]}).encode())
        self.assertEqual(restored['attrs']['epsilon'],1e-5)
        self.assertEqual(restored['value_range'],[float('-inf'),float('inf')])
        self.assertEqual(evidence['captured_sha256'],sha(captured))
        self.assertEqual(evidence['api_restoration']['source_sha256'],sha(api))
        upstream['input_shape']=[[3,4]]
        with self.assertRaisesRegex(ValueError,'value mismatch'):
            run(yaml.safe_dump({'cases':[upstream]}).encode())

if __name__ == '__main__':unittest.main()
