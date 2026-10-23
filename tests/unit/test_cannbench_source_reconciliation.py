import hashlib
import math
import sys
import unittest
import yaml
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'integrations/cannbench/workers'))
from reconcile_case_source import reconcile_case, load_verified_overlay

class ReconciliationTests(unittest.TestCase):
    def run_case(self,a,b,**kw):
        ca=yaml.safe_dump({'cases':[a]}).encode();so=yaml.safe_dump({'cases':[b]}).encode()
        args=dict(captured_bytes=ca,source_bytes=so,
                  expected_captured_sha256=hashlib.sha256(ca).hexdigest(),
                  expected_source_sha256=hashlib.sha256(so).hexdigest(),source_commit='a'*40)
        args.update(kw)
        return reconcile_case(a,b,**args)
    def test_restore_nonfinite_and_preserve_optional_none_and_notes(self):
        a={'case_id':12,'attrs':{'p':None,'optional':None},'value_range':[None,None],'note':'capture'}
        b={'case_id':12,'attrs':{'p':float('inf'),'optional':None},'value_range':[float('-inf'),float('nan')],'note':'upstream','t_hw_us':1.0}
        r,e=self.run_case(a,b)
        self.assertEqual(r['attrs']['p'],float('inf'));self.assertIsNone(r['attrs']['optional'])
        self.assertTrue(math.isnan(r['value_range'][1]));self.assertEqual(r['note'],'capture')
        self.assertEqual(len(e['changes']),3);self.assertIsNone(a['attrs']['p'])
    def test_refuse_unrelated_changes_missing_fields_and_hash_drift(self):
        for a,b in [({'shape':[2]},{'shape':[3]}),({'attrs':{}},{'attrs':{'p':None}}),
                    ({'value_range':[None,1]},{'value_range':[0,1]}),
                    ({'value_range':[0,1]},{'value_range':[float('-inf'),1]})]:
            with self.assertRaises(ValueError): self.run_case(a,b)
        with self.assertRaises(ValueError): self.run_case({}, {}, expected_source_sha256='bad')
        with self.assertRaises(ValueError): self.run_case({}, {}, source_commit='abbrev')

    def test_reject_forged_numeric_types_bound_to_same_yaml_value(self):
        raw = yaml.safe_dump({'cases': [{'case_id': 1, 'attrs': {'flag': True}}]}).encode()
        digest = hashlib.sha256(raw).hexdigest()
        for value in (1, 1.0):
            forged = {'case_id': 1, 'attrs': {'flag': value}}
            with self.assertRaisesRegex(ValueError, 'not bound'):
                reconcile_case(forged, forged, captured_bytes=raw, source_bytes=raw,
                               expected_captured_sha256=digest, expected_source_sha256=digest,
                               source_commit='a'*40)

    def test_prose_field_names_nested_in_inputs_are_not_ignored(self):
        with self.assertRaisesRegex(ValueError, 'value mismatch'):
            self.run_case({'case_id': 1, 'attrs': {'note': 1}},
                          {'case_id': 1, 'attrs': {'note': 2}})

    def test_real_staged_overlay_recomputes_and_rejects_drift(self):
        import tempfile
        import json
        import shutil
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'overlay'; task = Path(td)/'catalog'/'exp'
            (root/'exp').mkdir(parents=True); task.mkdir(parents=True)
            a = {'case_id': 12, 'value_range': [None, None]}
            b = {'case_id': 12, 'value_range': [float('-inf'), float('inf')]}
            ca = yaml.safe_dump({'cases':[a]}).encode()
            so = yaml.safe_dump({'cases':[b]}).encode()
            restored, evidence = self.run_case(a,b)
            (task/'cases.yaml').write_bytes(ca)
            (root/'exp/captured.cases.yaml').write_bytes(ca)
            (root/'exp/upstream.cases.yaml').write_bytes(so)
            overlay = yaml.safe_dump({'cases':[restored]}).encode()
            (root/'exp/case_12.yaml').write_bytes(overlay)
            record = dict(evidence, operator='exp', overlay='exp/case_12.yaml',
                          overlay_sha256=hashlib.sha256(overlay).hexdigest())
            raw = json.dumps({'records':[record]}).encode()
            (root/'MANIFEST.json').write_bytes(raw)
            h = hashlib.sha256(raw).hexdigest()
            result, provenance = load_verified_overlay(root,h,task,12)
            self.assertEqual(result['value_range'][0],float('-inf'))
            self.assertEqual(provenance['overlay_manifest_sha256'],h)
            self.assertIsNone(load_verified_overlay(root,h,task,1))
            with self.assertRaisesRegex(ValueError,'manifest hash'):
                load_verified_overlay(root,'0'*64,task,12)
            (task/'cases.yaml').write_text('cases: []')
            with self.assertRaisesRegex(ValueError,'current catalog'):
                load_verified_overlay(root,h,task,12)
            (task/'cases.yaml').write_bytes(ca)
            (root/'exp/case_12.yaml').write_text('cases: []')
            with self.assertRaisesRegex(ValueError,'case hash'):
                load_verified_overlay(root,h,task,12)

if __name__=='__main__': unittest.main()
