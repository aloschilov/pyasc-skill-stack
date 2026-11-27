from pathlib import Path
import sys
RECIPE = Path(__file__).resolve().parents[3]/"examples/tuning/swiglu"
sys.path.insert(0, str(RECIPE))
"""No external requests or runtime imports: budget and native binding defects."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from non_tool_budget import remaining,gate_type
from consumer_interface import native_source_binding
from selection import select

class BudgetTests(unittest.TestCase):
    def receipt(self,start,end,status='passed'):
        return ('baseline',{'action':'baseline','status':status,'tool_monotonic_start':start,'tool_monotonic_end':end},'a'*64)
    def test_no_tool_does_not_expand_generation(self):self.assertEqual(remaining(100,2800,1000,[])[0],0)
    def test_only_real_completed_time_credited(self):self.assertEqual(remaining(100,2800,1100,[self.receipt(300,500)])[0],100)
    def test_unknown_interval_rejected(self):
        with self.assertRaises(ValueError):remaining(100,2800,1100,[self.receipt(300,500,'unknown')])
    def test_overlong_interval_rejected(self):
        with self.assertRaises(ValueError):remaining(100,2800,1100,[self.receipt(300,600)])
    def test_future_interval_rejected(self):
        with self.assertRaises(ValueError):remaining(100,2800,400,[self.receipt(300,500)])
    def test_overlapping_intervals_rejected(self):
        with self.assertRaises(ValueError):remaining(100,2800,1100,[self.receipt(300,500),self.receipt(400,600)])
    def test_earlier_http_deadline(self):
        class Fake:
            def __init__(self):self.limits={"output_tokens_per_response":32768,"max_output_tokens":200000};self.records=[];self.deadline=2800;self.error=None;self.closed=False;self.dispatched=0
            def checkpoint(self):pass
            def dispatch(self,body,deliver=None):self.dispatched+=1;return self.deadline
        with tempfile.TemporaryDirectory()as d,patch('non_tool_budget.time.monotonic',return_value=200):
            gate=gate_type(Fake,d,lambda r:None)();self.assertEqual(gate.dispatch({}),1000);self.assertEqual(gate.dispatched,1)
    def test_exhausted_http_not_dispatched(self):
        class Fake:
            def __init__(self):self.limits={"output_tokens_per_response":32768,"max_output_tokens":200000};self.records=[];self.deadline=2800;self.error=None;self.closed=False;self.dispatched=0
            def checkpoint(self):pass
            def dispatch(self,body,deliver=None):self.dispatched+=1
        with tempfile.TemporaryDirectory()as d,patch('non_tool_budget.time.monotonic',return_value=1001):
            gate=gate_type(Fake,d,lambda r:None)()
            with self.assertRaises(ValueError):gate.dispatch({})
            self.assertEqual(gate.dispatched,0);self.assertIsNotNone(gate.error)

class BindingTests(unittest.TestCase):
    def event(self,tool,data,status='completed'):
        return {'part':{'tool':tool,'state':{'status':status,'input':data}}}
    def test_failed_variant_still_seals_original(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d);project=p/'project';project.mkdir();work=p/'work';snapshot=work/'variant/candidate';snapshot.mkdir(parents=True)
            (snapshot/'candidate.py').write_text('original');(project/'candidate.py').write_text('repair');(snapshot.parent/'hypothesis-before-evaluation.json').write_text('{}');(project/'decision.json').write_text('{}')
            events=[self.event('write',{'filePath':'decision.json','content':'{}'}),self.event('write',{'filePath':'candidate.py','content':'original'}),self.event('bash',{'command':'python3 -B /recipe/tool.py variant'},'error'),self.event('write',{'filePath':'candidate.py','content':'repair'})]
            with self.assertRaisesRegex(ValueError,'Post-evaluation'):native_source_binding(events,project,work)
    def test_unknown_write_not_promoted(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d);(p/'candidate.py').write_text('unbound')
            events=[self.event('edit',{'filePath':'candidate.py','oldString':'x','newString':'unbound'})]
            with self.assertRaises(ValueError):native_source_binding(events,p,p/'work')
    def test_exact_fixture_inventory(self):
        r={'status':'passed','analysis':[{'fixture':'wrong','order_A_over_B':{'ABBA':2,'BAAB':2}}]*3}
        with self.assertRaises(ValueError):select(r,['a','b','c'])
    def test_optimized_entry_stops_before_config(self):
        p=RECIPE/'tool.py'
        r=subprocess.run([sys.executable,'-O','-B',str(p),'baseline'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('Assertions must remain enabled',r.stderr);self.assertNotIn('session-config',r.stderr)


class ConcurrentGateTests(unittest.TestCase):
    def test_dispatch_finish_share_one_lock(self):
        import threading
        entered=threading.Event();release=threading.Event();finished=threading.Event();audit=[];errors=[]
        class Fake:
            def __init__(self):self.limits={"output_tokens_per_response":32768,"max_output_tokens":200000};self.records=[];self.deadline=2800;self.error=None;self.closed=False;self.dispatched=0
            def checkpoint(self):pass
            def dispatch(self,body,deliver=None):
                with self.lock:
                    if self.closed:raise ValueError('closed')
                    self.dispatched+=1;entered.set();release.wait(2);return self.deadline
            def finish(self):
                with self.lock:self.closed=True;finished.set()
        with tempfile.TemporaryDirectory()as d,patch('non_tool_budget.time.monotonic',return_value=200):
            gate=gate_type(Fake,d,lambda r:audit.append(r))()
            first=threading.Thread(target=lambda:gate.dispatch({}));first.start();self.assertTrue(entered.wait(1));deadline=gate.deadline
            final=threading.Thread(target=gate.finish);final.start();self.assertFalse(finished.wait(.03));self.assertEqual(gate.deadline,deadline)
            release.set();first.join(1);final.join(1);self.assertTrue(finished.is_set())
            count=len(audit)
            with self.assertRaises(ValueError):gate.dispatch({})
            self.assertEqual(gate.dispatched,1);self.assertEqual(len(audit),count);self.assertEqual(gate.deadline,deadline)
    def test_concurrent_dispatch_cannot_extend_active_deadline(self):
        import threading
        entered=threading.Event();release=threading.Event();audit=[]
        class Fake:
            def __init__(self):self.limits={"output_tokens_per_response":32768,"max_output_tokens":200000};self.records=[];self.deadline=2800;self.error=None;self.closed=False;self.dispatched=0
            def checkpoint(self):pass
            def dispatch(self,body,deliver=None):
                with self.lock:
                    self.dispatched+=1
                    if self.dispatched==1:entered.set();release.wait(2)
                    return self.deadline
        with tempfile.TemporaryDirectory()as d,patch('non_tool_budget.time.monotonic',return_value=200):
            gate=gate_type(Fake,d,lambda r:audit.append(r))()
            first=threading.Thread(target=lambda:gate.dispatch({}));first.start();self.assertTrue(entered.wait(1));deadline=gate.deadline
            second=threading.Thread(target=lambda:gate.dispatch({}));second.start()
            second.join(.03);self.assertTrue(second.is_alive());self.assertEqual(len(audit),1);self.assertEqual(gate.deadline,deadline)
            release.set();first.join(1);second.join(1);self.assertEqual(len(audit),2)

class AggregateBudgetTests(unittest.TestCase):
    def test_native_reservation_exact_boundary(self):
        class Base:
            def __init__(self):
                self.deadline=2800;self.error=None;self.closed=False;self.calls=0
                self.records=[{'usage':{'completion_tokens':167232}}]
                self.limits={'output_tokens_per_response':32768,'max_output_tokens':200000}
            def checkpoint(self):pass
            def dispatch(self,body,deliver=None):
                self.calls+=1;self.records.append({'usage':{'completion_tokens':1}})
        with tempfile.TemporaryDirectory()as d,patch('non_tool_budget.time.monotonic',return_value=200):
            gate=gate_type(Base,d,lambda _:None)();gate.dispatch({})
            self.assertEqual(gate.calls,1)
            with self.assertRaises(ValueError):gate.dispatch({})
            self.assertEqual(gate.calls,1);self.assertEqual(gate.limits['output_tokens_per_response'],32768)
            self.assertEqual(gate.error,'Aggregate output budget exhausted')
    def test_concurrent_requests_cannot_spend_same_reservation(self):
        import threading
        entered=threading.Event();release=threading.Event();errors=[]
        class Base:
            def __init__(self):
                self.deadline=2800;self.error=None;self.closed=False;self.calls=0
                self.records=[{'usage':{'completion_tokens':167232}}]
                self.limits={'output_tokens_per_response':32768,'max_output_tokens':200000}
            def checkpoint(self):pass
            def dispatch(self,body,deliver=None):
                self.calls+=1;entered.set();release.wait(2)
                self.records.append({'usage':{'completion_tokens':1}})
            def finish(self):
                with self.lock:self.closed=True
        with tempfile.TemporaryDirectory()as d,patch('non_tool_budget.time.monotonic',return_value=200):
            gate=gate_type(Base,d,lambda _:None)()
            def request():
                try:gate.dispatch({})
                except ValueError:errors.append('blocked')
            a=threading.Thread(target=request);a.start();self.assertTrue(entered.wait(1))
            b=threading.Thread(target=request);b.start();release.set();a.join(1);b.join(1)
            gate.finish();self.assertEqual(gate.calls,1);self.assertEqual(errors,['blocked'])
            with self.assertRaises(ValueError):gate.dispatch({})
            self.assertEqual(gate.calls,1)

if __name__=='__main__':unittest.main()
