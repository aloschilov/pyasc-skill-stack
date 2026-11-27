from pathlib import Path
import sys
RECIPE = Path(__file__).resolve().parents[3]/"examples/tuning/swiglu"
sys.path.insert(0, str(RECIPE))
import json
import tempfile
import unittest
from unittest.mock import patch
from consumer_interface import commands
from tool_delivery import audit
from non_tool_budget import gate_type

class DeliveryTests(unittest.TestCase):
    def body(self,arguments,name='bash'):
        return {'choices':[{'index':0,'message':{'tool_calls':[{'id':'call_1','type':'function','function':{'name':name,'arguments':json.dumps(arguments)}}]}}]}
    def arguments(self):return {'command':commands('/recipe')['compare'],'timeout':600000}
    def check(self,body):return audit(json.dumps(body).encode(),False,{'bash','read','write','edit','skill'},'/recipe','/consumer')
    def test_valid_nonstream(self):self.assertEqual(self.check(self.body(self.arguments()))['tool_calls'][0]['action'],'compare')
    def test_default_timeout_blocked_before_delivery(self):
        args=self.arguments();del args['timeout']
        with self.assertRaises(ValueError):self.check(self.body(args))
    def test_arbitrary_command_blocked_before_delivery(self):
        args=self.arguments();args['command']+='; id'
        with self.assertRaises(ValueError):self.check(self.body(args))
    def test_short_timeout_blocked(self):
        args=self.arguments();args['timeout']=120000
        with self.assertRaises(ValueError):self.check(self.body(args))
    def test_mixed_write_and_measurement_blocked(self):
        body=self.body(self.arguments());body['choices'][0]['message']['tool_calls']+=self.body({'filePath':'candidate.py','content':'x'},'write')['choices'][0]['message']['tool_calls']
        with self.assertRaises(ValueError):self.check(body)
    def test_fragmented_stream(self):
        args=json.dumps(self.arguments());parts=[{'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'id':'call_1','type':'function','function':{'name':'bash','arguments':args[:20]}}]}}]},{'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'function':{'arguments':args[20:]}}]}}]}]
        raw=('\n\n'.join('data: '+json.dumps(p)for p in parts)+'\n\ndata: [DONE]\n').encode()
        self.assertEqual(audit(raw,True,{'bash'},'/recipe','/consumer')['tool_calls'][0]['action'],'compare')
    def test_incomplete_stream(self):
        with self.assertRaises(ValueError):audit(b'data: {}\n',True,{'bash'},'/recipe','/consumer')
    def test_rejected_tool_never_delivered_raw_retained(self):
        args=self.arguments();del args['timeout'];raw=json.dumps(self.body(args)).encode();delivered=[];retained=[]
        class Base:
            def __init__(self):self.limits={"output_tokens_per_response":32768,"max_output_tokens":200000};self.records=[];self.deadline=2800;self.error=None;self.closed=False;self.records=[]
            def checkpoint(self):pass
            def dispatch(self,body,deliver=None):
                # Existing gate's order: capture raw, account usage, then deliver.
                retained.append(raw);self.records.append({'status':'complete','usage':{'completion_tokens':10}})
                return deliver(raw,'application/json')
        with tempfile.TemporaryDirectory()as d,patch('non_tool_budget.time.monotonic',return_value=200):
            gate=gate_type(Base,d,lambda r:None,lambda raw,body:audit(raw,False,{'bash'},'/recipe','/consumer'))()
            with self.assertRaises(ValueError):gate.dispatch({},lambda raw,kind:delivered.append(raw))
            self.assertEqual(retained,[raw]);self.assertEqual(delivered,[])
            self.assertEqual(gate.records[0]['usage']['completion_tokens'],10)
            self.assertFalse(gate.records[0]['tool_preexecution']['model_tool_executed'])


if __name__=='__main__':unittest.main()
