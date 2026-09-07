"""Regression checks for the integration gate, not modifications to pyasc JIT."""
import json
from pathlib import Path
import sys
import asctile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]/'integrations/cannbench/workers'))
import local_compile_gate as gate
from candidate.gelu import gelu_kernel

args = (gate.tensor([2049],'float32'),gate.tensor([2049],'float32'),2049,1024,1,False,True)
default = gate.prepare_specialization(gelu_kernel,args)
override = gate.prepare_specialization(gelu_kernel,args,{'reuse_alloc':2,'vf_fusion':True,'static_alloc':False})
assert default[0]!=override[0]
assert default[-1].reuse_alloc==1 and override[-1].reuse_alloc==2
assert override[-1].vf_fusion is True and override[-1].static_alloc is False
keyword = gate.prepare_specialization(gelu_kernel,args[:-1],{'stable_tail':True})
assert keyword[0]==default[0]
try:
    gate.prepare_specialization(gelu_kernel,args,{'unsupported_flag':True})
except TypeError:
    pass
else:
    raise AssertionError('Unknown invocation option silently accepted')
launches=[]
gate.CaptureKernel(1,'gelu_kernel',gelu_kernel,launches,set())[8](*args,reuse_alloc=2,vf_fusion=True,static_alloc=False)
assert launches[0].specialization==override[0] and launches[0].kwargs['reuse_alloc']==2
assert gate.prepare_specialization(gelu_kernel,launches[0].args,launches[0].kwargs)[0]==override[0]
result={'status':'passed','checks':['override propagation','different specialization keys',
    'kernel keyword arguments','unknown option rejected','capture/replay retains flags'],
    'scope':'integration gate only; runtime cache behavior covered by separate Model repeats'}
(ROOT/'evidence/gate-options.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
