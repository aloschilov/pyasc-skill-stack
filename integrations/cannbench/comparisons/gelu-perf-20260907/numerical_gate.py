"""CPU screening of exact-mode approximations; never used by submitted kernels."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy.special import erfcx
import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('precision', '/home/aloschilov/workspace/cann-bench/src/kernel_eval/utils/precision.py')
precision = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = precision
spec.loader.exec_module(precision)
torch.set_num_threads(2)

def evaluate(x, method, coefficients=None):
    z = x.float().abs() * 0.7071067811865476
    u = 1 / (1 + .5*z)
    if method == 'nr':
        coefficients = [-1.26551223, 1.00002368, .37409196, .09678418, -.18628806,
                        .27886807, -1.13520398, 1.48851587, -.82215223, .17087277]
    p = torch.full_like(z, coefficients[-1])
    for c in reversed(coefficients[:-1]):
        p = p*u+c
    small = x.float().abs() * .5 * u * torch.exp(p-z*z)
    return (x.float()*.5 + x.float().abs()*.5-small).to(x.dtype)

def main():
    # Fit a mathematical erfc approximation on the full positive half-line,
    # not benchmark data. erfcx(z)/u has a finite limit at u=0.
    u = np.linspace(.000001,1,20000)
    z = 2*(1/u-1)
    target = np.log(erfcx(z)/u)
    methods = [('nr',None)]
    for degree in (5,6,7,8):
        poly = Chebyshev.fit(u,target,degree,domain=[0,1]).convert(kind=Polynomial)
        methods.append(('fit'+str(degree),poly.coef.tolist()))
    results=[]
    for method, coefs in methods:
        tests=[]
        for dtype in ('float16','bfloat16','float32'):
            for lo,hi in [(-15,15),(-.2,.2),(-2,2),(-3,3),(-50,100)]:
                x=torch.linspace(lo,hi,500003).to(getattr(torch,dtype))
                y=evaluate(x,method,coefs)
                golden=torch.nn.functional.gelu(x.double())
                result=precision.compare_tensors(y,golden,dtype=dtype,cpu_output=torch.nn.functional.gelu(x))
                tests.append(dict(dtype=dtype,range=[lo,hi],passed=result.passed,
                                  mere=result.mere,mare=result.mare,error=result.error_msg))
        results.append(dict(method=method,coefficients=coefs,passed=all(t['passed'] for t in tests),tests=tests))
    (ROOT/'numerical-gate.json').write_text(json.dumps(results,indent=2)+'\n')
    for r in results:
        print(r['method'],r['passed'],max(t['mare'] or 0 for t in r['tests']))

if __name__=='__main__': main()
