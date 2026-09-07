"""CPU-only convergence/source provenance, not compiled-kernel qualification."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import torch

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
a = torch.linspace(2, 14, 200001, dtype=torch.float64)
gold = 0.5 * torch.special.erfc(a / math.sqrt(2))
rows = []
for depth in [4, 6, 8, 12, 16]:
    remainder = torch.zeros_like(a)
    for n in range(depth, 0, -1):
        remainder = n / (a + remainder)
    q = torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi) / (a + remainder)
    rows.append(dict(depth=depth, maximum_relative_error=float((q / gold - 1).abs().max())))
sources = [Path('/home/aloschilov/workspace/ops-nn/activation/gelu_v2/op_kernel/arch35/gelu_v2_dag.h'),
           Path('/usr/local/Ascend/cann-9.0.0/aarch64-linux/pkg_inc/op_common/atvoss/util/vec.h'),
           Path('/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/include/adv_api/math/erf_utils.h'),
           ROOT.parent/'gelu-asctile-jit-20260907/runtime/source/python/asctile/language/unary_ops.py']
result = dict(scope='CPU formula convergence only', reference='https://dlmf.nist.gov/7.9.E1',
              range=[2,14], samples=a.numel(), convergence=rows,
              ops_nn_revision=subprocess.check_output(['git','-C','/home/aloschilov/workspace/ops-nn','rev-parse','HEAD'],text=True).strip(),
              sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(ROOT/'evidence/tail-math.json').write_text(json.dumps(result, indent=2)+'\n')
fast_a = torch.linspace(3, 14, 100001, dtype=torch.float64)
fast_r = torch.zeros_like(fast_a)
for n in range(6, 0, -1):
    fast_r = n / (fast_a + fast_r)
fast_q = torch.exp(-0.5 * fast_a * fast_a) / math.sqrt(2 * math.pi) / (fast_a + fast_r)
fast_gold = 0.5 * torch.special.erfc(fast_a / math.sqrt(2))
fast_result = dict(scope='CPU double formula only; not compiled execution', range=[3,14],
                   depth=6, samples=fast_a.numel(),
                   maximum_relative_error=float((fast_q/fast_gold-1).abs().max()),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'evidence/tail-math-fast.json').write_text(json.dumps(fast_result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print(json.dumps(fast_result,indent=2))
