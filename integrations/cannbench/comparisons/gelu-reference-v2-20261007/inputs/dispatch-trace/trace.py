"""Reproducible source dispatch inventory; NOT a hardware trace or benchmark."""
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parent
INPUTS = OUT.parent
WORK = Path('/home/aloschilov/workspace')
NPU = WORK / 'torch-npu-gelu-trace-20261007'
PLUGIN = WORK / 'op-plugin-gelu-trace-20261007'
OPS = WORK / 'ops-nn'
BASE = OPS / 'third_party/opbase'
SDK = Path('/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc')
OP = OPS / 'activation/gelu_v2'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pin(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()

def main():
    sources = {
        'golden': INPUTS / 'golden.py',
        'cases': INPUTS / 'cases.yaml',
        'soc': NPU / 'torch_npu/csrc/core/npu/NpuVariables.cpp',
        'registration': PLUGIN / 'op_plugin/config/op_plugin_functions.yaml',
        'dispatch': PLUGIN / 'op_plugin/ops/opapi/GeluOutKernelNpuOpApi.cpp',
        'mode': PLUGIN / 'op_plugin/utils/KernelNpuNewParams.cpp',
        'api': OP / 'op_api/aclnn_gelu_v2.cpp',
        'launcher': OP / 'op_api/gelu_v2.cpp',
        'tiling': OP / 'op_host/arch35/gelu_v2_tiling_arch35.cpp',
        'template_args': OP / 'op_kernel/arch35/gelu_v2_struct.h',
        'kernel': OP / 'op_kernel/gelu_v2_apt.cpp',
        'dag': OP / 'op_kernel/arch35/gelu_v2_dag.h',
        'base_tiling': BASE / 'src/op_common/atvoss/elewise/elewise_tiling.cpp',
        'base_tiling_header': BASE / 'pkg_inc/op_common/atvoss/elewise/elewise_tiling.h',
        'scheduler': BASE / 'pkg_inc/op_common/atvoss/elewise/elewise_sch_16b.h',
        'vec': BASE / 'pkg_inc/op_common/atvoss/util/vec.h',
        'erf_config': SDK / 'include/adv_api/math/erf_utils.h',
        'erf': SDK / 'impl/adv_api/detail/math/erf/erf_c310_impl.h',
    }
    frozen = json.loads((OUT / 'trace.json').read_text())
    for name, source in sources.items():
        assert sha(source) == frozen['sources'][name]['sha256'], ('source changed', name)
    for key, checkout in (('torch_npu_pin', NPU), ('op_plugin_pin', PLUGIN),
                          ('ops_nn_pin', OPS), ('opbase_pin', BASE)):
        assert pin(checkout) == frozen[key], ('checkout changed', key)
    texts = {k: p.read_text() for k, p in sources.items()}
    # Fail visibly if sources no longer match the manually reviewed path.
    checks = {
        'golden': ['torch.nn.functional.gelu(x, approximate=approximate)'],
        'soc': ['inputVersion.compare(0, ascend950.size(), ascend950) == 0',
                'return GetSocVersion() >= SocVersion::Ascend950;'],
        'dispatch': ['!c10_npu::IsAclnnOnly() && !gelu_sc',
                     'EXEC_NPU_CMD(aclnnGeluV2, self, approximate_mode, out)'],
        'tiling': ['GET_TPL_TILING_KEY(1, approximate, dType)', '122880'],
        'kernel': ['ElementwiseSch16B', 'GeluV2Erf16BDag', 'GeluV2Erf32BDag', 'GeluV2TanhDag'],
        'dag': ['MicroAPI::RegTensor', '__VEC_SCOPE__', 'AscendC::Erf(dst, src, count)'],
        'vec': ['ErfAlgo::SUBSECTION_POLYNOMIAL_APPROXIMATION'],
        'erf_config': ['defaultErfConfig = { ErfAlgo::PADE_APPROXIMATION }'],
        'erf': ['__simd_vf__ inline void ErfCoreImpl', 'ErfPadeCompute', 'ErfSubsectionCompute'],
        'base_tiling': ['dim0 = dim0 * elewiseTilingParams.shape.GetDim(i)',
                        'ascendcPlatform.GetCoreNumAiv()'],
    }
    for name, snippets in checks.items():
        for snippet in snippets:
            assert snippet in texts[name], (name, snippet)
    for name in ('kernel', 'dag', 'scheduler', 'erf'):
        assert not re.search(r'\b(?:SIMT|simt|__simt_vf__)\b', texts[name]), name
    submodule = subprocess.check_output(['git', '-C', str(NPU), 'ls-tree', 'HEAD',
                                         'third_party/op-plugin'], text=True).split()[2]
    assert submodule == pin(PLUGIN), 'op-plugin must match torch-npu gitlink'
    cases = yaml.safe_load(texts['cases'])['cases']
    assert sorted(c['case_id'] for c in cases) == list(range(1, 21))
    rows = []
    for c in cases:
        dtype, mode = c['dtype'][0], c['attrs']['approximate']
        cpp = {'float16': 'half', 'bfloat16': 'bfloat16_t', 'float32': 'float'}[dtype]
        if mode == 'tanh':
            dag = f'GeluV2TanhDag<{cpp}>'
            algorithm = 'FP32 vector exp/div sigmoid-equivalent tanh approximation'
        elif dtype == 'float32':
            dag = 'GeluV2Erf32BDag<float>'
            algorithm = 'Vec::Erf<float>: SUBSECTION_POLYNOMIAL_APPROXIMATION'
        else:
            dag = f'GeluV2Erf16BDag<{cpp}>'
            algorithm = 'FP32 promotion; ErfFast<float>: default PADE_APPROXIMATION'
        assert dag in texts['kernel'] and dag in texts['tiling']
        shape = c['input_shape'][0]
        rows.append(dict(case_id=c['case_id'], shape=json.dumps(shape), elements=math.prod(shape),
            dtype=dtype, approximate=mode, aclnn='aclnnGeluV2',
            aclnn_mode=0 if mode == 'none' else 1,
            kernel='gelu_v2', scheduler='ElementwiseSch16B', schMode=1,
            template_approximate=1 if mode == 'none' else 2,
            template_dtype={'float16': 1, 'bfloat16': 2, 'float32': 3}[dtype],
            dag=dag, compute='SIMD', algorithm=algorithm,
            shape_effect='flattened element count, block partition, UB loops, masked tails',
            kernel_source=str(sources['kernel']), dag_source=str(sources['dag']),
            evidence='source-derived on inspected Ascend950 eager path; runner binary not captured'))
    with (OUT / 'cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(scope='20 historical CANNBench GeLU cases; source tracing only',
        torch_npu_pin=pin(NPU), op_plugin_pin=pin(PLUGIN), ops_nn_pin=pin(OPS), opbase_pin=pin(BASE),
        source_selection='Torch-NPU v2.10.0 latest public commit before 2026-09-16 17:39:53 UTC; not a recovered runner wheel commit',
        installed_erf_sdk='CANN9.0.0; job reported CANN9.1.0',
        sources={k: dict(path=str(p), sha256=sha(p)) for k, p in sources.items()},
        case_count=len(rows), simd_count=len(rows), hardware_execution=False,
        numeric_runner_tiling_recovered=False, cases=rows)
    (OUT / 'trace.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('case_count','simd_count','torch_npu_pin','op_plugin_pin')}, indent=2))

if __name__ == '__main__':
    main()
