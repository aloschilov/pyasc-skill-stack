"""Lower the complete geometry space and run positive and suppressed CaModel checks."""
import ast
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from asc import DataType
from asc._C import ir
from asc.codegen.specialization import Specialization
from asc.lib.utils import get_ascend_path
from asc.runtime import config
from asc.runtime.jit import MockTensor

from geometry import DTYPES, all_specs, full_row_probes, key, layout, slab_indices, validate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'workers'))
from local_compile_gate import prepare_specialization

PACKED = Path('context/skills/asctile-api-patterns/templates/packed-bf16-swiglu.py')
NAMES = {'rows':7,'slab':8,'flat':6}


def load_kernels(path, template_path):
    tree = ast.parse(path.read_bytes())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for n in tree.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value,str):
            continue
        if isinstance(n, ast.ImportFrom) and n.module == 'asc.experimental' and all(a.name=='asctile' for a in n.names):
            continue
        if n not in defs:
            raise ValueError('Compute module must contain only PyASC imports and kernel definitions')
    if {n.name:len(n.args.args) for n in defs} != NAMES or len(defs) != 3:
        raise ValueError('Generated kernel names or signatures differ from the device ABI')
    for fn in defs:
        u=fn.args.args[-1].arg
        if not any(k.arg=='unroll_factor' and isinstance(k.value,ast.Name) and k.value.id==u
                   for k in ast.walk(fn) if isinstance(k,ast.keyword)):
            raise ValueError(fn.name+' must use the dispatched constexpr unroll factor')
    packed_tree = ast.parse(template_path.read_bytes())
    packed, = [n for n in packed_tree.body if isinstance(n, ast.FunctionDef) and n.name=='_swiglu_pair_groups']
    # Preserve the authored bytes of each measured implementation. Only the template entry is imported separately.
    template = Path('/tmp/swiglu-public-template.py')
    template.write_text('from asc.experimental import asctile\n'+'\n'.join(template_path.read_text().splitlines()[
        packed.decorator_list[0].lineno-1:packed.end_lineno])+'\n')
    kernels = {}
    for filename, names in ((path,NAMES),(template,['_swiglu_pair_groups'])):
        module_spec = importlib.util.spec_from_file_location('kernels_'+filename.stem.replace('-','_'),filename)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        kernels.update({name:getattr(module,name) for name in names})
    return kernels


def lower(kernel, dtype, route, const, folder):
    data = MockTensor(DataType(dtype))
    args = [data,data]
    if route == 'slab':
        index = MockTensor(DataType('int32'))
        args += [index,index,64]
    else:
        args += [64,128]
    _, argtypes, constexprs, codegen, options = prepare_specialization(kernel, (*args,*const))
    if len(argtypes) != len(args) or len(constexprs) != len(const):
        raise ValueError('Unexpected runtime/constexpr ABI for '+route)
    module = kernel._run_codegen(Specialization(argtypes,constexprs),codegen)
    compiler = kernel.compiler(options)
    compiler.preprocess_module(module)
    compiler.run_passes(module)
    compiler.postprocess_module(module)
    memory = module.op.get_dict_of_int_attr(ir.attr.memory_consumed) or {}
    kernel.launcher.check_memory_overflow(memory)
    if any('Ffts' in str(arg) for arg in ir.get_kernel_arg_attrs(module)):
        raise ValueError('Unsupported implicit device parameter')
    source = compiler.run_translation(module)
    (folder/(key(dtype,route,const)+'.cpp')).write_text(source)
    # Capture the exact PyASC compiler and linker command for this specialization.
    commands = []
    sdk = str(Path(get_ascend_path()).resolve())
    def capture(argv, stage):
        if stage not in ('compile','link'):
            raise ValueError('Unknown compiler stage')
        # The service wheel is built on x86_64; the CI simulator may run on arm64.
        tikcpp = str((Path(sdk)/'compiler/tikcpp').resolve())
        args = [str(arg).replace(tikcpp,'@SDK@/compiler/tikcpp').replace(sdk,'@SDK@') for arg in argv]
        args[0] = '@SDK@/bin/'+Path(args[0]).name
        commands.append(args)
    compiler._run_cmd = capture
    compiler._gen_dst_kernel('/tmp',Path('@SRC@'),Path('@OBJ@'))
    if [Path(row[0]).name for row in commands] != ['bisheng','ld.lld']:
        raise ValueError('Expected a single vector compilation and link')
    return commands, {str(k):int(v) for k,v in memory.items()}


def check(kernels, dtype, shape, axis, seed, policy):
    print('CaModel', dtype, shape, axis, seed, flush=True)
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=getattr(torch,dtype))
    x.flatten()[:7] = torch.tensor([float('inf'),-float('inf'),float('nan'),-100.,0.,1.,100.])
    a,b = x.float().chunk(2,axis)
    expected = (torch.nn.functional.silu(a)*b).to(x.dtype)
    shape, rows, half, route, const, blocks = layout(shape,axis,dtype,policy,3)
    out = torch.full(shape, float('nan'),dtype=x.dtype)
    kwargs = [x,out]
    if route == 'slab':
        indices = [torch.tensor(index,dtype=torch.int32) for index in slab_indices(*const[:2])]
        kwargs.extend([*indices,rows])
    else:
        kwargs.extend([rows,(rows+const[0]-1)//const[0] if route=='_swiglu_pair_groups' else half])
    try:
        torch.testing.assert_close(out,expected,equal_nan=True)
    except AssertionError:
        pass
    else:
        raise ValueError('Suppressed launch passed the numerical oracle')
    kernels[route][blocks](*kwargs,*const)
    torch.testing.assert_close(out,expected,equal_nan=True)
    return {'shape':list(x.shape),'dim':axis,'dtype':dtype,'route':route,'seed':seed,
            'blocks':blocks,'constexprs':const,'positive':True,'suppressed_rejected':True}


def main(root, output):
    config.set_platform(config.Backend.Model,config.Platform.Ascend950PR_9599,check=False)
    candidate = root/'project/candidate.py'
    digest = sha256(candidate.read_bytes()).hexdigest()
    if digest != json.loads((root/'generation.json').read_bytes())['candidate_sha256']:
        raise ValueError('Generation checkpoint changed')
    raw_policy = (root/'project/policy.json').read_bytes()
    policy_hash = sha256(raw_policy).hexdigest()
    if policy_hash != json.loads((root/'generation.json').read_bytes())['policy_sha256']:
        raise ValueError('Generation policy changed')
    policy = validate(json.loads(raw_policy))
    template_path = root/PACKED
    kernels = load_kernels(candidate, template_path)
    checks = []
    for dtype in DTYPES:
        for shape,axis in [((131,14),1),((129,2),1),((3,128,8),1),((1,65538),1)]:
            for seed in (19,23):
                checks.append(check(kernels,dtype,shape,axis,seed,policy))
    for shape in full_row_probes(policy):
        for seed in (19,23):
            checks.append(check(kernels,'float32',shape,1,seed,policy))
    project = output/'package-source'
    folder = project/'csrc/ops/swi_glu/op_kernel'
    folder.mkdir(parents=True)
    flags, allocations = {}, {}
    for dtype,r,c in all_specs(policy):
        name = key(dtype,r,c)
        flags[name],allocations[name] = lower(kernels[r],dtype,r,c,folder)
    (project/'flags.json').write_text(json.dumps(flags,sort_keys=True))
    (output/'camodel.json').write_text(json.dumps({'backend':'Model','candidate_sha256':digest,
        'template_sha256':sha256(template_path.read_bytes()).hexdigest(), 'checks':checks,'policy_sha256':policy_hash,'allocations':allocations,
        'sources':{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}},indent=2))
    print('Qualified',len(checks),'numerical/control pairs and',len(flags),'vector specializations')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[-1]))
