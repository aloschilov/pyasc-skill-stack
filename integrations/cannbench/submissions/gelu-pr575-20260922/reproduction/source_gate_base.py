"""Alias-aware campaign policy: high-level AscTile plus narrow reviewed Erfc.

Reuses the common source visitor. This is a policy gate, not a sandbox/proof.
Resource helper byte identity binds the audited read-only host implementation.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
SHARED=ROOT/'source_contract.py'
spec=importlib.util.spec_from_file_location('common_asctile_policy',SHARED)
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)


class Check(common._SourceCheck):
    def check_name(self,node,name):
        if name=='asctile.inline_vf': return
        if name in ('_device_resources.query_resources','_device_resources.launch_blocks'):
            if self.in_kernel:self.fail(node,'resource queries belong on host only')
            return
        if name in ('_launch_safety.required_ub','_launch_safety.validate_launch'):
            if self.in_kernel:self.fail(node,'launch safety belongs on host only')
            return
        if name in ('_runtime_context.launch_context','_runtime_context.record_launch'):
            if self.in_kernel:self.fail(node,'runtime context belongs on host only')
            return
        super().check_name(node,name)

    def visit_ImportFrom(self,node):
        if node.module=='_runtime_context':
            if node.level!=1:self.fail(node,'context helper must be package-relative')
            for item in node.names:
                if item.name not in ('launch_context','record_launch'):self.fail(node,'unsupported context helper import')
                self.bindings[item.asname or item.name]='_runtime_context.'+item.name
            return
        if node.module=='_launch_safety':
            if node.level!=1:self.fail(node,'safety helper must be package-relative')
            for item in node.names:
                if item.name not in ('required_ub','validate_launch'):self.fail(node,'unsupported safety helper import')
                self.bindings[item.asname or item.name]='_launch_safety.'+item.name
            return
        if node.module=='_device_resources':
            if node.level!=1:self.fail(node,'resource helper must be package-relative')
            for item in node.names:
                if item.name not in ('query_resources','launch_blocks'):self.fail(node,'unsupported resource helper import')
                self.bindings[item.asname or item.name]='_device_resources.'+item.name
            return
        super().visit_ImportFrom(node)

    def visit_Call(self,node):
        if self.resolve(node.func)=='asctile.inline_vf':
            if not self.in_kernel:self.fail(node,'Erfc inline must be inside JIT kernel')
            template=node.args[0] if node.args else None
            if not isinstance(template,ast.Constant) or not isinstance(template.value,str):
                self.fail(node,'inline template must be literal reviewed Erfc')
            elif re.sub(r'\s+','',template.value)!='{AscendC::Erfc<float,false>($0,$1,$1.GetSize());}':
                self.fail(node,'inline template is outside reviewed Erfc exception')
        super().visit_Call(node)


def check_source(path):
    path=Path(path)
    try:tree=ast.parse(path.read_text())
    except (OSError,SyntaxError) as exc:return [str(exc)]
    check=Check();check.visit(tree)
    if not check.has_jit or not check.has_asctile_import:check.problems.add('no public AscTile JIT')
    problems=sorted(check.problems)+common.check_runtime_helper(path.parent/'_pyasc_runtime.py')
    if any(isinstance(n,ast.ImportFrom) and n.module=='_device_resources' for n in ast.walk(tree)):
        helper=path.parent/'_device_resources.py'
        expected=ROOT.parents[1]/'submission/cann_bench/_device_resources.py'
        if hashlib.sha256(expected.read_bytes()).hexdigest()!='4770fffcd3398fd2c26779b7563ac9d8c09540895ab9257b5b1d7c5d6da2171a':
            problems.append('common resource helper changed; require new independent review and explicit gate revision')
        if not helper.is_file() or helper.read_bytes().rstrip()!=expected.read_bytes().rstrip():
            problems.append('resource helper differs from reviewed common implementation')
    if any(isinstance(n,ast.ImportFrom) and n.module=='_launch_safety' for n in ast.walk(tree)):
        helper=path.parent/'_launch_safety.py'
        expected=ROOT/'launch_safety.py'
        if hashlib.sha256(expected.read_bytes()).hexdigest()!='2de02d2735268f54846f2cd28493f948be3d3c25d6489242acedeae5f89510e0':
            problems.append('launch safety changed; require gate revision and review')
        if not helper.is_file() or helper.read_bytes().rstrip()!=expected.read_bytes().rstrip():
            problems.append('launch safety helper differs from reviewed source')
    if any(isinstance(n,ast.ImportFrom) and n.module=='_runtime_context' for n in ast.walk(tree)):
        helper=path.parent/'_runtime_context.py';expected=ROOT/'runtime_context.py'
        if hashlib.sha256(expected.read_bytes()).hexdigest()!='504fae7f3c561402cba378ddaac3ec936fef3cb404d79fdf5bf928008b2b01ec':
            problems.append('runtime context changed; require source gate revision and review')
        if not helper.is_file() or helper.read_bytes().rstrip()!=expected.read_bytes().rstrip():
            problems.append('runtime context differs from source-bound host implementation')
    return problems

if __name__=='__main__':
    import json,sys
    results={p:check_source(p) for p in sys.argv[1:]}
    print(json.dumps(results,indent=2));sys.exit(any(results.values()))
