"""Same alias-aware high-level/Erfc policy; explicit revised helper identities."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
OLD=ROOT
# Avoid module-name collisions with this gate; load prior gates under aliases.
spec=importlib.util.spec_from_file_location('prior_source_gate',OLD/'source_gate_base.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)

class Check(prior.Check):
    def visit_ImportFrom(self,node):
        if node.module=='_gelu_metadata.gate':
            if node.level!=1:self.fail(node,'metadata must be package-relative')
            for item in node.names:
                if item.name not in ('require_default_device','validate_capacity','record_capacity'):
                    self.fail(node,'unsupported host metadata')
                self.bindings[item.asname or item.name]='_gelu_metadata.gate.'+item.name
            return
        super().visit_ImportFrom(node)
    def check_name(self,node,name):
        if name in {'_gelu_metadata.gate.'+p for p in ('require_default_device','validate_capacity','record_capacity')}:
            if self.in_kernel:self.fail(node,'host metadata inside kernel')
            return
        super().check_name(node,name)

def check_source(path):
    path=Path(path)
    tree=ast.parse(path.read_text());check=Check();check.visit(tree)
    if not check.has_jit or not check.has_asctile_import:check.problems.add('Missing public AscTile JIT')
    problems=sorted(check.problems)+prior.common.check_runtime_helper(path.parent/'_pyasc_runtime.py')
    expected=json.loads((ROOT/'helper-hashes.json').read_text())
    for name,digest in expected.items():
        member=path.parent/name
        if not member.is_file() or hashlib.sha256(member.read_bytes()).hexdigest()!=digest:
            problems.append('Host helper identity failed: '+name)
    return problems
