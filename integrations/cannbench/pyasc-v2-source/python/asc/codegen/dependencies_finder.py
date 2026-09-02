# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import types
import copy
import hashlib
import itertools
import ast
from typing import Any, Dict, Tuple

from ..language.core.constexpr import ConstExpr
from ..language.core.utils import jit_allowed
from .name_scope import NameScope
from .function import Function

PYASC_MODULE = "asc.language"


class DependenciesFinder(ast.NodeVisitor):
    """
    This AST visitor is used to find dependencies of a JITFunction. This can
    be used to invalidate a JITFunction's hash when its source code -- or
    that of its dependencies -- changes.

    This visitor also keeps track of the global variables touched by the
    JITFunction.  When we launch the kernel, we check that these have the same
    values as they did when we ran this visitor.  If not, we raise an error (or
    otherwise we could recompile).
    """

    def __init__(self, name_, globals_, nonlocals_, src_) -> None:
        super().__init__()
        self.name = name_
        self.hasher = hashlib.sha256(src_.encode("utf-8"))

        # This function's __globals__ dict.
        self.globals = globals_
        self.nonlocals = nonlocals_

        # Python builtins that can be accessed from pyasc kernels.
        self.supported_python_builtins = set(NameScope.default_builtins)

        self.supported_modules = {
            PYASC_MODULE,
            "typing",
        }

        # used_global_vals tells us which global variables are used by this
        # function and all those it transitively calls, plus the values of those
        # variables when each function was initially run.  (That is, if A calls
        # C, and B calls C, then the values for C in used_global_vals will be
        # from the first time C was run, either by A or B.)
        #
        # Each function may have a different __globals__ dict, so the global
        # variable `foo` may actually have a different value in the different
        # functions.  Thus this map is actually
        #  (var_name, id(__globals__)) -> (var_value, __globals__).
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

        self.visiting_arg_default_value = False

    @property
    def ret(self):
        return self.hasher.hexdigest()

    def update_hash(self, func):
        if not isinstance(func, Function):
            raise TypeError("func is not Function")
        # Merge our used_global_vals with those of the called function,
        # after checking that all overlapping values are consistent.
        for k in self.used_global_vals.keys() & func.used_global_vals.keys():
            var_name, _ = k
            v1, _ = self.used_global_vals[k]
            v2, _ = func.used_global_vals[k]
            if v1 != v2:
                raise RuntimeError(
                    f"Global variable {var_name} has value {v1} when compiling {self.name}, but inner kernel "
                    f"{func.__name__} has conflicting value {v2} from when it was first compiled. This is not allowed.")
        self.used_global_vals.update(func.used_global_vals)
        # update hash
        func_key = func.cache_key
        func_key += str(getattr(func, "noinline", False))
        self.hasher.update(func_key.encode("utf-8"))

    def record_reference(self, val, var_dict=None, name=None):
        # Only keep track of "interesting" global variables, that non-evil users
        # might change.  Don't consider functions, modules, builtins, etc.  This
        # helps keep the list of vars we have to check small.
        if val is None or isinstance(val, types.ModuleType) or jit_allowed(val):
            return

        module = getattr(val, "__module__", "")
        if module.startswith(PYASC_MODULE) or module in self.supported_modules:
            return

        if isinstance(val, Function):
            self.update_hash(val)
            return

        if (callable(val) and not isinstance(val, type) and not isinstance(val, ConstExpr)):
            raise RuntimeError(f"Unsupported function referenced: {val}")

        # Python default arguments are resolved only once, when the
        # function is defined.  So if you do `foo(a=A)` and the value of
        # A changes, foo will still use the old value of A.
        # It would be pretty evil if someone did `import x` and then
        # `x = blah`.
        if self.visiting_arg_default_value:
            return

        if var_dict is not None:
            self.used_global_vals[(name, id(var_dict))] = (copy.deepcopy(val), var_dict)
        return

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            return node.id

        if node.id in self.local_names:
            # The global name is hidden by the local name.
            return None

        def name_lookup(name):
            val = self.globals.get(name, None)
            if val is not None:
                return val, self.globals
            val = self.nonlocals.get(name, None)
            if val is not None:
                return val, self.nonlocals
            return None, None

        val, var_dict = name_lookup(node.id)
        if node.id in self.supported_python_builtins:
            return val

        self.record_reference(val, var_dict, node.id)
        return val

    def visit_Tuple(self, node):
        # We need to explicitly return the tuple values so that visit_Assign can
        # access them in the case of `a, b = ...`.
        return [self.visit(elt) for elt in node.elts]

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        while isinstance(lhs, ast.Attribute):
            lhs = self.visit(lhs.value)
        lhs_name = getattr(lhs, "__name__", "")
        if lhs is None or lhs_name in self.supported_modules:
            return None
        ret = getattr(lhs, node.attr)
        self.record_reference(ret)
        return ret

    def visit_FunctionDef(self, node):
        # Save the local name, which may hide the global name.
        self.local_names = {arg.arg for arg in node.args.args}
        self.generic_visit(node)

    def visit_arguments(self, node):
        # The purpose of this function is to visit everything in `arguments`
        # just like `generic_visit`, except when we're visiting default values
        # (i.e. the `foo` part of `def fn(x = foo)`), we set
        # self.visiting_arg_default_value = True.  This allows visit_Name to be
        # aware that we're inside function default values, which have special
        # semantics.

        # According to the AST docs, the arguments node has the following structure.
        # - arg* posonlyargs
        # - arg* args
        # - arg? vararg
        # - arg* kwonlyargs
        # - expr* kw_defaults
        # - arg? kwarg
        # - expr* defaults
        def visit_defaults(defaults):
            try:
                if self.visiting_arg_default_value:
                    raise ValueError("visiting_arg_default_value is wrong")
                self.visiting_arg_default_value = True
                for expr in defaults:
                    if expr is not None:
                        self.visit(expr)
            finally:
                self.visiting_arg_default_value = False

        for arg in itertools.chain(
                node.posonlyargs,
                node.args,
            [node.vararg] if node.vararg else [],
                node.kwonlyargs,
        ):
            self.visit(arg)

        visit_defaults(node.kw_defaults)

        if node.kwarg is not None:
            self.visit(node.kwarg)

        visit_defaults(node.defaults)

    def visitAssnTarget(self, node):
        # Target is either a single string, or a list of strings (if the assn
        # target is a tuple).
        target = self.visit(node)
        if isinstance(target, list):
            self.local_names |= set(target)
        else:
            self.local_names.add(target)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            # TODO(jlebar): I don't actually know how to hit this.  You don't
            # get it from `a, b = ...` -- in that case, node.targets is a single
            # Tuple, and in fact we *do* need to handle that case if we want
            # existing code to work.
            raise TypeError("Simultaneous multiple assignment is not supported.")

        self.visitAssnTarget(node.targets[0])

        # This will re-visit the target, but that's OK.
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.visitAssnTarget(node.target)

        # This will re-visit the target, but that's OK.
        self.generic_visit(node)

    def visit_For(self, node):
        self.visitAssnTarget(node.target)

        # This will re-visit the target, but that's fine.
        self.generic_visit(node)
