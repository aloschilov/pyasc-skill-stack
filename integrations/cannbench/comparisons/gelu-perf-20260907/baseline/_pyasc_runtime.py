"""pyasc v2 integration helpers for the CANNBench worker process.

The current v2 snapshot has two integration defects that matter for 950PR:

* ``asctile.JITFunction`` inherits option discovery/extraction from the base
  ``asc`` class, so documented AscTile options such as ``vf_fusion`` and
  ``reuse_alloc`` are rejected or leak into the kernel call;
* the base ``asc.Compiler`` always emits an FFTS/C2C launch argument, while
  the AscTile compiler correctly omits it for the C310 architecture.

The small JIT subclasses below retain the upstream codegen and launcher while
selecting the concrete compiler option type. ``c310_asc_jit`` changes only the
base compiler's kernel-argument legalization for C310. This keeps the
candidate self-contained without modifying the vendored pyasc wheel.
"""

import inspect
import os
import threading
from dataclasses import fields
from typing import Callable, Optional

from asc._C import passes
from asc.codegen.function_visitor import CustomBuiltins
from asc.common.compat import get_annotations, merge_dict
from asc.language.core.range import range as asc_range
from asc.runtime.compiler import Compiler as AscCompiler
from asc.runtime.config import CompilationArch
from asc.runtime.jit import CompilePrereqs, JITFunction as AscJITFunction
from asctile.runtime.custom_builtins import get_custom_builtins
from asctile.runtime.jit import JITFunction as AscTileJITFunction


_lock = threading.Lock()
_initialised = False


class _ConcreteOptionsMixin:
    """Use the compiler class bound to a JIT subclass for option handling."""

    @classmethod
    def get_config_keywords(cls) -> list[str]:
        keywords: list[str] = []
        for datacls in (
            cls.codegen.options_cls,
            cls.compiler.options_cls,
            cls.launcher.options_cls,
        ):
            keywords.extend(field.name for field in fields(datacls))
        return list(dict.fromkeys(keywords))

    def __call__(self, *args, **kwargs) -> None:
        kwargs = merge_dict(self.default_options, kwargs)
        codegen_options = self.extract_kwargs(self.codegen.options_cls, kwargs)
        compile_options = self.extract_kwargs(self.compiler.options_cls, kwargs)
        call_args = inspect.signature(self.fn).bind(*args, **kwargs).arguments
        annotations = get_annotations(self.fn)
        runtime_args, constexprs = self.split_args(call_args, annotations)
        arg_types = {
            name: self.get_arg_type(value)
            for name, value in runtime_args.items()
        }
        prereqs = CompilePrereqs(
            arg_types, constexprs, codegen_options, compile_options
        )
        kernel, mem_cache_key = self._compile_and_cache(prereqs)
        self._run_launcher(
            kernel,
            self.launch_options,
            tuple(runtime_args.values()),
            mem_cache_key,
        )


class _AscTileJITFunction(_ConcreteOptionsMixin, AscTileJITFunction):
    pass


class _C310AscCompiler(AscCompiler):
    """Base asc compiler with architecture-correct FFTS ABI generation."""

    def _schedule_postprocessing(self, pm: passes.PassManager) -> None:
        passes.ascendc.add_declare_py_struct(pm)
        passes.ascendc.add_generate_boilerplate(pm)
        if self.options.matmul_cube_only:
            passes.ascendc.add_define_cube_only(pm)
        passes.ascendc.add_legalize_kernel_args(
            pm, set_ffts_addr=(self.arch != CompilationArch.C310)
        )
        passes.ascendc.add_detect_kernel_type(pm)
        passes.ascendc.add_detect_enable_debug(pm)
        if self.options.verify_sync:
            passes.ascendc.add_verify_sync(pm)
        if self.options.strip_loc:
            passes.common.add_strip_debug_info(pm)


class _C310AscJITFunction(_ConcreteOptionsMixin, AscJITFunction):
    compiler = _C310AscCompiler


def asctile_jit(fn: Optional[Callable] = None, **options):
    """AscTile JIT decorator with concrete v2 option extraction."""
    options.setdefault("custom_builtins", get_custom_builtins())

    def decorator(function: Callable) -> _AscTileJITFunction:
        return _AscTileJITFunction(function, **options)

    return decorator if fn is None else decorator(fn)


def c310_asc_jit(fn: Optional[Callable] = None, **options):
    """Low-level asc JIT with architecture-correct C310 kernel arguments."""
    options.setdefault("custom_builtins", CustomBuiltins(range=asc_range))

    def decorator(function: Callable) -> _C310AscJITFunction:
        return _C310AscJITFunction(function, **options)

    return decorator if fn is None else decorator(fn)


def ensure_npu_platform() -> None:
    global _initialised
    if _initialised:
        return
    with _lock:
        if _initialised:
            return
        import asc.runtime.config as config

        override = os.environ.get("CANN_BENCH_PYASC_PLATFORM")
        if override:
            config.set_platform(config.Backend.NPU, config.Platform(override))
        else:
            config.set_platform(config.Backend.NPU)
        _initialised = True
