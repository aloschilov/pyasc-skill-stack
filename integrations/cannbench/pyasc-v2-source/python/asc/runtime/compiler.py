# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Type

from .._C import ir, passes, translation
from ..lib.runtime import CoreType, get_soc_version
from ..lib.utils import get_ascend_path
from .config import CompilationArch, KernelType, platform_to_arch
from .kernel_meta import CompiledKernel, KernelMeta
from . import utils


@dataclass
class CompileOptions:
    """Binary compilation and IR transformation options"""

    debug: bool = False
    strip_loc: bool = False
    verify_sync: bool = False
    print_ir_before_all: bool = False
    run_passes: bool = True
    kernel_type: Optional[KernelType] = None
    auto_sync: Optional[bool] = True
    auto_sync_log: Optional[str] = ""
    matmul_cube_only: bool = False

    always_compile: bool = False
    """
    Always run full compilation pipeline instead of fetching a cached objects from the prevoius runs.
    This option may be useful for testing and parallel execution.
    """

    bisheng_options: Optional[Tuple[str, ...]] = None
    """
    Append extra arguments to the ``bisheng`` command line used to produce an object file for the kernel.
    Please, run ``bisheng --help`` to get the list of options supported by the compiler.

    Note: To pass a single argument, use ``("-O3",)`` (with a trailing comma).
    Without the comma, ``("-O3")`` is just a string, not a tuple.
    """

    opt_level: int = 3
    """
    Optimization level for the Bisheng compiler. Supported values are ``1``, ``2``, ``3``.
    Typically, this parameter affects the ``-O`` argument of the command line for the compiler.
    """

    insert_sync: Optional[bool] = None
    """
    Insert synchronization instructions automatically.
    **This feature is enabled by default**, which is usually a must, but may be disabled for the debugging purposes.
    """

    vf_vec_len: Optional[int] = None
    """
    Vector register length for the C310 architecture.
    This option is only available on supported platforms (such as ``Ascend950PR_9599``). The default value is 256.
    """


@dataclass(frozen=True)
class CompilationTarget:
    common_arch: Optional[str] = None
    vec_arch: Optional[str] = None
    cube_arch: Optional[str] = None
    common_options: List[str] = field(default_factory=list)
    vec_options: List[str] = field(default_factory=list)
    cube_options: List[str] = field(default_factory=list)

    @staticmethod
    def get(kernel_type: KernelType, arch: CompilationArch) -> CompilationTarget:
        if arch in [CompilationArch.C220, CompilationArch.C310]:
            common_option = [
                "-std=c++17", "--cce-disable-kernel-global-attr-check", "-mllvm", "-cce-aicore-stack-size=0x8000",
                "-mllvm", "-cce-aicore-function-stack-size=0x8000", "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false"
            ]
            if kernel_type in [KernelType.MIX_AIC_1_1, KernelType.MIX_AIC_1_2]:
                return CompilationTarget(vec_arch="dav-%s-vec" % arch, cube_arch="dav-%s-cube" % arch,
                                         common_options=common_option)
            elif kernel_type in [KernelType.MIX_AIV_1_0, KernelType.MIX_AIV_HARD_SYNC, KernelType.AIV_ONLY]:
                return CompilationTarget(common_arch="dav-%s-vec" % arch, common_options=common_option)
            else:
                return CompilationTarget(common_arch="dav-%s-cube" % arch, common_options=common_option)
        raise RuntimeError(f"Compilation is not supported for {arch.value} platform")


class Compiler:
    options_cls: Type[CompileOptions] = CompileOptions

    def __init__(self, options: Optional[CompileOptions] = None):
        self.options = self.options_cls() if options is None else options
        self.soc_version = get_soc_version()
        if not self._check_compile_options():
            raise RuntimeError("Please check input compile option")
        self.arch = platform_to_arch(self.soc_version)
        if self.options.vf_vec_len is not None and self.arch != CompilationArch.C310:
            raise RuntimeError(f"The vector register length option is not supported for the {self.arch} architecture")
        if self.options.vf_vec_len is None and self.arch == CompilationArch.C310:
            self.options.vf_vec_len = 256
        self.dump_dir: Optional[Path] = None
        dump_dir = os.environ.get("PYASC_DUMP_PATH", None)
        if dump_dir is not None:
            try:
                self.dump_dir = Path(dump_dir).resolve()
            except OSError as e:
                raise RuntimeError("Get {} realpath failed.".format(str(dump_dir))) from e
        utils.FileUtils.create_dir(self.dump_dir)
        self.enable_debug = False
        compiler = shutil.which(os.environ.get("PYASC_COMPILER", "bisheng"))
        if compiler is None:
            raise RuntimeError("Compiler executable is not found, check PYASC_COMPILER environment variable")
        self.compiler = compiler
        linker = shutil.which(os.environ.get("PYASC_LINKER", "ld.lld"))
        if linker is None:
            raise RuntimeError("Linker executable is not found, check PYASC_LINKER environment variable")
        self.linker = linker

    @staticmethod
    def run_translation(mod: ir.ModuleOp) -> str:
        return translation.ir_to_ascendc(mod)

    @staticmethod
    def _schedule_lowering(pm: passes.PassManager) -> None:
        passes.ascendc.add_privatize_func(pm)
        passes.common.add_inliner(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_reconcile_unrealized_casts(pm)
        passes.ascendc.add_input_output_tensor(pm)
        passes.ascendc.add_hoist_tensor_allocation(pm, exclude_in_out=True)
        passes.ascendc.add_materialize_tensor(pm, always_buf=False)
        passes.ascendc.add_unify_pipe(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

    def _schedule_optimizing(self, pm: passes.PassManager) -> None:
        passes.common.add_licm(pm)
        passes.common.add_sccp(pm)
        passes.common.add_canonicalizer(pm)
        if self.options.insert_sync:
            passes.ascendc.add_erase_sync(pm)
            passes.ascendc.add_hoist_que_bind(pm)
            passes.ascendc.add_insert_que_sync(pm)
            passes.ascendc.add_unify_pipe(pm)
            passes.common.add_canonicalizer(pm)

    @staticmethod
    def _run_cmd(cmd: List[str], cmd_type: str) -> None:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        is_compile_cmd = cmd_type == "compile"
        if proc.returncode != 0:
            if is_compile_cmd and "--cce-aicore-only" in cmd:
                cmd_idx_four = 4
                cmd_idx_five = 5
                cmd.insert(cmd_idx_four, "-mllvm")
                cmd.insert(cmd_idx_five, "-disable-machine-licm")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                out, ret = proc.communicate()
                if ret == 0:
                    return
            raise RuntimeError("{} failed!\nError message is {}\nPlease rerun {}".format(
                cmd_type, out.decode(), (" ".join(cmd))))

    def preprocess_module(self, mod: ir.ModuleOp) -> None:
        builder = ir.Builder(mod.op)
        mod.set_attr(ir.attr.compilation_arch, builder.get_str_attr(self.arch.value))
        mod.set_attr(ir.attr.soc_version, builder.get_str_attr(self.soc_version.value))
        if self.options.vf_vec_len is not None:
            mod.set_attr(ir.attr.vf_vec_len, builder.get_i32_attr(self.options.vf_vec_len))

    def postprocess_module(self, mod: ir.ModuleOp) -> None:
        if self.options.kernel_type is None:
            kernel_type = mod.op.get_str_attr(ir.attr.kernel_type)
            if kernel_type == "mixed":
                self.options.kernel_type = (KernelType.AIC_ONLY
                                            if self.options.matmul_cube_only else KernelType.MIX_AIC_1_2)
            elif kernel_type == "vector":
                self.options.kernel_type = KernelType.AIV_ONLY
            elif kernel_type == "cube":
                self.options.kernel_type = KernelType.AIC_ONLY
        self.enable_debug = (mod.op.has_unit_attr(ir.attr.enable_debug)
                             and str(os.environ.get("ASCENDC_DUMP", "True")).lower() == "true")

    def run(self, mod: ir.ModuleOp, func_name: str) -> CompiledKernel:
        utils.FileUtils.dump_file(self.dump_dir, "codegen.mlir", str(mod))
        self.preprocess_module(mod)
        if self.options.run_passes:
            self.run_passes(mod)
        self.postprocess_module(mod)
        utils.FileUtils.dump_file(self.dump_dir, "ascir.mlir", str(mod))
        source = self.run_translation(mod)
        if self.enable_debug:
            source = self._gen_init_dump_code(source, func_name)
        utils.FileUtils.dump_file(self.dump_dir, "ascendc.cpp", source)
        kernel_args = ir.get_kernel_arg_attrs(mod)
        return self.run_compilation(source, kernel_args, enable_debug=self.enable_debug,
                                    memory_consumed=mod.op.get_dict_of_int_attr(ir.attr.memory_consumed))

    def run_passes(self, mod: ir.ModuleOp) -> None:
        pm = passes.PassManager(mod.get_context())
        pm.enable_verifier()
        if self.options.print_ir_before_all or os.environ.get("PYASC_PRINT_IR"):
            pm.enable_printing()
        if self.options.insert_sync is None:
            self.options.insert_sync = mod.need_insert_sync()
        self.schedule_passes(pm)
        pm.run(mod)

    def run_compilation(self, source: str, kernel_args: Optional[Tuple[ir.KernelArgument]] = None,
                        **compiled_kernel_args) -> CompiledKernel:
        with tempfile.TemporaryDirectory(prefix="pyasc_compiler_") as tmp_dir:
            src = Path(tmp_dir) / "input.cce"
            src.write_text(source)
            dst = Path(tmp_dir) / "output.o"
            self._gen_dst_kernel(tmp_dir, src, dst)
            if self.dump_dir is not None:
                shutil.copyfile(dst, self.dump_dir / "binary.o")
            if self.options.kernel_type in [KernelType.MIX_AIC_1_1, KernelType.MIX_AIC_1_2]:
                core_type = CoreType.AiCore
            elif self.options.kernel_type in [
                    KernelType.AIV_ONLY, KernelType.MIX_AIV_HARD_SYNC, KernelType.MIX_AIV_1_0
            ]:
                core_type = CoreType.VectorCore
            else:
                core_type = CoreType.CubeCore
            return CompiledKernel(dst.read_bytes(), KernelMeta(core_type, kernel_args, **compiled_kernel_args))

    def schedule_passes(self, pm: passes.PassManager) -> None:
        self._schedule_lowering(pm)
        self._schedule_optimizing(pm)
        self._schedule_postprocessing(pm)

    def _check_compile_options(self) -> bool:
        is_soc_version_valid = (self.soc_version.value.startswith("Ascend910B")
                                or self.soc_version.value.startswith("Ascend910_93")
                                or self.soc_version.value.startswith("Ascend950PR_95"))
        is_core_type_valid = (self.options.kernel_type is None
                              or (isinstance(self.options.kernel_type, KernelType)
                                  and self.options.kernel_type.value <= 7 and self.options.kernel_type.value >= 0))
        is_opt_level_valid = self.options.opt_level in [1, 2, 3]
        return is_soc_version_valid and is_core_type_valid and is_opt_level_valid

    def _schedule_postprocessing(self, pm: passes.PassManager) -> None:
        passes.ascendc.add_declare_py_struct(pm)
        passes.ascendc.add_generate_boilerplate(pm)
        if self.options.matmul_cube_only:
            passes.ascendc.add_define_cube_only(pm)
        passes.ascendc.add_legalize_kernel_args(pm, set_ffts_addr=True)
        passes.ascendc.add_detect_kernel_type(pm)
        passes.ascendc.add_detect_enable_debug(pm)
        if self.options.verify_sync:
            passes.ascendc.add_verify_sync(pm)
        if self.options.strip_loc:
            passes.common.add_strip_debug_info(pm)

    def _gen_init_dump_code(self, source: str, func_name: str) -> str:
        dump_code = ""
        dump_code += "  #if defined ASCENDC_DUMP\n"
        dump_code += "    constexpr uint32_t ascendc_one_core_dump_size = " + str(utils.ONE_CORE_DUMP_SIZE) + ";\n"
        if self.options.kernel_type in [
                KernelType.MIX_AIC_1_0, KernelType.MIX_AIV_1_0, KernelType.MIX_AIC_1_1, KernelType.MIX_AIC_1_2
        ]:
            dump_code += "    AscendC::InitDump(true, dump_addr, ascendc_one_core_dump_size);\n"
        else:
            dump_code += "    AscendC::InitDump(false, dump_addr, ascendc_one_core_dump_size);\n"
        if self.enable_debug:
            dump_code += "    uint64_t asc_timestamp = 0;\n"
            dump_code += "    uint64_t asc_version = 0;\n"
            dump_code += "    __gm__ char* asc_version_str = nullptr;\n"
            dump_code += "    GetCannVersion(asc_version_str, asc_version, asc_timestamp);\n"
            dump_code += "    if (asc_timestamp == 0) {\n"
            dump_code += "      AscendC::printf(\"[WARNING]: CANN TimeStamp is invalid, \
                CANN TimeStamp is %u\\n\", asc_timestamp);\n"

            dump_code += "    } else {\n"
            dump_code += "      AscendC::printf(\"CANN Version: %s, TimeStamp: %u\\n\", \
                (__gm__ const char*)(asc_version_str), asc_timestamp);\n"

            dump_code += "    }\n"
        dump_code += "  #endif\n"
        source_lines = source.split('\n')
        kernel_code_with_dump = ""
        for line in source_lines:
            if func_name in line and "__aicore__" in line:
                split_line = line.split(")")
                new_line = split_line[0] + ", __gm__ uint8_t* dump_addr)" + split_line[1]
                kernel_code_with_dump += new_line + "\n"
                kernel_code_with_dump += dump_code
            else:
                kernel_code_with_dump += line + "\n"
        return kernel_code_with_dump

    def _gen_dst_kernel(self, tmp_dir: str, src: Path, dst: Path) -> None:
        target = CompilationTarget.get(self.options.kernel_type, self.arch)
        common_options = []
        ascend_path = get_ascend_path()
        tikcpp_path = os.path.realpath(os.path.join(ascend_path, "compiler", "tikcpp"))
        cann_version_file_path = os.path.join(tikcpp_path, "..", "..", "include", "version", "cann_version.h")
        if os.path.exists(cann_version_file_path):
            common_options += ["-include", cann_version_file_path]
        common_options += [
            "-I",
            os.path.join(tikcpp_path, "tikcfw"), "-I",
            os.path.join(tikcpp_path, "tikcfw/impl"), "-I",
            os.path.join(tikcpp_path, "tikcfw/interface")
        ]
        if self.options.bisheng_options:
            common_options.extend(self.options.bisheng_options)
        common_options.extend(target.common_options)
        if self.options.kernel_type in [KernelType.MIX_AIC_1_1, KernelType.MIX_AIC_1_2]:
            dst_cube_file = Path(tmp_dir) / "output_cube.o"
            cmds = self._get_compiler_cmd(target.cube_arch, src, dst_cube_file, common_options)
            self._run_cmd(cmds, "compile")
            dst_vec_file = Path(tmp_dir) / "output_vec.o"
            cmds = self._get_compiler_cmd(target.vec_arch, src, dst_vec_file, common_options)
            self._run_cmd(cmds, "compile")
            link_cmd = [
                self.linker, "-m", "aicorelinux", "-Ttext=0",
                "%s" % str(dst_cube_file),
                "%s" % str(dst_vec_file), "-static", "-o",
                "%s" % str(dst)
            ]
            self._run_cmd(link_cmd, "link")
        elif self.options.kernel_type in [
                KernelType.MIX_AIC_1_0, KernelType.MIX_AIC_HARD_SYNC, KernelType.MIX_AIV_1_0,
                KernelType.MIX_AIV_HARD_SYNC
        ]:
            dst_mix_file = Path(tmp_dir) / "output_mix_aic.o"
            if self.options.kernel_type in [KernelType.MIX_AIV_1_0, KernelType.MIX_AIV_HARD_SYNC]:
                dst_mix_file = Path(tmp_dir) / "output_mix_aiv.o"
            cmds = self._get_compiler_cmd(target.common_arch, src, dst_mix_file, common_options)
            self._run_cmd(cmds, "compile")
            link_cmd = [
                self.linker, "-m", "aicorelinux", "-Ttext=0",
                "%s" % str(dst_mix_file), "-static", "-o",
                "%s" % str(dst)
            ]
            self._run_cmd(link_cmd, "link")
        else:
            cmds = self._get_compiler_cmd(target.common_arch, src, dst, common_options)
            self._run_cmd(cmds, "compile")
            link_cmd = [self.linker, "-m", "aicorelinux", "-Ttext=0", "%s" % str(dst), "-static", "-o", "%s" % str(dst)]
            self._run_cmd(link_cmd, "link")

    def _get_compiler_cmd(self, arch: str, src_path: Path, dst_path: Path, common_options: List[str]) -> List[str]:
        opt_level = "-O" + str(self.options.opt_level)
        compile_cmds = [self.compiler, "-c", "-x", "cce", opt_level]
        compile_cmds += [str(src_path), "--cce-aicore-arch=%s" % arch, "--cce-aicore-only", "-o", str(dst_path)]
        compile_cmds += common_options
        if not self.enable_debug:
            compile_cmds += ["-DASCENDC_DUMP=0"]
        else:
            compile_cmds += ["-DASCENDC_DUMP=1"]
        if self.options.debug:
            compile_cmds += ["-g", "-mllvm", "--cce-aicore-jump-expand=true"]
        if self.arch == CompilationArch.C310:
            compile_cmds += ["-Xclang", f"-fcce-vf-vl={self.options.vf_vec_len}"]
        if self.options.auto_sync:
            compile_cmds += ["--cce-auto-sync", "-mllvm", "-api-deps-filter"]
            if self.options.auto_sync_log != "":
                compile_cmds += ["-cce-auto-sync-log=%s" % self.options.auto_sync_log]
        if self.options.kernel_type in [KernelType.MIX_AIC_1_1, KernelType.MIX_AIC_1_2]:
            compile_cmds += ["-cce-enable-mix"]
            compile_cmds += ["-D__MIX_CORE_MACRO__=1"]
        if self.options.kernel_type == KernelType.MIX_AIC_1_1:
            compile_cmds += ["-D__MIX_CORE_AIC_RATION__=1"]
        compile_cmds += ["-D__NPU_TILING__", "-DTILING_KEY_VAR=0"]
        return compile_cmds
