# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
from typing import Literal, Optional

from asc._C import ir, passes
from asc.runtime.compiler import CompileOptions as CompileOptionsBase, Compiler as CompilerBase, CompilationArch


@dataclass
class CompileOptions(CompileOptionsBase):
    """Binary compilation and IR transformation options (for ``asc2`` kernels)"""

    insert_sync: bool = True

    reuse_alloc: Literal[0, 1, 2] = 0
    """
    Try to reduce the on-chip memory usage by replacing the tensors with those allocated earlier but became unused.
    Having this feature enabled may help to avoid memory overflow but may introduce performance regressions.

    ===== ======
    Value Effect
    ===== ======
    ``0`` Disable the feature (default)
    ``1`` Enable the feature, use a legacy implementation (recommended)
    ``2`` Enable the feature, use an experimental implementation
    ===== ======
    """

    static_alloc: Optional[bool] = None
    """
    Perform static allocation for tiles instead of relying on Ascend C TPipe backend.
    The static allocation feature may help to reduce an overhead caused by scalar code.

    **This feature is enabled by default** on supported platforms (such as ``Ascend950PR_9599``).
    """

    vf_fusion: bool = False
    """
    Fuse groups of consecutive vector operations into VF blocks using Ascend C register API.
    This feature may help to eliminate unnecessary memory transfers and improve data locality.
    """


class Compiler(CompilerBase):
    options_cls = CompileOptions

    def __init__(self, options: Optional[CompileOptions] = None):
        super().__init__(options)
        if self.options.vf_fusion and self.arch != CompilationArch.C310:
            raise RuntimeError(f"The vf fusion option is not supported for the {self.arch} architecture")

    def preprocess_module(self, mod: ir.ModuleOp) -> None:
        super().preprocess_module(mod)
        builder = ir.Builder(mod.op)
        if self.options.static_alloc is not None:
            mod.set_attr(ir.attr.static_alloc, builder.get_bool_attr(self.options.static_alloc))

    def schedule_passes(self, pm: passes.PassManager) -> None:
        arch_c310 = self.arch == CompilationArch.C310
        passes.ascendc.add_privatize_func(pm)
        passes.common.add_inliner(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_reconcile_unrealized_casts(pm)
        passes.asctile.add_resolve_auto_location(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.asctile.add_location_cast_to_copy(pm)
        passes.asctile.add_verify_tensor_location(pm)
        passes.asctile.add_split_cube_load(pm)
        passes.asctile.add_cube_transpose_to_load(pm)
        passes.asctile.add_legalize_matmul(pm)
        passes.common.add_canonicalizer(pm)
        passes.asctile.add_mark_matmul_acc_with_bias(pm)
        passes.asctile.add_fold_cast(pm)
        passes.asctile.add_transform_math_ops(pm)
        passes.asctile.add_transform_store_fixpipe(pm)
        passes.asctile.add_detect_bias_load(pm)
        passes.asctile.add_mark_reuse_source(pm)
        if arch_c310:
            passes.asctile.add_vector_transpose_to_load_store(pm)
            passes.asctile.add_unscalarize_reduction(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
        passes.asctile.add_wrap_cv_groups(pm)
        passes.asctile.add_merge_cv_groups(pm)
        passes.asclower.add_expand_math(pm)
        passes.asclower.add_redress_i1_tensor(pm)
        passes.asclower.add_lower_arith(pm)
        passes.asclower.add_lower_arith_binary(pm)
        passes.asclower.add_lower_atomic(pm)
        passes.asclower.add_lower_asctile_data_transfer(pm)
        passes.asclower.add_lower_asctile(pm)
        passes.asclower.add_lower_asctile_to_basic(pm)
        passes.asclower.add_lower_math(pm)
        passes.asclower.add_lower_scf(pm)
        passes.asclower.add_lower_tensor(pm)
        passes.asclower.add_displace_concat(pm)
        passes.common.add_canonicalizer(pm)
        passes.asclower.add_realize_conversion_cast(pm)
        passes.asclower.add_expand_mask(pm)
        passes.ascendc.add_promote_cv_block(pm)
        passes.ascendc.add_fill_asc_operands(pm)
        passes.ascendc.add_fixup_mmad_acc_params_pass(pm)
        passes.ascendc.add_input_output_tensor(pm)
        if self.options.reuse_alloc == 1:
            passes.ascendc.add_reuse_ub_allocation(pm, reuse_in_out=True)
        passes.asctile.add_unroll_loop(pm, annotate=True)
        passes.ascendc.add_compute_reuse_group(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.ascendc.add_hoist_tensor_allocation(pm, exclude_in_out=not arch_c310)
        passes.ascendc.add_refine_cube_position(pm)
        if self.options.reuse_alloc == 1:
            passes.ascendc.add_reuse_ub_allocation(pm, reuse_in_out=False)
        elif self.options.reuse_alloc == 2:
            passes.ascendc.add_reuse_tensor_allocation(pm)
        passes.common.add_canonicalizer(pm)
        if self.options.vf_fusion:
            passes.ascvf.add_find_vf_group(pm)
            passes.ascvf.add_lower_to_reg(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            passes.ascvf.add_reorder_ops_in_vec_scope(pm)
            passes.ascvf.add_fuse_vf_for(pm)
            passes.ascvf.add_eliminate_common_mask(pm)
            passes.ascvf.add_dispatch_hoist(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            passes.ascvf.add_insert_local_mem_bar(pm)
            passes.ascvf.add_materialize_load_store(pm)
        passes.ascendc.add_dispatch_alloc(pm)
        passes.ascendc.add_unify_bias_tensor(pm)
        passes.ascendc.add_unify_pipe(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_sccp(pm)
        passes.common.add_canonicalizer(pm)
        passes.ascendc.add_promote_cv_block(pm)
        passes.ascendc.add_insert_cross_core_sync(pm)
        if self.options.insert_sync:
            passes.ascendc.add_erase_sync(pm)
            passes.ascendc.add_hoist_que_bind(pm)
            if arch_c310:
                passes.ascendc.add_insert_bufid_sync(pm)
                passes.ascendc.add_insert_bias_bufid_sync(pm)
                passes.common.add_canonicalizer(pm)
                passes.ascendc.add_fuse_bufid_sync(pm)
            else:
                passes.ascendc.add_insert_que_sync(pm)
            passes.ascendc.add_unify_pipe(pm)
            passes.common.add_canonicalizer(pm)
        passes.ascendc.add_declare_py_struct(pm)
        passes.ascendc.add_generate_boilerplate(pm)
        passes.ascendc.add_insert_subblock_guard(pm)
        if self.options.matmul_cube_only:
            passes.ascendc.add_define_cube_only(pm)
        passes.ascendc.add_legalize_kernel_args(pm, set_ffts_addr=not arch_c310)
        passes.ascendc.add_detect_kernel_type(pm)
        passes.ascendc.add_detect_enable_debug(pm)
        if self.options.verify_sync:
            passes.ascendc.add_verify_sync(pm)
        if self.options.strip_loc:
            passes.common.add_strip_debug_info(pm)
        passes.ascendc.add_compute_memory_consumption(pm)
