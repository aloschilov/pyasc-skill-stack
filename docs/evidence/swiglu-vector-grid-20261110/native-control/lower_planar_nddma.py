"""Lower the selected diagnostic specialization for inspection before any launch."""
import argparse
from importlib import import_module
from pathlib import Path
import sys


def lower_fp32(package, workers, kernel_name, geometry, output):
    return lower_typed(package, workers, kernel_name, ("float32", "float32"), geometry, output)


def lower_typed(package, workers, kernel_name, tensor_dtypes, geometry, output):
    sys.path[:0] = [str(package.parent), str(workers)]
    from asc import DataType
    from asc.codegen.specialization import Specialization
    from asc.runtime import config
    from asc.runtime.jit import MockTensor
    from local_compile_gate import prepare_specialization
    config.set_platform(config.Backend.Model, config.Platform.Ascend950PR_9599, check=False)
    kernel = getattr(import_module(f"{package.name}.swi_glu"), kernel_name)
    tensors = tuple(MockTensor(DataType(dtype)) for dtype in tensor_dtypes)
    _, types, constants, codegen, options = prepare_specialization(kernel, (*tensors, *geometry))
    module = kernel._run_codegen(Specialization(types, constants), codegen)
    compiler = kernel.compiler(options)
    compiler.preprocess_module(module)
    compiler.run_passes(module)
    compiler.postprocess_module(module)
    from asc._C import ir
    kernel.launcher.check_memory_overflow(module.op.get_dict_of_int_attr(ir.attr.memory_consumed) or {})
    source = compiler.run_translation(module)
    with output.open("x") as f:
        f.write(source)
    return source


def lower(package, workers, case, output):
    geometry = {11: (273, 509, 8, 512), 13: (143, 536, 4, 536)}[case]
    source = lower_fp32(package, workers, "planar_nddma", geometry, output)
    if source.count("AscendC::DataCopy<") + source.count("AscendC::DataCopy(") < 1 or "NdDmaParams<float,3>" not in source.replace(" ", ""):
        raise ValueError("expected a rank-three DMA descriptor; inspect the saved output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("workers", type=Path)
    parser.add_argument("case", type=int, choices=(11, 13))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lower(args.package.resolve(), args.workers.resolve(), args.case, args.output)
