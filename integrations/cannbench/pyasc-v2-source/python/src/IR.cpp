/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "InitFuncDef.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/EmitAsc/Utils/Attributes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // automatic casts between containers and python types

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace mlir;

namespace {

constexpr unsigned index64 = 64;

OpPrintingFlags getOpPrintingFlags()
{
    auto printingFlags = OpPrintingFlags();
    printingFlags.enableDebugInfo();
    return printingFlags;
}

std::optional<SmallVector<emitasc::KernelArgument>> getKernelArgAttrs(ModuleOp op)
{
    func::FuncOp kernelFunc;
    op.walk([&](func::FuncOp fn) -> WalkResult {
        if (fn->hasAttrOfType<UnitAttr>(ascendc::attr::global)) {
            kernelFunc = fn;
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    if (!kernelFunc) {
        return std::nullopt;
    }
    SmallVector<emitasc::KernelArgument> kernelArgs;
    unsigned numArgs = kernelFunc.getNumArguments();
    kernelArgs.reserve(numArgs);
    for (unsigned i = 0; i < numArgs; i++) {
        auto attr = kernelFunc.getArgAttrOfType<emitasc::KernelArgumentAttr>(i, emitasc::attr::kernelArg);
        kernelArgs.push_back(attr ? attr.getValue() : emitasc::KernelArgument::Explicit);
    }
    return kernelArgs;
}

ShapedType cloneShapedType(Type type, std::optional<Type> elemType, const std::optional<std::vector<int64_t>>& shape)
{
    auto shapedType = llvm::dyn_cast_if_present<ShapedType>(type);
    if (!shapedType)
        throw std::runtime_error("clone_shaped_type(): must be shaped type");
    Type useElemType = elemType.has_value() ? *elemType : shapedType.getElementType();
    return shapedType.cloneWith(shape, useElemType);
}

void bindAttrs(py::module& m)
{
    m.attr("dynshape") = py::int_(ShapedType::kDynamic);
    m.attr("ub_block_size") = py::int_(ascendc::ubBlockSize);

    auto modAttr = m.def_submodule("attr");
    modAttr.attr("compilation_arch") = py::str(ascendc::attr::compilationArch);
    modAttr.attr("enable_debug") = py::str(ascendc::attr::enableDebug);
    modAttr.attr("gm_barrier") = py::str(asctile::attr::gmBarrier);
    modAttr.attr("kernel_type") = py::str(ascendc::attr::kernelType);
    modAttr.attr("memory_consumed") = py::str(ascendc::attr::memoryConsumed);
    modAttr.attr("static_alloc") = py::str(ascendc::attr::staticAlloc);
    modAttr.attr("soc_version") = py::str(ascendc::attr::socVersion);
    modAttr.attr("unroll_factor") = py::str(asctile::attr::unrollFactor);
    modAttr.attr("vf_vec_len") = py::str(ascendc::attr::vfVecLen);
}

void bindEnums(py::module& m)
{
    using ret = py::return_value_policy;

    py::enum_<ascendc::AddressSpace>(m, "AddressSpace", py::module_local())
        .value("ca", ascendc::AddressSpace::ca)
        .value("cb", ascendc::AddressSpace::cb)
        .value("cbuf", ascendc::AddressSpace::cbuf)
        .value("cc", ascendc::AddressSpace::cc)
        .value("Default", ascendc::AddressSpace::Default)
        .value("fbuf", ascendc::AddressSpace::fbuf)
        .value("gm", ascendc::AddressSpace::gm)
        .value("ubuf", ascendc::AddressSpace::ubuf);

    py::enum_<ascendc::AippInputFormat>(m, "AippInputFormat", py::module_local())
        .value("YUV420SP_U8", ascendc::AippInputFormat::YUV420SP_U8)
        .value("XRGB8888_U8", ascendc::AippInputFormat::XRGB8888_U8)
        .value("RGB888_U8", ascendc::AippInputFormat::RGB888_U8)
        .value("YUV400_U8", ascendc::AippInputFormat::YUV400_U8)
        .def_static("symbolize", [](uint8_t inputFormat) -> ascendc::AippInputFormat {
            return static_cast<ascendc::AippInputFormat>(inputFormat);
        });

    py::enum_<ascendc::CacheLine>(m, "CacheLine", py::module_local())
        .value("SINGLE_CACHE_LINE", ascendc::CacheLine::SINGLE_CACHE_LINE)
        .value("ENTIRE_DATA_CACHE", ascendc::CacheLine::ENTIRE_DATA_CACHE)
        .def_static("symbolize", [](uint8_t v) -> ascendc::CacheLine { return static_cast<ascendc::CacheLine>(v); });

    py::enum_<arith::CmpFPredicate>(m, "CmpFPredicate", py::module_local())
        .value("OEQ", arith::CmpFPredicate::OEQ)
        .value("ONE", arith::CmpFPredicate::ONE)
        .value("OGE", arith::CmpFPredicate::OGE)
        .value("OGT", arith::CmpFPredicate::OGT)
        .value("OLE", arith::CmpFPredicate::OLE)
        .value("OLT", arith::CmpFPredicate::OLT);

    py::enum_<arith::CmpIPredicate>(m, "CmpIPredicate", py::module_local())
        .value("eq", arith::CmpIPredicate::eq)
        .value("ne", arith::CmpIPredicate::ne)
        .value("sge", arith::CmpIPredicate::sge)
        .value("sgt", arith::CmpIPredicate::sgt)
        .value("sle", arith::CmpIPredicate::sle)
        .value("slt", arith::CmpIPredicate::slt);

    py::enum_<ascendc::DcciDst>(m, "DcciDst", py::module_local())
        .value("CACHELINE_ALL", ascendc::DcciDst::CACHELINE_ALL)
        .value("CACHELINE_UB", ascendc::DcciDst::CACHELINE_UB)
        .value("CACHELINE_OUT", ascendc::DcciDst::CACHELINE_OUT)
        .value("CACHELINE_ATOMIC", ascendc::DcciDst::CACHELINE_ATOMIC)
        .def_static("symbolize", [](uint8_t v) -> ascendc::DcciDst { return static_cast<ascendc::DcciDst>(v); });

    py::enum_<ascendc::MaskMode>(m, "MaskMode", py::module_local())
        .value("NORMAL", ascendc::MaskMode::NORMAL)
        .value("COUNTER", ascendc::MaskMode::COUNTER)
        .def_static("symbolize", [](uint8_t v) -> ascendc::MaskMode { return static_cast<ascendc::MaskMode>(v); });

    py::enum_<ascendc::ReduceOrder>(m, "ReduceOrder", py::module_local())
        .value("ORDER_VALUE_INDEX", ascendc::ReduceOrder::ORDER_VALUE_INDEX)
        .value("ORDER_INDEX_VALUE", ascendc::ReduceOrder::ORDER_INDEX_VALUE)
        .value("ORDER_ONLY_VALUE", ascendc::ReduceOrder::ORDER_ONLY_VALUE)
        .value("ORDER_ONLY_INDEX", ascendc::ReduceOrder::ORDER_ONLY_INDEX)
        .def_static(
            "symbolize", [](uint8_t v) -> ascendc::ReduceOrder { return static_cast<ascendc::ReduceOrder>(v); });

    py::enum_<ascendc::RoundMode>(m, "RoundMode", py::module_local())
        .value("CAST_NONE", ascendc::RoundMode::CAST_NONE)
        .value("CAST_RINT", ascendc::RoundMode::CAST_RINT)
        .value("CAST_FLOOR", ascendc::RoundMode::CAST_FLOOR)
        .value("CAST_CEIL", ascendc::RoundMode::CAST_CEIL)
        .value("CAST_ROUND", ascendc::RoundMode::CAST_ROUND)
        .value("CAST_TRUNC", ascendc::RoundMode::CAST_TRUNC)
        .value("CAST_ODD", ascendc::RoundMode::CAST_ODD)
        .def_static("symbolize", [](uint8_t v) -> ascendc::RoundMode { return static_cast<ascendc::RoundMode>(v); });

    py::enum_<ascendc::TPosition>(m, "TPosition", py::module_local())
        .def_static(
            "symbolize", [](uint8_t pos) -> ascendc::TPosition { return static_cast<ascendc::TPosition>(pos); });

    py::enum_<ascendc::CMPMODE>(m, "CMPMODE", py::module_local())
        .value("LT", ascendc::CMPMODE::LT)
        .value("GT", ascendc::CMPMODE::GT)
        .value("EQ", ascendc::CMPMODE::EQ)
        .value("LE", ascendc::CMPMODE::LE)
        .value("GE", ascendc::CMPMODE::GE)
        .value("NE", ascendc::CMPMODE::NE)
        .def_static(
            "symbolize", [](uint8_t cmpMode) -> ascendc::CMPMODE { return static_cast<ascendc::CMPMODE>(cmpMode); });

    py::enum_<ascendc::SELMODE>(m, "SELMODE", py::module_local())
        .value("VSEL_CMPMASK_SPR", ascendc::SELMODE::VSEL_CMPMASK_SPR)
        .value("VSEL_TENSOR_SCALAR_MODE", ascendc::SELMODE::VSEL_TENSOR_SCALAR_MODE)
        .value("VSEL_TENSOR_TENSOR_MODE", ascendc::SELMODE::VSEL_TENSOR_TENSOR_MODE)
        .def_static(
            "symbolize", [](uint8_t selMode) -> ascendc::SELMODE { return static_cast<ascendc::SELMODE>(selMode); });

    py::enum_<asctile::TensorLocation>(
        m, "asctile_TensorLocation", py::module_local(), "A memory location of a :py:class:`LocalTensor`.")
        .value("Auto", asctile::TensorLocation::Auto, "Automatic resolution (if available)")
        .value("BT", asctile::TensorLocation::BT, "Bias Table buffer (Cube)")
        .value("L0A", asctile::TensorLocation::L0A, "L0A buffer (Cube)")
        .value("L0B", asctile::TensorLocation::L0B, "L0B buffer (Cube)")
        .value("L0C", asctile::TensorLocation::L0C, "L0C buffer (Cube)")
        .value("L1", asctile::TensorLocation::L1, "L1 buffer (Cube)")
        .value("UB", asctile::TensorLocation::UB, "Unified buffer (Vector)")
        .value("FIX", asctile::TensorLocation::FIX, "FixPipe buffer (Not supported)")
        .def(py::init([](const std::string& name) {
            auto normName = StringRef(name).upper();
            if (auto loc = asctile::symbolizeTensorLocation(normName))
                return *loc;
            throw py::value_error(name + " does not match a valid TensorLocation name");
        }))
        .def_static("symbolize", [](int32_t loc) -> asctile::TensorLocation {
            return static_cast<asctile::TensorLocation>(loc);
        });

    py::enum_<asctile::AtomicKind>(m, "AtomicKind", py::module_local())
        .value("Add", asctile::AtomicKind::Add)
        .value("Max", asctile::AtomicKind::Max)
        .value("Min", asctile::AtomicKind::Min)
        .def_static(
            "symbolize", [](int32_t kind) -> asctile::AtomicKind { return static_cast<asctile::AtomicKind>(kind); });

    py::enum_<asctile::CompareMode>(m, "CompareMode", py::module_local())
        .value("LT", asctile::CompareMode::LT)
        .value("GT", asctile::CompareMode::GT)
        .value("EQ", asctile::CompareMode::EQ)
        .value("LE", asctile::CompareMode::LE)
        .value("GE", asctile::CompareMode::GE)
        .value("NE", asctile::CompareMode::NE)
        .def_static(
            "symbolize", [](uint8_t mode) -> asctile::CompareMode { return static_cast<asctile::CompareMode>(mode); });

    py::enum_<asctile::ReduceKind>(m, "ReduceKind", py::module_local())
        .value("Sum", asctile::ReduceKind::Sum)
        .value("Max", asctile::ReduceKind::Max)
        .value("Min", asctile::ReduceKind::Min)
        .value("Prod", asctile::ReduceKind::Prod)
        .value("Mean", asctile::ReduceKind::Mean)
        .value("All", asctile::ReduceKind::All)
        .value("Any", asctile::ReduceKind::Any)
        .value("XorSum", asctile::ReduceKind::XorSum)
        .def_static("symbolize", [](int32_t kind) { return static_cast<asctile::ReduceKind>(kind); });

    py::enum_<asctile::RoundMode>(m, "asctile_RoundMode", py::module_local())
        .value("Default", asctile::RoundMode::Default)
        .value("NoRound", asctile::RoundMode::NoRound)
        .value("Rint", asctile::RoundMode::Rint)
        .value("Floor", asctile::RoundMode::Floor)
        .value("Ceil", asctile::RoundMode::Ceil)
        .value("Round", asctile::RoundMode::Round)
        .value("Trunc", asctile::RoundMode::Trunc)
        .value("Odd", asctile::RoundMode::Odd)
        .def_static("symbolize", [](int32_t mode) { return static_cast<asctile::RoundMode>(mode); });
}

void bindContextAndDialect(py::module& m)
{
    py::class_<MLIRContext>(m, "Context", py::module_local())
        .def(py::init<>())
        .def("disable_multithreading", [](MLIRContext& self) { self.disableMultithreading(); });

    m.def("load_dialects", [](MLIRContext& context) {
        DialectRegistry registry;
        registry.insert<
            //
            arith::ArithDialect, ascendc::AscendCDialect, asctile::AscTileDialect, emitasc::EmitAscDialect,
            emitc::EmitCDialect, func::FuncDialect, math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
            tensor::TensorDialect, vector::VectorDialect
            //
            >();
        ascendc::registerExternalModels(registry);
        ascendc::registerInlinerInterfaces(registry);
        asctile::registerExternalModels(registry);
        emitasc::registerExternalModels(registry);
        func::registerAllExtensions(registry);
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();
    });
}

void bindType(py::module& m)
{
    py::class_<Type>(m, "Type", py::module_local())
        .def("dump", &Type::dump)
        .def("is_integer", [](Type& self) -> bool { return self.isInteger(); })
        .def("is_index", &Type::isIndex)
        .def(
            "__eq__",
            [](Type& self, py::object& other) {
                Type* otherTy = py::cast<Type*>(other);
                return (otherTy != nullptr) && (*otherTy == self);
            })
        .def(
            "__ne__",
            [](Type& self, py::object& other) {
                Type* otherTy = py::cast<Type*>(other);
                return (otherTy == nullptr) || (*otherTy != self);
            })
        .def(
            "get_py_name",
            [](Type& self) -> std::optional<std::string> {
                if (isa<IntegerType>(self)) {
                    std::string name = self.isUnsignedInteger() ? "uint" : "int";
                    name += std::to_string(self.getIntOrFloatBitWidth());
                    return name;
                }
                if (isa<BFloat16Type>(self))
                    return "bfloat16";
                if (isa<Float16Type, Float32Type, Float64Type>(self)) {
                    std::string name = "float";
                    name += std::to_string(self.getIntOrFloatBitWidth());
                    return name;
                }
                if (isa<NoneType>(self))
                    return "void";
                return std::nullopt;
            })
        .def("__str__", [](Type& self) {
            std::string str;
            llvm::raw_string_ostream os(str);
            self.print(os);
            os.flush();
            return os.str();
        });
}

void bindMemref(py::module& m)
{
    using namespace pybind11::literals;
    m.def("get_element_type", [](const Type& shapedType) -> Type {
        auto type = llvm::dyn_cast_if_present<ShapedType>(shapedType);
        if (!type)
            throw std::runtime_error("get_element_type(): must be shaped type");
        return type.getElementType();
    });

    m.def("get_shape", [](const Type& shapedType) -> std::vector<int64_t> {
        auto type = llvm::dyn_cast_if_present<ShapedType>(shapedType);
        if (!type)
            throw std::runtime_error("get_shape(): must be shaped type");
        return type.getShape().vec();
    });

    m.def(
        "clone_shaped_type",
        [](Type shapedType, const std::vector<int64_t>& shape) -> Type {
            return cloneShapedType(shapedType, std::nullopt, shape);
        },
        "shaped_type"_a, "shape"_a);
    m.def(
        "clone_shaped_type",
        [](Type shapedType, Type elementType, const std::optional<std::vector<int64_t>>& shape) -> Type {
            return cloneShapedType(shapedType, elementType, shape);
        },
        "shaped_type"_a, "element_type"_a, "shape"_a = py::none());

    m.def("get_vector_type", [](Type& elementType, std::vector<int64_t>& shape) -> Type {
        return VectorType::get(shape, elementType);
    });

    m.def(
        "get_memref_type",
        [](Type& elementType, const std::variant<std::vector<int64_t>, int64_t>& shape,
           std::optional<int64_t> addressSpace) -> Type {
            Attribute memorySpace;
            if (auto as = addressSpace.value_or(0)) {
                memorySpace = IntegerAttr::get(IntegerType::get(elementType.getContext(), index64), as);
            }
            SmallVector<int64_t> sh;
            if (std::holds_alternative<int64_t>(shape)) {
                sh.push_back(std::get<int64_t>(shape));
            } else {
                const auto& shapeVec = std::get<std::vector<int64_t>>(shape);
                sh.append(shapeVec.begin(), shapeVec.end());
            }
            return MemRefType::get(sh, elementType, AffineMap{}, memorySpace);
        },
        "element_type"_a, "shape"_a, "address_space"_a = py::none());
    m.def(
        "get_unranked_memref_type",
        [](Type& elementType, std::optional<int64_t> addressSpace) -> Type {
            Attribute memorySpace;
            if (auto as = addressSpace.value_or(0))
                memorySpace = IntegerAttr::get(IntegerType::get(elementType.getContext(), index64), as);
            return UnrankedMemRefType::get(elementType, memorySpace);
        },
        "element_type"_a, "address_space"_a = py::none());
}

void bindTensorType(py::module& m)
{
    using namespace pybind11::literals;
    m.def("get_global_tensor_type", [](Type& elementType, std::vector<int64_t>& shape) -> Type {
        return ascendc::GlobalTensorType::get(shape, elementType);
    });

    m.def("get_global_tensor_type", [](Type& elementType) -> Type {
        return ascendc::GlobalTensorType::get(elementType);
    });

    m.def("get_local_tensor_type", [](Type& elementType, std::vector<int64_t>& shape) -> Type {
        return ascendc::LocalTensorType::get(shape, elementType);
    });

    m.def(
        "get_local_tensor_type", [](Type& elementType) -> Type { return ascendc::LocalTensorType::get(elementType); });

    m.def("get_opaque_type_name", [](Type& type) -> std::string {
        return cast<emitc::OpaqueType>(type).getValue().str();
    });
}

void bindAscTileType(py::module& m)
{
    using namespace pybind11::literals;

    m.def(
        "get_asctile_GlobalTensorType",
        [](const std::vector<int64_t>& shape, Type elementType) -> Type {
            return asctile::GlobalTensorType::get(shape, elementType);
        },
        "shape"_a, "element_type"_a);
    m.def(
        "get_asctile_LocalTensorType",
        [](const std::vector<int64_t>& shape, Type elementType, asctile::TensorLocation loc) -> Type {
            return asctile::LocalTensorType::get(shape, elementType, loc);
        },
        "shape"_a, "element_type"_a, "loc"_a = asctile::TensorLocation::UB);
    m.def(
        "get_tensor_location",
        [](Type type) -> asctile::TensorLocation {
            auto tileType = llvm::dyn_cast_if_present<asctile::LocalTensorType>(type);
            if (!tileType)
                throw std::runtime_error("get_tensor_location(): must be LocalTensorType");
            return tileType.getLoc();
        },
        "type"_a);
}

void bindLocation(py::module& m)
{
    py::class_<Location>(m, "Location", py::module_local()).def("__str__", [](Location& self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
    });
}

void bindValue(py::module& m)
{
    using ret = py::return_value_policy;
    py::class_<Value>(m, "Value", py::module_local())
        .def("get_context", &Value::getContext, ret::reference)
        .def(
            "get_defining_op",
            [](Value& self) -> std::optional<Operation*> {
                auto* def = self.getDefiningOp();
                if (def)
                    return def;
                return std::nullopt;
            },
            ret::reference)
        .def("replace_all_uses_with", [](Value& self, Value& newValue) { self.replaceAllUsesWith(newValue); })
        .def(
            "replace_uses_in_block",
            [](Value& self, Block* block, Value& newValue) {
                self.replaceUsesWithIf(newValue, [block](OpOperand& opnd) -> bool {
                    auto* op = opnd.getOwner();
                    Block* parentBlock = op->getBlock();
                    while (parentBlock) {
                        if (parentBlock == block)
                            return true;
                        if (auto* parentOp = parentBlock->getParentOp())
                            parentBlock = parentOp->getBlock();
                        else
                            parentBlock = nullptr;
                    }
                    return false;
                });
            })
        .def("get_type", &Value::getType)
        .def("dump", &Value::dump)
        .def("id", [](Value& self) { return reinterpret_cast<uint64_t>(self.getImpl()); });
}

void bindRegion(py::module& m)
{
    using ret = py::return_value_policy;
    py::class_<OpResult, Value>(m, "OpResult", py::module_local());
    py::class_<BlockArgument, Value>(m, "BlockArgument", py::module_local());
    py::class_<Region>(m, "Region", py::module_local())
        .def("get_parent_region", &Region::getParentRegion, ret::reference)
        .def(
            "get_block",
            [](Region& self, unsigned index) -> Block& {
                if (index >= self.getBlocks().size())
                    throw std::runtime_error("block index is out of range");
                return *std::next(self.begin(), index);
            },
            ret::reference)
        .def("size", [](Region& self) { return self.getBlocks().size(); })
        .def("empty", &Region::empty)
        .def("id", [](Region& self) { return (uint64_t)&self; });
}

void bindBlocks(py::module& m)
{
    using ret = py::return_value_policy;
    py::class_<Block>(m, "Block", py::module_local())
        .def(py::init())
        .def("dump", &Block::dump)
        .def("id", [](Block& self) { return (uint64_t)&self; })
        .def("has_terminator", &Block::mightHaveTerminator)
        .def("get_terminator", &Block::getTerminator, ret::reference)
        .def(
            "add_argument",
            [](Block& self, Type& type) -> BlockArgument {
                return self.addArgument(type, UnknownLoc::get(type.getContext()));
            })
        .def("get_argument", &Block::getArgument)
        .def("get_arguments", [](Block& self) -> std::vector<BlockArgument> { return self.getArguments().vec(); })
        .def(
            "merge_block_before",
            [](Block& self, Block& dst) {
                // See RewriterBase::mergeBlocks()
                if (self.getNumArguments() != 0)
                    throw std::runtime_error("Unable to merge block with arguments");
                dst.getOperations().splice(dst.begin(), self.getOperations());
                self.dropAllUses();
                if (self.getParent())
                    self.erase();
            })
        .def("clear", &Block::clear)
        .def("erase", &Block::erase);
}

void bindInlineBlock(py::module& m)
{
    using namespace pybind11::literals;

    m.def(
        "inline_block_at_end",
        [](Block* src, Block* dst, const std::optional<std::vector<Value>>& args) {
            // See RewriterBase::inlineBlockBefore()
            ValueRange argValues({});
            if (args)
                argValues = *args;
            auto before = dst->end();
            if (argValues.size() != src->getNumArguments())
                throw std::runtime_error("incorrect # of argument replacement values");
            // Replace all of the successor arguments with the provided values
            for (auto [arg, newVal] : llvm::zip(src->getArguments(), argValues))
                arg.replaceAllUsesWith(newVal);
            dst->getOperations().splice(before, src->getOperations());
            if (!src->empty()) {
                throw std::runtime_error("expected 'src' to be empty");
            }
            if (src->getParent())
                src->erase();
        },
        "src"_a, "dst"_a, "args"_a = py::none());
}

void bindAttritube(py::module& m)
{
    using ret = py::return_value_policy;
    py::class_<Attribute>(m, "Attribute", py::module_local())
        .def("dump", &Attribute::dump)
        .def("id", [](Attribute& self) { return reinterpret_cast<uint64_t>(self.getAsOpaquePointer()); });

    py::class_<ArrayAttr, Attribute>(m, "ArrayAttr", py::module_local());

    py::class_<TypedAttr>(m, "TypedAttr", py::module_local()).def("get_type", &TypedAttr::getType);

    m.def("get_type_attr", [](const Type& type) -> Attribute { return TypeAttr::get(type); });

    m.def("get_splat_attr", [](const Type& type, const Attribute& element) -> TypedAttr {
        auto shaped = dyn_cast<ShapedType>(type);
        if (!shaped)
            throw std::runtime_error("get_splat_attr(): type must be ShapedType");
        return SplatElementsAttr::get(shaped, element);
    });
}

void bindOperation(py::module& m)
{
    using ret = py::return_value_policy;
    py::class_<Operation, std::unique_ptr<Operation, py::nodelete>>(m, "Operation", py::module_local())
        .def(
            "get_name",
            [](Operation& self) {
                llvm::StringRef opName = self.getName().getStringRef();
                return opName.str();
            })
        .def("get_num_operands", &Operation::getNumOperands)
        .def("get_operand", &Operation::getOperand)
        .def("get_num_results", &Operation::getNumResults)
        .def("get_result", &Operation::getResult)
        .def("get_num_regions", &Operation::getNumRegions)
        .def("get_region", &Operation::getRegion, ret::reference)
        .def("get_block", &Operation::getBlock, ret::reference)
        .def(
            "has_unit_attr",
            [](Operation& self, const std::string& name) -> bool { return self.hasAttrOfType<UnitAttr>(name); })
        .def(
            "get_str_attr",
            [](Operation& self, const std::string& name) -> std::optional<std::string> {
                auto ret = self.getAttrOfType<StringAttr>(name);
                if (!ret)
                    return std::nullopt;
                return ret.getValue().str();
            })
        .def(
            "get_bool_attr",
            [](Operation& self, const std::string& name) -> std::optional<bool> {
                auto ret = self.getAttrOfType<BoolAttr>(name);
                if (!ret)
                    return std::nullopt;
                return ret.getValue();
            })
        .def(
            "get_integer_attr",
            [](Operation& self, const std::string& name) -> py::object {
                auto ret = self.getAttrOfType<IntegerAttr>(name);
                if (!ret)
                    return py::none();
                return py::int_(ret.getValue().getSExtValue());
            })
        .def(
            "get_dict_of_int_attr",
            [](Operation& self, const std::string& name) -> py::object {
                auto attr = self.getAttrOfType<DictionaryAttr>(name);
                if (!attr)
                    return py::none();
                py::dict result;
                for (auto entry : attr) {
                    py::str key(entry.getName().str());
                    py::int_ value(cast<IntegerAttr>(entry.getValue()).getValue().getSExtValue());
                    result[key] = value;
                }
                return result;
            })
        .def("get_flat_symbol_ref_attr", [](Operation& self, const std::string& name) -> py::object {
            auto ret = self.getAttrOfType<FlatSymbolRefAttr>(name);
            if (!ret)
                return py::none();
            return py::str(ret.getValue().str());
        });
}

void bindOpstate(py::module& m)
{
    using namespace py::literals;
    using ret = py::return_value_policy;
    py::class_<OpState>(m, "OpState", py::module_local())
        .def("get_context", &OpState::getContext, ret::reference)
        .def("set_attr", [](OpState& self, std::string& name, Attribute& attr) { self->setAttr(name, attr); })
        .def("get_num_results", [](OpState& self) -> unsigned { return self->getNumResults(); })
        .def(
            "get_result",
            [](OpState& self, unsigned idx) -> Value {
                if (idx >= self->getNumResults())
                    throw pybind11::index_error("Op result index out of range");
                return self->getResult(idx);
            })
        .def(
            "get_region",
            [](OpState& self, unsigned idx) -> Region& {
                if (idx >= self->getNumRegions())
                    throw pybind11::index_error("Op region index out of range");
                return self->getRegion(idx);
            },
            ret::reference)
        .def("dump", [](OpState& self) { self->dump(); })
        .def(
            "__str__",
            [](OpState& self) -> std::string {
                std::string str;
                llvm::raw_string_ostream os(str);
                auto printingFlags = getOpPrintingFlags();
                self->print(os, printingFlags);
                return str;
            })
        .def("append_operand", [](OpState& self, Value& val) { self->insertOperands(self->getNumOperands(), val); })
        .def(
            "verify",
            [](OpState& self, bool raising) -> bool {
                llvm::SourceMgr sourceMgr;
                std::string diagnostic;
                llvm::raw_string_ostream os(diagnostic);
                SourceMgrDiagnosticHandler handler(sourceMgr, self.getContext(), os);
                auto result = verify(self.getOperation());
                if (result.failed() && raising) {
                    constexpr StringLiteral prefix = "Failed to verify the operation";
                    diagnostic = (Twine(prefix) + (diagnostic.empty() ? "" : ":\n") + diagnostic).str();
                    throw std::runtime_error(diagnostic.c_str());
                }
                return result.succeeded();
            },
            "raising"_a = false)
        .def_property_readonly("op", &OpState::getOperation, ret::reference);
}

void bindModuleop(py::module& m)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    py::class_<ModuleOp, OpState>(m, "ModuleOp", py::module_local())
        .def("dump", &ModuleOp::dump)
        .def(
            "get_body", [](ModuleOp& self) -> Block* { return self.getBody(); }, ret::reference)
        .def(
            "has_function",
            [](ModuleOp& self, const std::string& name, const std::optional<Type>& type) -> bool {
                auto* op = SymbolTable::lookupSymbolIn(self, name);
                if (auto funcOp = dyn_cast_if_present<func::FuncOp>(op))
                    return !type || funcOp.getFunctionType() == *type;
                return false;
            },
            "name"_a, "type"_a = py::none())
        .def(
            "need_insert_sync",
            [](ModuleOp& self) {
                auto result = self.walk([](ascendc::LocalTensorAutoOp) { return WalkResult::interrupt(); });
                return result.wasInterrupted();
            })
        .def("erase", [](ModuleOp& self) { self->erase(); });
}

void bindFuncop(py::module& m)
{
    using ret = py::return_value_policy;
    py::class_<func::FuncOp, OpState>(m, "FuncOp", py::module_local())
        .def(
            "get_arg",
            [](func::FuncOp& self, unsigned idx) -> BlockArgument {
                if (idx >= self.getNumArguments())
                    throw pybind11::index_error("Function argument index out of range");
                return self.getArgument(idx);
            })
        .def("get_num_args", &func::FuncOp::getNumArguments)
        .def(
            "add_entry_block", [](func::FuncOp& self) -> Block* { return self.addEntryBlock(); }, ret::reference)
        .def(
            "set_type",
            [](func::FuncOp& self, const Type& funcType) {
                auto type = dyn_cast<FunctionType>(funcType);
                if (!type)
                    throw std::runtime_error("set_type(): must be FunctionType");
                self.setFunctionType(type);
            })
        .def(
            "set_arg_names",
            [](func::FuncOp& self, const std::vector<std::string>& names) {
                if (names.size() != self.getNumArguments())
                    throw std::runtime_error("Number of names must be equal to number of arguments");
                for (unsigned i = 0; i < names.size(); i++) {
                    auto arg = self.getArgument(i);
                    auto name = StringAttr::get(self.getContext(), names[i]);
                    arg.setLoc(NameLoc::get(name, arg.getLoc()));
                }
            })
        .def(
            "get_body", [](func::FuncOp& self) -> Block& { return self.getFunctionBody().front(); }, ret::reference)
        .def(
            "make_aicore",
            [](func::FuncOp& self) { self->setAttr(ascendc::attr::aicore, UnitAttr::get(self.getContext())); })
        .def("make_global", [](func::FuncOp& self) {
            self.setPublic();
            self->setAttr(ascendc::attr::global, UnitAttr::get(self.getContext()));
        });
}

void bindScfop(py::module& m)
{
    using ret = py::return_value_policy;
    py::class_<scf::ForOp, OpState>(m, "ForOp", py::module_local())
        .def("get_induction_var", &scf::ForOp::getInductionVar)
        .def("get_body", [](scf::ForOp& self) -> Block* { return self.getBody(); }, ret::reference);
    py::class_<scf::IfOp, OpState>(m, "IfOp", py::module_local())
        .def("get_then_block", &scf::IfOp::thenBlock, ret::reference)
        .def("get_else_block", &scf::IfOp::elseBlock, ret::reference)
        .def("get_then_yield", &scf::IfOp::thenYield)
        .def("get_else_yield", &scf::IfOp::elseYield);
    py::class_<scf::YieldOp, OpState>(m, "YieldOp", py::module_local());
    py::class_<scf::WhileOp, OpState>(m, "WhileOp", py::module_local())
        .def("get_before", &scf::WhileOp::getBefore, ret::reference)
        .def("get_after", &scf::WhileOp::getAfter, ret::reference);
    py::class_<scf::ConditionOp, OpState>(m, "ConditionOp", py::module_local());
}

void bindAscTile(py::module& m)
{
    using ret = py::return_value_policy;
    py::class_<asctile::CountMaskOp, OpState>(m, "CountMaskOp", py::module_local())
        .def("get_region", &asctile::CountMaskOp::getRegion, ret::reference);
    py::class_<asctile::BitwiseMaskOp, OpState>(m, "BitwiseMaskOp", py::module_local())
        .def("get_region", &asctile::BitwiseMaskOp::getRegion, ret::reference);
}

void bindKernelArgument(py::module& m)
{
    py::enum_<emitasc::KernelArgument>(m, "KernelArgument", py::module_local())
        .value("Explicit", emitasc::KernelArgument::Explicit)
        .value("FftsAddr", emitasc::KernelArgument::FftsAddr);

    m.def("get_kernel_arg_attrs", [](ModuleOp& mod) -> py::object {
        auto kernelArgs = getKernelArgAttrs(mod);
        if (!kernelArgs) {
            return py::none();
        }
        py::list result;
        for (auto arg : kernelArgs.value()) {
            result.append(arg);
        }
        return result;
    });
}

} // namespace

namespace pybind11 {
namespace asc {
void initIRModule(py::module&& m)
{
    bindAttrs(m);
    bindEnums(m);
    bindContextAndDialect(m);
    bindType(m);
    bindMemref(m);
    bindTensorType(m);
    bindAscTileType(m);
    bindLocation(m);
    bindValue(m);
    bindRegion(m);
    bindBlocks(m);
    bindInlineBlock(m);
    bindAttritube(m);
    bindOperation(m);
    bindOpstate(m);
    bindModuleop(m);
    bindFuncop(m);
    bindScfop(m);
    bindAscTile(m);
    bindKernelArgument(m);
    py::class_<OpBuilder::InsertPoint>(m, "InsertPoint", py::module_local());

    initBuilderInIRModule(m);
}
} // namespace asc
} // namespace pybind11
