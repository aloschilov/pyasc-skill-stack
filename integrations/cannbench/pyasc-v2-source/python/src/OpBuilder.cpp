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
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // automatic casts between containers and python types

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace mlir;

namespace {

std::vector<Type> noTypes;
std::vector<Value> noValues;

class PyOpBuilder {
    OpBuilder builder;
    Location loc;

public:
    explicit PyOpBuilder(MLIRContext* context) : builder(context), loc(builder.getUnknownLoc()) {}
    explicit PyOpBuilder(Operation* op) : builder(op), loc(op->getLoc()) {}
    ~PyOpBuilder() = default;

    void setLoc(Location newLoc) { loc = newLoc; }

    void setLoc(const std::string& name, bool reset = false)
    {
        if (reset) {
            setLoc(NameLoc::get(builder.getStringAttr(name)));
        } else {
            setLoc(NameLoc::get(builder.getStringAttr(name), loc));
        }
    }

    void setLoc(const std::string& fileName, int line, int column, const std::optional<std::string>& name)
    {
        Location newLoc = FileLineColLoc::get(builder.getContext(), fileName, line, column);
        if (name) {
            newLoc = NameLoc::get(builder.getStringAttr(*name), newLoc);
        }
        setLoc(newLoc);
    }

    Location getLoc() { return loc; }

    void resetLoc() { loc = builder.getUnknownLoc(); }

    OpBuilder& getBuilder() { return builder; }

    OpBuilder* operator->() { return &builder; }

    void setInsertionPointToStart(Block& block)
    {
        if (!block.empty()) {
            setLoc(block.begin()->getLoc());
        } else {
            resetLoc();
        }
        builder.setInsertionPointToStart(&block);
    }

    void setInsertionPointToEnd(Block& block)
    {
        if (!block.empty()) {
            setLoc(block.back().getLoc());
        } else {
            resetLoc();
        }
        builder.setInsertionPointToEnd(&block);
    }

    void setInsertionPointAfter(Operation& op)
    {
        setLoc(op.getLoc());
        builder.setInsertionPointAfter(&op);
    }

    void restoreInsertionPoint(OpBuilder::InsertPoint pt)
    {
        if (pt.isSet() && pt.getPoint() != pt.getBlock()->end()) {
            setLoc(pt.getPoint()->getLoc());
        } else if (pt.isSet() && !pt.getBlock()->empty()) {
            setLoc(pt.getBlock()->back().getLoc());
        } else {
            resetLoc();
        }
        builder.restoreInsertionPoint(pt);
    }

    Operation* create(
        StringRef operationName, ValueRange operands, TypeRange types = {}, ArrayRef<NamedAttribute> attributes = {})
    {
        return builder.create(loc, builder.getStringAttr(operationName), operands, types, attributes);
    }

    template <typename OpTy, typename... Args>
    auto create(Args&&... args) -> OpTy
    {
        return builder.create<OpTy>(loc, std::forward<Args>(args)...);
    }

    // Overload to create or fold a single result operation.
    template <typename OpTy, typename... Args>
    std::enable_if_t<OpTy::template hasTrait<OpTrait::OneResult>(), Value> createOrFold(Args&&... args)
    {
        return builder.createOrFold<OpTy>(loc, std::forward<Args>(args)...);
    }

    // Overload to create or fold a zero result operation.
    template <typename OpTy, typename... Args>
    std::enable_if_t<OpTy::template hasTrait<OpTrait::ZeroResults>(), OpTy> createOrFold(Args&&... args)
    {
        return builder.createOrFold<OpTy>(loc, std::forward<Args>(args)...);
    }

    std::optional<func::FuncOp> getCurrentFunction()
    {
        Block* block = builder.getInsertionBlock();
        if (!block) {
            return std::nullopt;
        }
        Operation* parent = block->getParentOp();
        if (!parent) {
            return std::nullopt;
        }
        if (auto op = dyn_cast<func::FuncOp>(parent)) {
            return op;
        }
        if (auto op = parent->getParentOfType<func::FuncOp>()) {
            return op;
        }
        return std::nullopt;
    }
};

ascendc::HardEvent getHardEvent(uint8_t event, const std::string& opName)
{
    if (auto eventAttr = ascendc::symbolizeHardEvent(event)) {
        return *eventAttr;
    }
    throw std::runtime_error(opName + ": unknown hard event");
}

void bindInit(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(py::init<MLIRContext*>());
    clss.def(py::init<Operation*>());
}

void bindLocations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def("get_loc", &PyOpBuilder::getLoc)
        .def(
            "set_loc", [](PyOpBuilder& self, Location& loc) { self.setLoc(loc); }, "loc"_a)
        .def(
            "set_loc", [](PyOpBuilder& self, const std::string& name, bool reset) { self.setLoc(name, reset); },
            "name"_a, "reset"_a = false)
        .def(
            "set_loc",
            [](PyOpBuilder& self, const std::string& fileName, int line, int column,
               const std::optional<std::string>& name) { self.setLoc(fileName, line, column, name); },
            "filename"_a, "line"_a, "column"_a, "name"_a = py::none());
}

void bindModifyInsertionPoint(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "set_insertion_point_to_start",
            [](PyOpBuilder& self, Block& block) { self.setInsertionPointToStart(block); })
        .def("set_insertion_point_to_end", [](PyOpBuilder& self, Block& block) { self.setInsertionPointToEnd(block); })
        .def("set_insertion_point", [](PyOpBuilder& self, Operation& op) { self->setInsertionPoint(&op); })
        .def("set_insertion_point", [](PyOpBuilder& self, OpState& op) { self->setInsertionPoint(op.getOperation()); })
        .def("set_insertion_point_after", [](PyOpBuilder& self, Operation& op) { self.setInsertionPointAfter(op); })
        .def(
            "get_insertion_block", [](PyOpBuilder& self) -> Block* { return self->getInsertionBlock(); },
            ret::reference)
        .def(
            "save_insertion_point",
            [](PyOpBuilder& self) -> OpBuilder::InsertPoint { return self->saveInsertionPoint(); })
        .def("restore_insertion_point", &PyOpBuilder::restoreInsertionPoint)
        .def("get_current_function", &PyOpBuilder::getCurrentFunction);
}

void bindGetBasicType(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def("get_none_type", [](PyOpBuilder& self) -> Type { return self->getNoneType(); })
        .def("get_index_type", [](PyOpBuilder& self) -> Type { return self->getIndexType(); })
        .def("get_int_type", [](PyOpBuilder& self, unsigned width) -> Type { return self->getIntegerType(width); })
        .def("get_i1_type", [](PyOpBuilder& self) -> Type { return self->getI1Type(); })
        .def("get_i8_type", [](PyOpBuilder& self) -> Type { return self->getI8Type(); })
        .def("get_i16_type", [](PyOpBuilder& self) -> Type { return self->getI16Type(); })
        .def("get_i32_type", [](PyOpBuilder& self) -> Type { return self->getI32Type(); })
        .def("get_i64_type", [](PyOpBuilder& self) -> Type { return self->getI64Type(); })
        .def(
            "get_uint_type",
            [](PyOpBuilder& self, unsigned width) -> Type { return self->getIntegerType(width, false); })
        .def("get_ui8_type", [](PyOpBuilder& self) -> Type { return self->getIntegerType(8U, false); })
        .def("get_ui16_type", [](PyOpBuilder& self) -> Type { return self->getIntegerType(16U, false); })
        .def("get_ui32_type", [](PyOpBuilder& self) -> Type { return self->getIntegerType(32U, false); })
        .def("get_ui64_type", [](PyOpBuilder& self) -> Type { return self->getIntegerType(64U, false); })
        .def(
            "get_float_type",
            [](PyOpBuilder& self, unsigned width) -> Type {
                if (width == 16U) {
                    return self->getF16Type();
                }
                if (width == 32U) {
                    return self->getF32Type();
                }
                if (width == 64U) {
                    return self->getF64Type();
                }
                throw std::runtime_error("Unsupported width for FloatType");
            })
        .def("get_f16_type", [](PyOpBuilder& self) -> Type { return self->getF16Type(); })
        .def("get_bf16_type", [](PyOpBuilder& self) -> Type { return self->getBF16Type(); })
        .def("get_f32_type", [](PyOpBuilder& self) -> Type { return self->getF32Type(); })
        .def("get_f64_type", [](PyOpBuilder& self) -> Type { return self->getF64Type(); });
}

void bindGetSpecialType(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "get_function_type",
            [](PyOpBuilder& self, const std::vector<Type>& inTypes, const std::optional<std::vector<Type>>& outTypes)
                -> Type { return self->getFunctionType(inTypes, outTypes.value_or(noTypes)); },
            "in_types"_a, "out_types"_a = py::none())
        .def(
            "get_opaque_type",
            [](PyOpBuilder& self, const std::string& name) -> Type {
                return self->getType<emitc::OpaqueType>(self->getStringAttr(name));
            })
        .def(
            "get_queue_type",
            [](PyOpBuilder& self, uint8_t position, int64_t depth) -> Type {
                auto pos = ascendc::symbolizeTPosition(position);
                if (!pos) {
                    throw std::runtime_error("Unknown position for QueueType");
                }
                return self->getType<ascendc::QueueType>(*pos, depth);
            })
        .def(
            "get_quebind_type",
            [](PyOpBuilder& self, uint8_t src, uint8_t dst, int64_t depth) -> Type {
                auto srcPos = ascendc::symbolizeTPosition(src);
                if (!srcPos) {
                    throw std::runtime_error("Unknown position for QueBindType");
                }
                auto dstPos = ascendc::symbolizeTPosition(dst);
                if (!dstPos) {
                    throw std::runtime_error("Unknown position for QueBindType");
                }
                return self->getType<ascendc::QueBindType>(*srcPos, *dstPos, depth);
            })
        .def(
            "get_buffer_type",
            [](PyOpBuilder& self, uint8_t position) -> Type {
                auto pos = ascendc::symbolizeTPosition(position);
                if (!pos) {
                    throw std::runtime_error("Unknown position for TBufType");
                }
                return self->getType<ascendc::TBufType>(*pos);
            })
        .def(
            "get_tbuf_pool_type",
            [](PyOpBuilder& self, uint8_t position, uint32_t bufIDSize) -> Type {
                auto pos = ascendc::symbolizeTPosition(position);
                if (!pos) {
                    throw std::runtime_error("Unknown position for TBufType");
                }

                return self->getType<ascendc::TBufPoolType>(*pos, bufIDSize);
            })
        .def(
            "get_matmul_type",
            [](PyOpBuilder& self, uint8_t posA, uint8_t fmtA, Type typeA, bool istransA, uint8_t layoutA, uint8_t posB,
               uint8_t fmtB, Type typeB, bool istransB, uint8_t layoutB, uint8_t posC, uint8_t fmtC, Type typeC,
               bool istransC, uint8_t layoutC, uint8_t posBias, uint8_t fmtBias, Type typeBias, bool doNorm,
               bool doBasicBlock, bool doMultiDataLoad, int32_t basicM, int32_t basicN, int32_t basicK,
               bool intrinsicsCheck, bool isNBatch, bool enVecNd2nz, bool doSpecialBasicBlock, int32_t doMte2Preload,
               int32_t singleCoreM, int32_t singleCoreN, int32_t singleCoreK, int32_t stepM, int32_t stepN,
               int32_t baseMn, int32_t singleCoreMn, bool enUnitFlag, bool isPerTensor, bool hasAntiQuantOffset,
               bool doIbShareNorm, bool doSpecialMdl, bool enableInit, int32_t batchMode, bool enableEnd,
               bool enableGetTensorC, bool enableSetOrgShape, bool enableSetBias, bool enableSetTail,
               bool enableQuantVector, bool enableSetDefineData, int32_t iterateMode, bool enableReuse,
               bool enableUbReuse, bool enableL1CacheUb, bool intraBlockPartSum, int32_t iterateOrder,
               int32_t scheduleType, bool enableDoubleCache, bool isBiasBatch, bool enableStaticPadZeros,
               bool isPartialOutput, bool enableMixDualMaster, bool isA2b2Shared, bool isEnableChannelSplit,
               bool enableKdimReorderLoad, bool isCo1Shared, int32_t sharedCo1BufferSize,
               int32_t batchOutMode) -> Type {
                auto sposA = ascendc::symbolizeTPosition(posA);
                auto sfmtA = ascendc::symbolizeCubeFormat(fmtA);
                auto slayoutA = ascendc::symbolizeLayoutMode(layoutA);
                auto sposB = ascendc::symbolizeTPosition(posB);
                auto sfmtB = ascendc::symbolizeCubeFormat(fmtB);
                auto slayoutB = ascendc::symbolizeLayoutMode(layoutB);
                auto sposC = ascendc::symbolizeTPosition(posC);
                auto sfmtC = ascendc::symbolizeCubeFormat(fmtC);
                auto slayoutC = ascendc::symbolizeLayoutMode(layoutC);
                auto sposBias = ascendc::symbolizeTPosition(posBias);
                auto sfmtBias = ascendc::symbolizeCubeFormat(fmtBias);
                mlir::MLIRContext* ctx = self.getBuilder().getContext();
                IntegerType i32 = IntegerType::get(ctx, 32);
                IntegerType i8 = IntegerType::get(ctx, 8);
                auto matmulConfigAttr = ascendc::MatmulConfigAttr::get(
                    self.getBuilder().getContext(), BoolAttr::get(ctx, doNorm), BoolAttr::get(ctx, doBasicBlock),
                    BoolAttr::get(ctx, doMultiDataLoad), IntegerAttr::get(i32, basicM), IntegerAttr::get(i32, basicN),
                    IntegerAttr::get(i32, basicK), BoolAttr::get(ctx, intrinsicsCheck), BoolAttr::get(ctx, isNBatch),
                    BoolAttr::get(ctx, enVecNd2nz), BoolAttr::get(ctx, doSpecialBasicBlock),
                    IntegerAttr::get(i32, doMte2Preload), IntegerAttr::get(i32, singleCoreM),
                    IntegerAttr::get(i32, singleCoreN), IntegerAttr::get(i32, singleCoreK),
                    IntegerAttr::get(i32, stepM), IntegerAttr::get(i32, stepN), IntegerAttr::get(i32, baseMn),
                    IntegerAttr::get(i32, singleCoreMn), BoolAttr::get(ctx, enUnitFlag),
                    BoolAttr::get(ctx, isPerTensor), BoolAttr::get(ctx, hasAntiQuantOffset),
                    BoolAttr::get(ctx, doIbShareNorm), BoolAttr::get(ctx, doSpecialMdl), BoolAttr::get(ctx, enableInit),
                    IntegerAttr::get(i32, batchMode), BoolAttr::get(ctx, enableEnd),
                    BoolAttr::get(ctx, enableGetTensorC), BoolAttr::get(ctx, enableSetOrgShape),
                    BoolAttr::get(ctx, enableSetBias), BoolAttr::get(ctx, enableSetTail),
                    BoolAttr::get(ctx, enableQuantVector), BoolAttr::get(ctx, enableSetDefineData),
                    IntegerAttr::get(i32, iterateMode), BoolAttr::get(ctx, enableReuse),
                    BoolAttr::get(ctx, enableUbReuse), BoolAttr::get(ctx, enableL1CacheUb),
                    BoolAttr::get(ctx, intraBlockPartSum), IntegerAttr::get(i32, iterateOrder),
                    IntegerAttr::get(i32, scheduleType), BoolAttr::get(ctx, enableDoubleCache),
                    BoolAttr::get(ctx, isBiasBatch), BoolAttr::get(ctx, enableStaticPadZeros),
                    BoolAttr::get(ctx, isPartialOutput), BoolAttr::get(ctx, enableMixDualMaster),
                    BoolAttr::get(ctx, isA2b2Shared), BoolAttr::get(ctx, isEnableChannelSplit),
                    BoolAttr::get(ctx, enableKdimReorderLoad), BoolAttr::get(ctx, isCo1Shared),
                    IntegerAttr::get(i32, sharedCo1BufferSize), IntegerAttr::get(i32, batchOutMode));
                return self->getType<ascendc::MatmulType>(
                    *sposA, *sfmtA, typeA, istransA, *slayoutA, *sposB, *sfmtB, typeB, istransB, *slayoutB, *sposC,
                    *sfmtC, typeC, istransC, *slayoutC, *sposBias, *sfmtBias, typeBias, matmulConfigAttr);
            })
        .def("get_asc_MaskType", [](PyOpBuilder& self) -> Type { return self->getType<ascendc::MaskType>(); })
        .def(
            "get_emitasc_PyStructType",
            [](PyOpBuilder& self, const std::string& name, const std::vector<Type>& types,
               const std::vector<std::string>& names) -> Type {
                if (types.size() != names.size()) {
                    throw std::runtime_error("Not equal types and names sizes");
                }
                SmallVector<StringRef> refs(names.begin(), names.end());
                return self->getType<emitasc::PyStructType>(
                    self->getStringAttr(name), self->getTypeArrayAttr(types), self->getStrArrayAttr(refs));
            })
        .def(
            "get_asc_DataCopyPadExtParamsType",
            [](PyOpBuilder& self, Type elementType) -> Type {
                return self->getType<ascendc::DataCopyPadExtParamsType>(elementType);
            })
        .def(
            "get_asc_MrgSortSrcListType",
            [](PyOpBuilder& self, Type elementType) -> Type {
                return self->getType<ascendc::MrgSortSrcListType>(elementType);
            })
        .def(
            "get_local_mem_allocator_type",
            [](PyOpBuilder& self, uint64_t hardware) -> Type {
                return ascendc::LocalMemAllocatorType::get(self->getContext(), hardware);
            })
#include "ascir/API/AscTypeBindings.h.inc"
        ;
}

void bindGetAttributes(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def("get_index_attr", [](PyOpBuilder& self, int64_t value) -> Attribute { return self->getIndexAttr(value); })
        .def("get_bool_attr", [](PyOpBuilder& self, bool value) -> Attribute { return self->getBoolAttr(value); })
        .def("get_i8_attr", [](PyOpBuilder& self, int8_t value) -> Attribute { return self->getI8IntegerAttr(value); })
        .def(
            "get_i16_attr",
            [](PyOpBuilder& self, int16_t val) -> Attribute { return self->getIntegerAttr(self->getI16Type(), val); })
        .def(
            "get_i32_attr",
            [](PyOpBuilder& self, int32_t value) -> Attribute { return self->getI32IntegerAttr(value); })
        .def(
            "get_i64_attr",
            [](PyOpBuilder& self, int64_t value) -> Attribute { return self->getI64IntegerAttr(value); })
        .def("get_f16_attr", [](PyOpBuilder& self, float value) -> Attribute { return self->getF16FloatAttr(value); })
        .def(
            "get_bf16_attr",
            [](PyOpBuilder& self, float value) -> Attribute { return self->getFloatAttr(self->getBF16Type(), value); })
        .def("get_f32_attr", [](PyOpBuilder& self, float value) -> Attribute { return self->getF32FloatAttr(value); })
        .def("get_f64_attr", [](PyOpBuilder& self, double value) -> Attribute { return self->getF64FloatAttr(value); })
        .def(
            "get_str_attr",
            [](PyOpBuilder& self, const std::string& value) -> Attribute { return self->getStringAttr(value); })
        .def("get_unit_attr", [](PyOpBuilder& self) -> Attribute { return self->getUnitAttr(); })
        .def(
            "get_type_array_attr",
            [](PyOpBuilder& self, const std::vector<Type>& types) -> ArrayAttr {
                return self->getTypeArrayAttr(types);
            })
        .def(
            "get_i32_array_attr",
            [](PyOpBuilder& self, const std::vector<int32_t>& values) -> ArrayAttr {
                return self->getI32ArrayAttr(values);
            })
        .def(
            "get_bool_array_attr",
            [](PyOpBuilder& self, const std::vector<bool>& values) -> ArrayAttr {
                llvm::SmallVector<bool> temp{values.begin(), values.end()};
                return self->getBoolArrayAttr(temp);
            })
        .def(
            "get_opaque_attr",
            [](PyOpBuilder& self, const std::string& value) -> Attribute {
                return self->getAttr<emitc::OpaqueAttr>(value);
            })
        .def("set_emit_as_unsigned", [](PyOpBuilder& self, Operation& op) {
            op.setAttr(ascendc::attr::emitAsUnsigned, self->getUnitAttr());
        });
}
void bindCreateOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create",
            [](PyOpBuilder& self, const std::string& name, const std::vector<Value>& operands,
               const std::vector<Type>& types) -> Operation* { return self.create(name, operands, types); },
            ret::reference)
        .def(
            "create_block",
            [](PyOpBuilder& self, std::optional<Region*> parent,
               const std::optional<std::vector<Type>>& types) -> Block* {
                Region* region = parent.value_or(self->getBlock()->getParent());
                auto typesVec = types.value_or(noTypes);
                std::vector<Location> locs(typesVec.size(), self->getUnknownLoc());
                return self->createBlock(region, Region::iterator(), types.value_or(noTypes), locs);
            },
            "parent"_a = py::none(), "types"_a = py::none(), ret::reference)
        .def("create_ModuleOp", [](PyOpBuilder& self) -> ModuleOp { return self.create<ModuleOp>(); })
        .def(
            "create_UnrealizedConversionCastOp",
            [](PyOpBuilder& self, const Type& result, const Value& value) -> Value {
                auto op = self.create<UnrealizedConversionCastOp>(result, value);
                return op.getResult(0);
            });
}

void bindCreateOperationsFunction(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_func_FuncOp",
            [](PyOpBuilder& self, const std::string& name, const Type& type) -> func::FuncOp {
                auto funcTy = dyn_cast<FunctionType>(type);
                if (!funcTy) {
                    throw std::invalid_argument("Invalid function type");
                }
                return self.create<func::FuncOp>(name, funcTy);
            })
        .def(
            "create_func_ReturnOp",
            [](PyOpBuilder& self, const std::optional<std::vector<Value>>& operands) {
                self.create<func::ReturnOp>(operands.value_or(noValues));
            },
            "operands"_a = py::none())
        .def(
            "create_func_CallOp",
            [](PyOpBuilder& self, const std::string& name, const std::optional<std::vector<Value>>& args,
               const std::optional<std::vector<Type>>& retTypes) -> OpState {
                return self.create<func::CallOp>(
                    self->getStringAttr(name), retTypes.value_or(noTypes), args.value_or(noValues));
            },
            "name"_a, "args"_a = py::none(), "ret_types"_a = py::none());
}

void bindCreateSignedConstants(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def("get_index", [](PyOpBuilder& self, int64_t v) -> Value { return self.create<arith::ConstantIndexOp>(v); })
        .def(
            "get_i1",
            [](PyOpBuilder& self, bool v) -> Value { return self.create<arith::ConstantOp>(self->getBoolAttr(v)); })
        .def(
            "get_i8",
            [](PyOpBuilder& self, int8_t v) -> Value {
                return self.create<arith::ConstantOp>(self->getI8IntegerAttr(v));
            })
        .def(
            "get_i16",
            [](PyOpBuilder& self, int16_t v) -> Value {
                return self.create<arith::ConstantOp>(self->getI16IntegerAttr(v));
            })
        .def(
            "get_i32",
            [](PyOpBuilder& self, int32_t v) -> Value {
                return self.create<arith::ConstantOp>(self->getI32IntegerAttr(v));
            })
        .def("get_i64", [](PyOpBuilder& self, int64_t v) -> Value {
            return self.create<arith::ConstantOp>(self->getI64IntegerAttr(v));
        });
}

void bindCreateUnsignedConstants(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "get_ui8",
            [](PyOpBuilder& self, uint8_t v) -> Value {
                auto type = self->getIntegerType(8U, false);
                return self.create<emitc::ConstantOp>(type, self->getIntegerAttr(type, v));
            })
        .def(
            "get_ui16",
            [](PyOpBuilder& self, uint16_t v) -> Value {
                auto type = self->getIntegerType(16U, false);
                return self.create<emitc::ConstantOp>(type, self->getIntegerAttr(type, v));
            })
        .def(
            "get_ui32",
            [](PyOpBuilder& self, uint32_t v) -> Value {
                auto type = self->getIntegerType(32U, false);
                return self.create<emitc::ConstantOp>(type, self->getIntegerAttr(type, v));
            })
        .def("get_ui64", [](PyOpBuilder& self, uint64_t v) -> Value {
            auto type = self->getIntegerType(64U, false);
            return self.create<emitc::ConstantOp>(type, self->getIntegerAttr(type, static_cast<int64_t>(v)));
        });
}

void bindCreateFloatConstants(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "get_f16",
            [](PyOpBuilder& self, float v) -> Value {
                return self.create<arith::ConstantOp>(self->getF16FloatAttr(v));
            })
        .def(
            "get_bf16",
            [](PyOpBuilder& self, float v) -> Value {
                return self.create<arith::ConstantOp>(self->getFloatAttr(self->getBF16Type(), v));
            })
        .def(
            "get_f32",
            [](PyOpBuilder& self, float v) -> Value {
                return self.create<arith::ConstantOp>(self->getF32FloatAttr(v));
            })
        .def("get_f64", [](PyOpBuilder& self, double v) -> Value {
            return self.create<arith::ConstantOp>(self->getF64FloatAttr(v));
        });
}

void bindCreateAirthBasicOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_arith_ConstantOp",
            [](PyOpBuilder& self, TypedAttr attr) -> Value { return self.create<arith::ConstantOp>(attr); })
        .def(
            "create_arith_AddIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::AddIOp>(lhs, rhs); })
        .def(
            "create_arith_AddFOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::AddFOp>(lhs, rhs); })
        .def(
            "create_arith_SubIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::SubIOp>(lhs, rhs); })
        .def(
            "create_arith_SubFOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::SubFOp>(lhs, rhs); })
        .def(
            "create_arith_MulIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::MulIOp>(lhs, rhs); })
        .def(
            "create_arith_MulFOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::MulFOp>(lhs, rhs); })
        .def(
            "create_arith_DivSIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::DivSIOp>(lhs, rhs); })
        .def(
            "create_arith_DivFOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::DivFOp>(lhs, rhs); })
        .def(
            "create_arith_MaxSIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::MaxSIOp>(lhs, rhs); })
        .def(
            "create_arith_MaximumFOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::MaximumFOp>(lhs, rhs); })
        .def(
            "create_arith_MinSIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::MinSIOp>(lhs, rhs); })
        .def(
            "create_arith_MinimumFOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::MinimumFOp>(lhs, rhs); })
        .def(
            "create_arith_ShLIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::ShLIOp>(lhs, rhs); })
        .def("create_arith_ShRSIOp", [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value {
            return self.create<arith::ShRSIOp>(lhs, rhs);
        });
}

void bindCreateAirthSpecialOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_arith_CeilDivSIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value {
                return self.create<arith::CeilDivSIOp>(lhs, rhs);
            })
        .def(
            "create_arith_RemSIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::RemSIOp>(lhs, rhs); })
        .def(
            "create_arith_RemFOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::RemFOp>(lhs, rhs); })
        .def(
            "create_arith_ShLIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::ShLIOp>(lhs, rhs); })
        .def("create_arith_ShRSIOp", [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value {
            return self.create<arith::ShRSIOp>(lhs, rhs);
        });
}

void bindCreateAirthComparisonOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_arith_AndIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::AndIOp>(lhs, rhs); })
        .def(
            "create_arith_OrIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::OrIOp>(lhs, rhs); })
        .def(
            "create_arith_XOrIOp",
            [](PyOpBuilder& self, Value& lhs, Value& rhs) -> Value { return self.create<arith::XOrIOp>(lhs, rhs); })
        .def(
            "create_arith_CmpIOp",
            [](PyOpBuilder& self, arith::CmpIPredicate pred, Value& lhs, Value& rhs) -> Value {
                return self.create<arith::CmpIOp>(pred, lhs, rhs);
            })
        .def(
            "create_arith_CmpFOp",
            [](PyOpBuilder& self, arith::CmpFPredicate pred, Value& lhs, Value& rhs) -> Value {
                return self.create<arith::CmpFOp>(pred, lhs, rhs);
            })
        .def("create_arith_SelectOp", [](PyOpBuilder& self, Value cond, Value lhs, Value rhs) -> Value {
            return self.create<arith::SelectOp>(cond, lhs, rhs);
        });
}

void bindCreateAirthExtendedOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_arith_NegFOp",
            [](PyOpBuilder& self, Value& opnd) -> Value { return self.create<arith::NegFOp>(opnd); })
        .def(
            "create_arith_IndexCastOp",
            [](PyOpBuilder& self, Value& value, Type& result) -> Value {
                return self.create<arith::IndexCastOp>(result, value);
            })
        .def(
            "create_arith_SIToFPOp",
            [](PyOpBuilder& self, Value& in, Type& out) -> Value { return self.create<arith::SIToFPOp>(out, in); })
        .def(
            "create_arith_FPToSIOp",
            [](PyOpBuilder& self, Value& in, Type& out) -> Value { return self.create<arith::FPToSIOp>(out, in); })
        .def(
            "create_arith_TruncIOp",
            [](PyOpBuilder& self, Value& in, Type& out) -> Value { return self.create<arith::TruncIOp>(out, in); })
        .def(
            "create_arith_TruncFOp",
            [](PyOpBuilder& self, Value& in, Type& out) -> Value { return self.create<arith::TruncFOp>(out, in); })
        .def(
            "create_arith_ExtSIOp",
            [](PyOpBuilder& self, Value& in, Type& out) -> Value { return self.create<arith::ExtSIOp>(out, in); })
        .def("create_arith_ExtFOp", [](PyOpBuilder& self, Value& in, Type& out) -> Value {
            return self.create<arith::ExtFOp>(out, in);
        });
}

void bindCreateMathOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;
    clss.def(
            "create_math_CosOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::CosOp>(value); })
        .def(
            "create_math_SinOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::SinOp>(value); })
        .def(
            "create_math_TanOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::TanOp>(value); })
        .def(
            "create_math_SinhOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::SinhOp>(value); })
        .def(
            "create_math_CoshOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::CoshOp>(value); })
        .def(
            "create_math_TanhOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::TanhOp>(value); })
        .def(
            "create_math_ExpOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::ExpOp>(value); })
        .def(
            "create_math_Exp2Op",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::Exp2Op>(value); })
        .def(
            "create_math_LogOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::LogOp>(value); })
        .def(
            "create_math_Log2Op",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::Log2Op>(value); })
        .def(
            "create_math_FloorOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::FloorOp>(value); })
        .def(
            "create_math_CeilOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::CeilOp>(value); })
        .def(
            "create_math_AbsFOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::AbsFOp>(value); })
        .def(
            "create_math_AbsIOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::AbsIOp>(value); })
        .def(
            "create_math_ErfOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::ErfOp>(value); })
        .def(
            "create_math_RsqrtOp",
            [](PyOpBuilder& self, Value& value) -> Value { return self.create<math::RsqrtOp>(value); })
        .def("create_math_SqrtOp", [](PyOpBuilder& self, Value& value) -> Value {
            return self.create<math::SqrtOp>(value);
        });
}

void bindCreateScfOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_scf_ForOp",
            [](PyOpBuilder& self, Value& lb, Value& ub, Value& step, std::optional<std::vector<Value>>& initArgs)
                -> scf::ForOp { return self.create<scf::ForOp>(lb, ub, step, initArgs.value_or(noValues)); },
            "lb"_a, "ub"_a, "step"_a, "init_args"_a = py::none())
        .def(
            "create_scf_IfOp",
            [](PyOpBuilder& self, Value& condition, const std::optional<std::vector<Type>>& retTypes, bool withElse)
                -> scf::IfOp { return self.create<scf::IfOp>(retTypes.value_or(noTypes), condition, withElse); },
            "condition"_a, "ret_types"_a = py::none(), "with_else"_a = false)
        .def(
            "create_scf_YieldOp",
            [](PyOpBuilder& self, std::optional<std::vector<Value>>& yields) -> scf::YieldOp {
                return self.create<scf::YieldOp>(yields.value_or(noValues));
            },
            "yields"_a = py::none())
        .def(
            "create_scf_WhileOp",
            [](PyOpBuilder& self, const std::optional<std::vector<Type>>& retTypes,
               const std::optional<std::vector<Value>>& initArgs) -> scf::WhileOp {
                return self.create<scf::WhileOp>(retTypes.value_or(noTypes), initArgs.value_or(noValues));
            },
            "ret_types"_a = py::none(), "init_args"_a = py::none())
        .def(
            "create_scf_ConditionOp",
            [](PyOpBuilder& self, Value& cond, const std::optional<std::vector<Value>>& args) -> scf::ConditionOp {
                return self.create<scf::ConditionOp>(cond, args.value_or(noValues));
            },
            "condition"_a, "args"_a = py::none());
}

void bindCreateMemrefOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_memref_AllocaOp",
            [](PyOpBuilder& self, const Type& type, bool emitAsUnsigned) -> Value {
                auto mrType = dyn_cast<MemRefType>(type);
                if (!mrType) {
                    throw std::runtime_error("create_memref_AllocaOp(): must be MemRefType");
                }
                auto op = self.create<memref::AllocaOp>(mrType);
                if (emitAsUnsigned) {
                    op->setAttr(ascendc::attr::emitAsUnsigned, self->getUnitAttr());
                }
                return op.getResult();
            },
            "type"_a, "unsigned"_a = false)
        .def(
            "create_memref_LoadOp",
            [](PyOpBuilder& self, const Value& base, const std::vector<Value>& indices) -> Value {
                return self.create<memref::LoadOp>(base, indices);
            },
            "base"_a, "indices"_a)
        .def(
            "create_memref_StoreOp",
            [](PyOpBuilder& self, const Value& src, const Value& base, const std::vector<Value>& indices) {
                self.create<memref::StoreOp>(src, base, indices);
            },
            "src"_a, "base"_a, "indices"_a);
}

void bindCreateVectorOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_vector_LoadOp",
            [](PyOpBuilder& self, Type& result, Value& base, const std::optional<std::vector<Value>>& indices)
                -> Value { return self.create<vector::LoadOp>(result, base, indices.value_or(noValues)); },
            "result"_a, "base"_a, "indices"_a = py::none())
        .def(
            "create_vector_StoreOp",
            [](PyOpBuilder& self, Value& value, Value& base, const std::optional<std::vector<Value>>& indices) {
                self.create<vector::StoreOp>(value, base, indices.value_or(noValues));
            },
            "value"_a, "base"_a, "indices"_a = py::none());
}

void bindCreateEmitcOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_emitc_CastOp",
            [](PyOpBuilder& self, const Value& in, const Type& type) -> Value {
                return self.create<emitc::CastOp>(type, in);
            })
        .def(
            "create_emitc_ConstantOp",
            [](PyOpBuilder& self, const Attribute& value, const Type& type) -> Value {
                return self.create<emitc::ConstantOp>(type, value);
            })
        .def(
            "create_emitc_IncludeOp",
            [](PyOpBuilder& self, const std::string& filename) { self.create<emitc::IncludeOp>(StringRef(filename)); })
        .def("create_emitc_VerbatimOp", [](PyOpBuilder& self, const std::string& str) {
            self.create<emitc::VerbatimOp>(StringRef(str));
        });
}

void bindCreateEmitascOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_emitasc_CopyStructOp",
            [](PyOpBuilder& self, const Value& base) -> Value {
                return self.create<emitasc::CopyStructOp>(cast<MemRefType>(base.getType()).getElementType(), base);
            })
        .def(
            "create_emitasc_MemberOp",
            [](PyOpBuilder& self, const Value& base, const std::string& field, const Type& type) -> Value {
                return self.create<emitasc::MemberOp>(type, base, field);
            })
        .def(
            "create_emitasc_PtrOffsetOp",
            [](PyOpBuilder& self, Value& base, Value& offset) -> Value {
                return self.create<emitasc::PtrOffsetOp>(base.getType(), base, IntegerAttr(), offset);
            })
        .def(
            "create_emitasc_SetMemberOp",
            [](PyOpBuilder& self, const Value& base, const std::string& field, const Value& value) {
                self.create<emitasc::SetMemberOp>(base, field, value);
            })
        .def(
            "create_emitasc_VerbatimOp",
            [](PyOpBuilder& self, const std::string& value, const std::optional<std::vector<Value>>& args) {
                self.create<emitasc::VerbatimOp>(self->getStringAttr(value), args.value_or(noValues));
            },
            "value"_a, "args"_a = py::none());
}

void bindCreateAscPipeOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_asc_FftsCrossCoreSyncOp",
            [](PyOpBuilder& self, uint8_t pipe, const Value& config) {
                auto pipeAttr = ascendc::symbolizePipe(pipe);
                if (!pipeAttr) {
                    throw std::runtime_error("Unknown pipe for FftsCrossCoreSyncOp");
                }
                self.create<ascendc::FftsCrossCoreSyncOp>(*pipeAttr, config);
            })
        .def("create_asc_PipeOp", [](PyOpBuilder& self) -> Value { return self.create<ascendc::PipeOp>(); })
        .def(
            "create_asc_PipeBarrierOp",
            [](PyOpBuilder& self, uint8_t pipe) {
                auto pipeAttr = ascendc::symbolizePipe(pipe);
                if (!pipeAttr) {
                    throw std::runtime_error("Unknown pipe for PipeBarrierOp");
                }
                self.create<ascendc::PipeBarrierOp>(*pipeAttr);
            })
        .def(
            "create_asc_SetFlagOp",
            [](PyOpBuilder& self, uint8_t event, Value eventId) {
                auto eventAttr = getHardEvent(event, "SetFlagOp");
                self.create<ascendc::SetFlagOp>(eventAttr, eventId);
            })
        .def(
            "create_asc_WaitFlagOp",
            [](PyOpBuilder& self, uint8_t event, Value eventId) {
                auto eventAttr = getHardEvent(event, "WaitFlagOp");
                self.create<ascendc::WaitFlagOp>(eventAttr, eventId);
            })
        .def(
            "create_asc_CrossCoreSetFlagOp",
            [](PyOpBuilder& self, Value flagId, uint8_t modeId, uint8_t pipe) {
                auto pipeAttr = ascendc::symbolizePipe(pipe);
                if (!pipeAttr) {
                    throw std::runtime_error("Unknown pipe for CrossCoreSetFlagOp");
                }
                self.create<ascendc::CrossCoreSetFlagOp>(flagId, modeId, *pipeAttr);
            })
        .def(
            "create_asc_CrossCoreWaitFlagOp",
            [](PyOpBuilder& self, Value flagId, uint8_t modeId, uint8_t pipe) {
                auto pipeAttr = ascendc::symbolizePipe(pipe);
                if (!pipeAttr) {
                    throw std::runtime_error("Unknown pipe for CrossCoreWaitFlagOp");
                }
                self.create<ascendc::CrossCoreWaitFlagOp>(flagId, modeId, *pipeAttr);
            })
        .def(
            "create_asc_TPipeAllocEventIDOp",
            [](PyOpBuilder& self, const Type& eventId, const Value& pipe, uint8_t event) -> Value {
                auto eventAttr = getHardEvent(event, "TPipeAllocEventIDOp");
                return self.create<ascendc::TPipeAllocEventIDOp>(eventId, pipe, eventAttr);
            })
        .def("create_asc_DataSyncBarrierOp", [](PyOpBuilder& self, uint8_t memDsbType) {
            auto memDsbAttr = ascendc::symbolizeMemDsbT(memDsbType);
            if (!memDsbAttr) {
                throw std::runtime_error("Unknown MemDsbT type for DataSyncBarrierOp");
            }
            self.create<ascendc::DataSyncBarrierOp>(*memDsbAttr);
        });
}

void bindCreateAscEventOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_asc_TPipeReleaseEventIDOp",
            [](PyOpBuilder& self, const Value& pipe, const Value& id, uint8_t event) {
                auto eventAttr = getHardEvent(event, "TPipeReleaseEventIDOp");
                self.create<ascendc::TPipeReleaseEventIDOp>(pipe, id, eventAttr);
            })
        .def(
            "create_asc_TPipeFetchEventIDOp",
            [](PyOpBuilder& self, const Type& eventId, const Value& pipe, uint8_t event) -> Value {
                auto eventAttr = getHardEvent(event, "TPipeFetchEventIDOp");
                return self.create<ascendc::TPipeFetchEventIDOp>(eventId, pipe, eventAttr);
            })
        .def(
            "create_asc_PrintfOp",
            [](PyOpBuilder& self, const std::string& desc, const std::optional<std::vector<Value>>& vars) {
                self.create<ascendc::PrintfOp>(self->getStringAttr(desc), vars.value_or(noValues));
            },
            "desc"_a, "vars"_a = py::none())
        .def(
            "create_asc_GlobalTensorSetL2CacheHintOp",
            [](PyOpBuilder& self, const Value& tensor, uint8_t mode, uint8_t rwMode) {
                auto modeAttr = ascendc::symbolizeCacheMode(mode);
                auto rwModeAttr = ascendc::symbolizeCacheRwMode(rwMode);

                if (!modeAttr) {
                    throw std::runtime_error("Unknown mode for GlobalTensorSetL2CacheHintOp");
                }
                if (!rwModeAttr) {
                    throw std::runtime_error("Unknown rwMode for GlobalTensorSetL2CacheHintOp");
                }
                self.create<ascendc::GlobalTensorSetL2CacheHintOp>(tensor, *modeAttr, *rwModeAttr);
            })
        .def(
            "create_asc_LocalTensorAutoOp",
            [](PyOpBuilder& self, const Type& result) -> Value {
                return self.create<ascendc::LocalTensorAutoOp>(result);
            },
            "result"_a)
#include "ascir/Dialect/Asc/IR/AscOpBindings.h.inc"
        ;
}

void bindCreateAscCommonOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
        "create_asc_GatherMaskAndResult",
        [](PyOpBuilder& self, const Type& type, const Value& dst, const Value& src0, const Value& src1Pattern,
           const Value& reduceMode, const Value& mask, const Value& params, uint8_t mode) {
            auto modeAttr = ascendc::symbolizeGatherMaskMode(mode);
            if (!modeAttr) {
                throw std::runtime_error("Unknown mode for GatherMaskOp");
            }
            return self.create<ascendc::GatherMaskOp>(type, dst, src0, src1Pattern, reduceMode, mask, params, *modeAttr)
                .getResult();
        });
    clss.def(
        "create_asc_GetMrgSortResults",
        [](PyOpBuilder& self, const Type& type1, const Type& type2, const Type& type3, const Type& type4) {
            auto operation = self.create<ascendc::GetMrgSortResultOp>(type1, type2, type3, type4);
            return py::make_tuple(
                operation.getResult(0), operation.getResult(1), operation.getResult(2), operation.getResult(3));
        });
    clss.def(
        "create_asc_GetStoreAtomicConfigAndResult",
        [](PyOpBuilder& self, const Type& type, const Type& op) {
            auto operation = self.create<ascendc::GetStoreAtomicConfigOp>(type, op);
            return py::make_tuple(operation.getResult(0), operation.getResult(1));
        },
        "type"_a, "op"_a);
    clss.def("create_asc_SetStoreAtomicConfigOp", [](PyOpBuilder& self, uint8_t type, uint8_t op) {
        auto typeAttr = ascendc::symbolizeAtomicDtype(type);
        if (!typeAttr) {
            throw std::runtime_error("Unknown type for AtomicDtype");
        }
        auto opAttr = ascendc::symbolizeAtomicOp(op);
        if (!opAttr) {
            throw std::runtime_error("Unknown type for AtomicOp");
        }
        self.create<ascendc::SetStoreAtomicConfigOp>(*typeAttr, *opAttr);
    });
    clss.def(
        "create_asc_GetStoreAtomicConfigAndResult",
        [](PyOpBuilder& self, const Type& type, const Type& op) {
            auto operation = self.create<ascendc::GetStoreAtomicConfigOp>(type, op);
            return py::make_tuple(operation.getResult(0), operation.getResult(1));
        },
        "type"_a, "op"_a);
}

void bindCreateAscTileOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_asctile_CountMaskOp",
            [](PyOpBuilder& self, Value& count, std::optional<Value> other) -> asctile::CountMaskOp {
                Value otherVal = other.has_value() ? *other : Value();
                return self.create<asctile::CountMaskOp>(count, otherVal);
            },
            py::arg("count"), py::arg("other") = py::none())
        .def(
            "create_asctile_BitwiseMaskOp",
            [](PyOpBuilder& self, Value& highBits, Value& lowBits,
               std::optional<Value> other) -> asctile::BitwiseMaskOp {
                Value otherVal = other.has_value() ? *other : Value();
                return self.create<asctile::BitwiseMaskOp>(highBits, lowBits, otherVal);
            },
            py::arg("highBits"), py::arg("lowBits"), py::arg("other") = py::none())
        .def(
            "create_asctile_InlineVFOp",
            [](PyOpBuilder& self, Type result, const std::vector<Value>& inputs, const std::string& code) -> Value {
                return self.create<asctile::InlineVFOp>(result, ValueRange{inputs}, StringRef(code));
            })
#include "ascir/Dialect/AscTile/IR/AscTileOpBindings.h.inc"
        ;
}

void bindCreateTensorOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_tensor_CastOp",
            [](PyOpBuilder& self, Type result, Value operand) -> Value {
                return self.create<tensor::CastOp>(result, operand);
            })
        .def(
            "create_tensor_ConcatOp",
            [](PyOpBuilder& self, Type result, uint64_t dim, const std::vector<Value>& inputs) -> Value {
                return self.create<tensor::ConcatOp>(result, dim, inputs);
            })
        .def(
            "create_tensor_SplatOp",
            [](PyOpBuilder& self, Type result, Value input) -> Value {
                return self.create<tensor::SplatOp>(result, input);
            })
        .def("cast_tensor_location", [](PyOpBuilder& self, asctile::TensorLocation loc, Value tensor) -> Value {
            auto type = dyn_cast<asctile::LocalTensorType>(tensor.getType());
            if (!type)
                throw std::runtime_error("cast_tensor_location(): value must have LocalTensorType");
            return self.create<tensor::CastOp>(
                asctile::LocalTensorType::get(type.getShape(), type.getElementType(), loc), tensor);
        });
}

} // namespace

namespace pybind11 {
namespace asc {

void initBuilderInIRModule(py::module& m)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    py::class_<PyOpBuilder> clss(m, "Builder", py::module_local(), py::dynamic_attr());
    bindInit(clss);
    bindLocations(clss);
    bindModifyInsertionPoint(clss);
    bindGetBasicType(clss);
    bindGetSpecialType(clss);
    bindGetAttributes(clss);
    bindCreateOperations(clss);
    bindCreateOperationsFunction(clss);
    bindCreateSignedConstants(clss);
    bindCreateUnsignedConstants(clss);
    bindCreateFloatConstants(clss);
    bindCreateAirthBasicOperations(clss);
    bindCreateAirthSpecialOperations(clss);
    bindCreateAirthComparisonOperations(clss);
    bindCreateAirthExtendedOperations(clss);
    bindCreateMathOperations(clss);
    bindCreateScfOperations(clss);
    bindCreateMemrefOperations(clss);
    bindCreateVectorOperations(clss);
    bindCreateEmitcOperations(clss);
    bindCreateEmitascOperations(clss);
    bindCreateAscPipeOperations(clss);
    bindCreateAscEventOperations(clss);
    bindCreateAscCommonOperations(clss);
    bindCreateAscTileOperations(clss);
    bindCreateTensorOperations(clss);
}

} // namespace asc
} // namespace pybind11
