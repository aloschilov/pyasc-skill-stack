/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * */

#ifndef LIB_CONVERSION_LOWERTOASC_COMMON_H
#define LIB_CONVERSION_LOWERTOASC_COMMON_H

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>

namespace mlir {
namespace asclower {

struct I1ReplacementType {
    using Int = int8_t;
    using UInt = uint8_t;

    static inline constexpr unsigned width = 8U;
    IntegerType iType;
    IntegerType uiType;

    explicit I1ReplacementType(MLIRContext* context)
        : iType(IntegerType::get(context, width, IntegerType::Signless)),
          uiType(IntegerType::get(context, width, IntegerType::Unsigned))
    {}
    ~I1ReplacementType() = default;

    static constexpr UInt max()
    {
        constexpr UInt base = 1U;
        unsigned shift = width - 1U;
        return (base << shift) | ~(base << shift);
    }
};

struct LoweringTypeConverter : public TypeConverter {
    static Value addUnrealizedCast(OpBuilder& builder, Type type, ValueRange inputs, Location loc)
    {
        auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
        return cast.getResult(0);
    }

    LoweringTypeConverter()
    {
        addConversion([](Type type) { return std::optional<Type>{type}; });
        addArgumentMaterialization(addUnrealizedCast);
        addSourceMaterialization(addUnrealizedCast);
        addTargetMaterialization(addUnrealizedCast);
    }
};

struct TensorTypeConverter : public LoweringTypeConverter {
    TensorTypeConverter() : LoweringTypeConverter()
    {
        addConversion([](asctile::LocalTensorType type) {
            auto elType = type.getElementType();
            SmallVector<int64_t> shape(type.getShape());
            if (elType.isInteger(1)) {
                I1ReplacementType replType(type.getContext());
                auto numElements = static_cast<int64_t>(llvm::divideCeil(type.getNumElements(), replType.width));
                shape = {numElements};
                elType = replType.iType;
            }
            if (type.getLoc() == asctile::TensorLocation::BT && (elType.isF16() || elType.isBF16()))
                elType = FloatType::getF32(type.getContext());
            return ascendc::LocalTensorType::get(shape, elType);
        });
        addConversion([](asctile::GlobalTensorType type) {
            return ascendc::GlobalTensorType::get(type.getShape(), type.getElementType());
        });
        addConversion([](MemRefType type) -> Type {
            auto addrSpace = ascendc::AddressSpace::gm;
            if (auto attr = dyn_cast_if_present<IntegerAttr>(type.getMemorySpace()))
                addrSpace = static_cast<ascendc::AddressSpace>(attr.getValue().getSExtValue());
            if (addrSpace == ascendc::AddressSpace::Default)
                addrSpace = ascendc::AddressSpace::gm;
            if (addrSpace == ascendc::AddressSpace::gm)
                return ascendc::GlobalTensorType::get(type.getShape(), type.getElementType());
            return ascendc::LocalTensorType::get(type.getShape(), type.getElementType());
        });
    }
};

using ConvertRewriter = ConversionPatternRewriter;

template <typename OpType>
struct ConvertOp : public OpConversionPattern<OpType> {
    using OpAdaptor = typename OpType::Adaptor;

    using OpConversionPattern<OpType>::OpConversionPattern;
    using OpConversionPattern<OpType>::typeConverter;

    virtual void rewrite(OpType op, ConvertRewriter& rewriter) const
    {
        llvm_unreachable("either rewrite() or matchAndRewrite() must be overloaded");
    }

    void rewrite(OpType op, [[maybe_unused]] OpAdaptor adaptor, ConvertRewriter& rewriter) const final
    {
        rewrite(op, rewriter);
    }

    virtual LogicalResult matchAndRewrite(OpType op, ConvertRewriter& rewriter) const
    {
        if (this->match(op).failed())
            return failure();
        rewrite(op, rewriter);
        return success();
    }

    LogicalResult matchAndRewrite(OpType op, [[maybe_unused]] OpAdaptor adaptor, ConvertRewriter& rewriter) const final
    {
        return matchAndRewrite(op, rewriter);
    }

    static ascendc::TPosition locationToPosition(asctile::TensorLocation loc)
    {
        switch (loc) {
            case asctile::TensorLocation::L1:
                return ascendc::TPosition::A1;
            case asctile::TensorLocation::L0A:
                return ascendc::TPosition::A2;
            case asctile::TensorLocation::L0B:
                return ascendc::TPosition::B2;
            case asctile::TensorLocation::FIX:
                [[fallthrough]];
            case asctile::TensorLocation::L0C:
                return ascendc::TPosition::CO1;
            case asctile::TensorLocation::UB:
                return ascendc::TPosition::VECCALC;
            case asctile::TensorLocation::BT:
                return ascendc::TPosition::C2;
            default:
                llvm_unreachable("unexpected TensorLocation value");
        }
    }

    static ascendc::LocalTensorAutoOp createTensorOp(
        OpBuilder& builder, Location loc, ArrayRef<int64_t> shape, Type elementType,
        std::optional<ascendc::TPosition> position = std::nullopt)
    {
        auto localTensorAuto = builder.create<ascendc::LocalTensorAutoOp>(
            loc, ascendc::LocalTensorType::get(shape, elementType), position.value_or(ascendc::TPosition::VECCALC));
        return localTensorAuto;
    }

    ascendc::LocalTensorAutoOp createTensorOp(
        OpBuilder& builder, Location loc, Type convertibleType,
        std::optional<ascendc::TPosition> position = std::nullopt) const
    {
        auto convertedType = typeConverter->convertType(convertibleType);
        assert(isa<ascendc::LocalTensorType>(convertedType) && "must be convertible");
        auto tensorType = cast<ascendc::LocalTensorType>(convertedType);
        if (auto tileType = dyn_cast<asctile::LocalTensorType>(convertibleType)) {
            position = position.has_value() ? position : locationToPosition(tileType.getLoc());
        }
        return createTensorOp(builder, loc, tensorType.getShape(), tensorType.getElementType(), position);
    }

    ascendc::LocalTensorReinterpretCastOp createReCastOp(
        OpBuilder& builder, Location loc, Value convertibleTensor, ArrayRef<int64_t> shape, Type elementType) const
    {
        auto type = ascendc::LocalTensorType::get(shape, elementType);
        auto tensor = typeConverter->materializeTargetConversion(builder, loc, type, convertibleTensor);
        return builder.create<ascendc::LocalTensorReinterpretCastOp>(loc, type, tensor);
    }

    ascendc::LocalTensorReinterpretCastOp createReCastOp(
        OpBuilder& builder, Location loc, Value convertibleTensor, Type convertibleType) const
    {
        auto convertedType = typeConverter->convertType(convertibleType);
        assert(isa<ascendc::LocalTensorType>(convertedType) && "must be convertible");
        auto tensorType = cast<ascendc::LocalTensorType>(convertedType);
        return createReCastOp(builder, loc, convertibleTensor, tensorType.getShape(), tensorType.getElementType());
    }

    static int64_t calCount(Value tensor) { return calCount(tensor.getType()); }

    static int64_t calCount(Type type)
    {
        auto shaped = dyn_cast<ShapedType>(type);
        assert(shaped && shaped.hasStaticShape() && "must be ShapedType with static shape");
        return shaped.getNumElements();
    }
};

} // namespace asclower
} // namespace mlir

#endif // LIB_CONVERSION_LOWERTOASC_COMMON_H
