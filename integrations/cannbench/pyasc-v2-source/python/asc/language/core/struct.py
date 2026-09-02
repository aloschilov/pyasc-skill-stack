# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import abc
import ctypes
from typing import Any, ClassVar, Dict, Optional, Type, Union, overload
from typing_extensions import Self

from ..._C import ir
from .constexpr import Numeric
from .dtype import DataType
from .ir_value import IRHandle, IRValue, PlainValue, RuntimeNumeric, materialize_ir_value as _mat
from .utils import global_builder, require_jit

dtype_ctype: Dict[str, Type[ctypes._SimpleCData]] = {
    "int1": ctypes.c_bool,
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint8": ctypes.c_uint8,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
}

missing = object()


class BaseField(abc.ABC):

    def __init__(self, default: Optional[Any] = missing, name: Optional[str] = None) -> None:
        self.default = default
        self.name = name

    @abc.abstractmethod
    def ctypes_cls(self) -> Type[ctypes._CData]:
        raise NotImplementedError

    @abc.abstractmethod
    def ctypes_value(self, value: Union[Numeric, Struct]) -> ctypes._CData:
        raise NotImplementedError

    @abc.abstractmethod
    def from_ir(self, handle: IRHandle) -> IRValue:
        raise NotImplementedError

    def ir_name(self, default: str) -> str:
        return default if self.name is None else self.name

    @abc.abstractmethod
    def ir_type(self, setter: bool = False) -> ir.Type:
        raise NotImplementedError

    @abc.abstractmethod
    def to_ir(self, value: Union[RuntimeNumeric, Struct]) -> IRHandle:
        raise NotImplementedError


class Field(BaseField):

    def __init__(self, dtype: DataType, default: Optional[Any] = missing, name: Optional[str] = None,
                 enum: Optional[str] = None) -> None:
        super().__init__(default, name)
        self.dtype = dtype
        self.enum_name = enum

    def ctypes_cls(self) -> Type[ctypes._CData]:
        return dtype_ctype[self.dtype.name]

    def ctypes_value(self, value: Numeric) -> ctypes._SimpleCData:
        return value

    def from_ir(self, handle: IRHandle) -> PlainValue:
        return PlainValue(handle)

    def ir_type(self, setter: bool = False) -> ir.Type:
        if setter and self.enum_name is not None:
            return global_builder.get_ir_builder().get_opaque_type(self.enum_name)
        return self.dtype.to_ir()

    def to_ir(self, value: RuntimeNumeric) -> IRHandle:
        return _mat(value, self.dtype).to_ir()


class StructField(BaseField):

    def __init__(self, struct_cls: Type[Struct], default: Optional[Any] = None, name: Optional[str] = None) -> None:
        if default is None:
            default = struct_cls()
        super().__init__(default, name)
        self.struct_cls = struct_cls

    def ctypes_cls(self) -> Type[ctypes._CData]:
        return self.struct_cls.ctypes_class

    def ctypes_value(self, value: Struct) -> ctypes.Structure:
        return value.ctypes_struct

    def from_ir(self, handle: IRHandle) -> IRValue:
        return self.struct_cls.from_ir(handle)

    def ir_type(self, setter: bool = False) -> ir.Type:
        return self.struct_cls.get_ir_type()

    def to_ir(self, value: Struct) -> IRHandle:
        return value.to_ir()


class Struct(IRValue):
    __slots__ = ("ctypes_struct", "handle")

    ctypes_struct: ctypes.Structure
    ctypes_class: ClassVar[Type[ctypes.Structure]]
    struct_fields: ClassVar[Dict[str, BaseField]]

    @overload
    def __init__(self, **fields: Numeric) -> None:
        """This constructor should be used to create Struct instance on host"""
        ...

    @overload
    def __init__(self, **fields: RuntimeNumeric) -> None:
        """This constructor should be used to create Struct instance in JIT function"""
        ...

    @overload
    def __init__(self, *, handle: IRHandle) -> None:
        """This constructor should not be called by user"""
        ...

    def __init__(self, *, handle: Optional[IRHandle] = None, **fields):
        if handle is not None:
            self.handle = handle
            return
        if global_builder.get_ir_builder() is not None:
            self.__initjit__(**fields)
            return
        set_fields: Dict[str, Numeric] = {}
        unset_fields = set(self.struct_fields)
        for name, value in fields.items():
            if name not in unset_fields:
                raise RuntimeError(f"{self.__class__.__name__} does not have a field '{name}'")
            set_fields[name] = value
            unset_fields.discard(name)
        for name in unset_fields:
            default = self.struct_fields[name].default
            if default is missing:
                raise RuntimeError(f"{self.__class__.__name__} does not have a default value for '{name}'")
            set_fields[name] = default
        if not issubclass(self.ctypes_class, ctypes.Structure):
            raise RuntimeError(f"ctypes.Structure class must be generated in {self.__class__.__name__}")
        ctypes_args = {name: self.get_field(name).ctypes_value(value) for name, value in set_fields.items()}
        self.ctypes_struct = self.ctypes_class(**ctypes_args)

    def __init_subclass__(cls: Type[Struct], pack: int = 8, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.struct_fields = {name: value for name, value in vars(cls).items() if isinstance(value, BaseField)}
        if len(cls.struct_fields) == 0:
            raise RuntimeError(f"Struct '{cls.__name__}' must have at least one field")
        ctypes_fields = [(name, field.ctypes_cls()) for name, field in cls.struct_fields.items()]
        attrs = {"_fields_": ctypes_fields, "_pack_": pack}
        cls.ctypes_class = type(f"_ctypes_{cls.__name__}", (ctypes.Structure, ), attrs)

    def __getattribute__(self, name: str) -> Numeric:
        attr = super().__getattribute__(name)
        if isinstance(attr, BaseField):
            attr = getattr(self.ctypes_struct, name)
        return attr

    def __setattr__(self, name: str, value: Numeric) -> None:
        if name in self.struct_fields:
            setattr(self.ctypes_struct, name, self.get_field(name).ctypes_value(value))
        else:
            super().__setattr__(name, value)

    @require_jit
    def __initjit__(self, **fields) -> None:
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(self.get_ir_type(), [])
        for name, value in fields.items():
            self.__setattrjit__(name, value)

    @require_jit
    def __getattrjit__(self, name: str) -> PlainValue:
        field = self.get_field(name)
        ir_name = field.ir_name(name)
        handle = global_builder.get_ir_builder().create_emitasc_MemberOp(self.to_ir(), ir_name, field.ir_type())
        return field.from_ir(handle)

    @require_jit
    def __setattrjit__(self, name: str, value: RuntimeNumeric) -> None:
        field = self.get_field(name)
        ir_name = field.ir_name(name)
        ir_value = field.to_ir(value)
        ir_type = field.ir_type(setter=True)
        builder = global_builder.get_ir_builder()
        if ir_type != ir_value.get_type():
            ir_value = builder.create_asc_ConstructOp(ir_type, [ir_value])
        builder.create_emitasc_SetMemberOp(self.to_ir(), ir_name, ir_value)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Self:
        return cls(handle=handle)

    @classmethod
    def get_field(cls, name: str) -> BaseField:
        field = cls.struct_fields.get(name)
        if field is None:
            raise RuntimeError(f"{cls.__name__} does not have a field '{name}'")
        return field

    @classmethod
    def get_ir_type(cls) -> ir.Type:
        types = []
        names = []
        for name, field in cls.struct_fields.items():
            names.append(name)
            types.append(field.ir_type())
        return global_builder.get_ir_builder().get_emitasc_PyStructType(cls.__name__, types, names)

    def to_ir(self) -> IRHandle:
        return self.handle

    def addressof(self) -> int:
        return int(ctypes.addressof(self.ctypes_struct))

    def pack(self) -> bytes:
        if not isinstance(self.ctypes_struct, ctypes.Structure):
            raise RuntimeError(f"ctypes.Structure instance must be generated in {self.__class__.__name__}")
        return ctypes.string_at(ctypes.addressof(self.ctypes_struct), ctypes.sizeof(self.ctypes_struct))

    def create_local(self) -> Self:
        builder = global_builder.get_ir_builder()
        handle = builder.create_emitasc_CopyStructOp(self.to_ir())
        return self.from_ir(handle)
