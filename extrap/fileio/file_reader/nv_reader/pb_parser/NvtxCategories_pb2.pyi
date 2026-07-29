from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class NvtxCategory(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NvtxCategoryInvalid: _ClassVar[NvtxCategory]
    NvtxCategoryState: _ClassVar[NvtxCategory]
NvtxCategoryInvalid: NvtxCategory
NvtxCategoryState: NvtxCategory
