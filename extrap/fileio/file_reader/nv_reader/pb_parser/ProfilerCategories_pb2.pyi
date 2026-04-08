from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class ProfilerCategory(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CategoryInvalid: _ClassVar[ProfilerCategory]
    CategoryConfiguration: _ClassVar[ProfilerCategory]
    CategoryControl: _ClassVar[ProfilerCategory]
    CategoryResults: _ClassVar[ProfilerCategory]
    CategoryStatus: _ClassVar[ProfilerCategory]
CategoryInvalid: ProfilerCategory
CategoryConfiguration: ProfilerCategory
CategoryControl: ProfilerCategory
CategoryResults: ProfilerCategory
CategoryStatus: ProfilerCategory
