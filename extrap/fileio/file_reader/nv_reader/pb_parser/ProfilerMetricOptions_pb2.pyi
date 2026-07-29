from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GpuArch(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    InvalidArch: _ClassVar[GpuArch]
    GK10x: _ClassVar[GpuArch]
    GK11x: _ClassVar[GpuArch]
    GK20xGM10x: _ClassVar[GpuArch]
    GM20xGM10x: _ClassVar[GpuArch]
    GM20b: _ClassVar[GpuArch]
    GP100: _ClassVar[GpuArch]
    GP10x: _ClassVar[GpuArch]
    GP10b: _ClassVar[GpuArch]
    GV100: _ClassVar[GpuArch]
    GV11b: _ClassVar[GpuArch]
    TU10x: _ClassVar[GpuArch]
InvalidArch: GpuArch
GK10x: GpuArch
GK11x: GpuArch
GK20xGM10x: GpuArch
GM20xGM10x: GpuArch
GM20b: GpuArch
GP100: GpuArch
GP10x: GpuArch
GP10b: GpuArch
GV100: GpuArch
GV11b: GpuArch
TU10x: GpuArch

class MetricOptionFilter(_message.Message):
    __slots__ = ("MinArch", "MaxArch")
    MINARCH_FIELD_NUMBER: _ClassVar[int]
    MAXARCH_FIELD_NUMBER: _ClassVar[int]
    MinArch: GpuArch
    MaxArch: GpuArch
    def __init__(self, MinArch: _Optional[_Union[GpuArch, str]] = ..., MaxArch: _Optional[_Union[GpuArch, str]] = ...) -> None: ...
