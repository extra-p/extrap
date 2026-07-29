from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class APIType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    APIType_CUDA: _ClassVar[APIType]
    APIType_OpenCL: _ClassVar[APIType]
    APIType_Direct3D: _ClassVar[APIType]
    APIType_OpenGL: _ClassVar[APIType]

class SourceSassLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SourceSassLevel_Unset: _ClassVar[SourceSassLevel]
    SourceSassLevel_Sass1: _ClassVar[SourceSassLevel]
    SourceSassLevel_Sass2: _ClassVar[SourceSassLevel]
    SourceSassLevel_Sass3: _ClassVar[SourceSassLevel]
    SourceSassLevel_Sass4: _ClassVar[SourceSassLevel]
    SourceSassLevel_Sass5: _ClassVar[SourceSassLevel]
    SourceSassLevel_Sass6: _ClassVar[SourceSassLevel]
    SourceSassLevel_Sass7: _ClassVar[SourceSassLevel]
APIType_CUDA: APIType
APIType_OpenCL: APIType
APIType_Direct3D: APIType
APIType_OpenGL: APIType
SourceSassLevel_Unset: SourceSassLevel
SourceSassLevel_Sass1: SourceSassLevel
SourceSassLevel_Sass2: SourceSassLevel
SourceSassLevel_Sass3: SourceSassLevel
SourceSassLevel_Sass4: SourceSassLevel
SourceSassLevel_Sass5: SourceSassLevel
SourceSassLevel_Sass6: SourceSassLevel
SourceSassLevel_Sass7: SourceSassLevel

class Uint64x3(_message.Message):
    __slots__ = ("X", "Y", "Z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    X: int
    Y: int
    Z: int
    def __init__(self, X: _Optional[int] = ..., Y: _Optional[int] = ..., Z: _Optional[int] = ...) -> None: ...

class LaunchData(_message.Message):
    __slots__ = ("ProcessID", "ThreadID", "APICallID", "API", "KernelID", "KernelMangledName", "KernelFunctionName", "KernelDemangledName", "ProgramHandle", "ContextHandle", "DeviceID", "WorkDimensions", "GlobalWorkOffset", "GlobalWorkSize", "LocalWorkSize", "ContextID", "StreamID")
    PROCESSID_FIELD_NUMBER: _ClassVar[int]
    THREADID_FIELD_NUMBER: _ClassVar[int]
    APICALLID_FIELD_NUMBER: _ClassVar[int]
    API_FIELD_NUMBER: _ClassVar[int]
    KERNELID_FIELD_NUMBER: _ClassVar[int]
    KERNELMANGLEDNAME_FIELD_NUMBER: _ClassVar[int]
    KERNELFUNCTIONNAME_FIELD_NUMBER: _ClassVar[int]
    KERNELDEMANGLEDNAME_FIELD_NUMBER: _ClassVar[int]
    PROGRAMHANDLE_FIELD_NUMBER: _ClassVar[int]
    CONTEXTHANDLE_FIELD_NUMBER: _ClassVar[int]
    DEVICEID_FIELD_NUMBER: _ClassVar[int]
    WORKDIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    GLOBALWORKOFFSET_FIELD_NUMBER: _ClassVar[int]
    GLOBALWORKSIZE_FIELD_NUMBER: _ClassVar[int]
    LOCALWORKSIZE_FIELD_NUMBER: _ClassVar[int]
    CONTEXTID_FIELD_NUMBER: _ClassVar[int]
    STREAMID_FIELD_NUMBER: _ClassVar[int]
    ProcessID: int
    ThreadID: int
    APICallID: int
    API: APIType
    KernelID: int
    KernelMangledName: str
    KernelFunctionName: str
    KernelDemangledName: str
    ProgramHandle: int
    ContextHandle: int
    DeviceID: int
    WorkDimensions: int
    GlobalWorkOffset: Uint64x3
    GlobalWorkSize: Uint64x3
    LocalWorkSize: Uint64x3
    ContextID: int
    StreamID: int
    def __init__(self, ProcessID: _Optional[int] = ..., ThreadID: _Optional[int] = ..., APICallID: _Optional[int] = ..., API: _Optional[_Union[APIType, str]] = ..., KernelID: _Optional[int] = ..., KernelMangledName: _Optional[str] = ..., KernelFunctionName: _Optional[str] = ..., KernelDemangledName: _Optional[str] = ..., ProgramHandle: _Optional[int] = ..., ContextHandle: _Optional[int] = ..., DeviceID: _Optional[int] = ..., WorkDimensions: _Optional[int] = ..., GlobalWorkOffset: _Optional[_Union[Uint64x3, _Mapping]] = ..., GlobalWorkSize: _Optional[_Union[Uint64x3, _Mapping]] = ..., LocalWorkSize: _Optional[_Union[Uint64x3, _Mapping]] = ..., ContextID: _Optional[int] = ..., StreamID: _Optional[int] = ...) -> None: ...

class SourceData(_message.Message):
    __slots__ = ("Reference", "Code", "Intermediate", "Binary", "SassLevel", "SMRevision", "BinaryFlags")
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    INTERMEDIATE_FIELD_NUMBER: _ClassVar[int]
    BINARY_FIELD_NUMBER: _ClassVar[int]
    SASSLEVEL_FIELD_NUMBER: _ClassVar[int]
    SMREVISION_FIELD_NUMBER: _ClassVar[int]
    BINARYFLAGS_FIELD_NUMBER: _ClassVar[int]
    Reference: int
    Code: str
    Intermediate: bytes
    Binary: bytes
    SassLevel: SourceSassLevel
    SMRevision: int
    BinaryFlags: int
    def __init__(self, Reference: _Optional[int] = ..., Code: _Optional[str] = ..., Intermediate: _Optional[bytes] = ..., Binary: _Optional[bytes] = ..., SassLevel: _Optional[_Union[SourceSassLevel, str]] = ..., SMRevision: _Optional[int] = ..., BinaryFlags: _Optional[int] = ...) -> None: ...
