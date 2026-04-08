import NvtxCategories_pb2 as _NvtxCategories_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NvtxMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NvtxMethodInvalid: _ClassVar[NvtxMethod]
    NvtxMethodRequestStateMessage: _ClassVar[NvtxMethod]
    NvtxMethodReplyStateMessage: _ClassVar[NvtxMethod]

class NvtxColorType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NvtxColorTypeUnknown: _ClassVar[NvtxColorType]
    NvtxColorTypeArgb: _ClassVar[NvtxColorType]

class NvtxPayloadType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NvtxPayloadTypeUnknown: _ClassVar[NvtxPayloadType]
    NvtxPayloadTypeUint64: _ClassVar[NvtxPayloadType]
    NvtxPayloadTypeInt64: _ClassVar[NvtxPayloadType]
    NvtxPayloadTypeDouble: _ClassVar[NvtxPayloadType]
    NvtxPayloadTypeUint32: _ClassVar[NvtxPayloadType]
    NvtxPayloadTypeInt32: _ClassVar[NvtxPayloadType]
    NvtxPayloadTypeFloat: _ClassVar[NvtxPayloadType]

class NvtxMessageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NvtxMessageTypeUnknown: _ClassVar[NvtxMessageType]
    NvtxMessageTypeAscii: _ClassVar[NvtxMessageType]
    NvtxMessageTypeUnicode: _ClassVar[NvtxMessageType]
    NvtxMessageTypeRegistered: _ClassVar[NvtxMessageType]

class NvtxNameFamily(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NvtxNameFamilyUnknown: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyCategory: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyOsThread: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyCudaDevice: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyCudaContext: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyCudaStream: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyCudaEvent: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyClDevice: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyClContext: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyClCommandQueue: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyClMemObject: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyClSampler: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyClProgram: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyClEvent: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyCudaRtDevice: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyCudaRtStream: _ClassVar[NvtxNameFamily]
    NvtxNameFamilyCudaRtEvent: _ClassVar[NvtxNameFamily]
NvtxMethodInvalid: NvtxMethod
NvtxMethodRequestStateMessage: NvtxMethod
NvtxMethodReplyStateMessage: NvtxMethod
NvtxColorTypeUnknown: NvtxColorType
NvtxColorTypeArgb: NvtxColorType
NvtxPayloadTypeUnknown: NvtxPayloadType
NvtxPayloadTypeUint64: NvtxPayloadType
NvtxPayloadTypeInt64: NvtxPayloadType
NvtxPayloadTypeDouble: NvtxPayloadType
NvtxPayloadTypeUint32: NvtxPayloadType
NvtxPayloadTypeInt32: NvtxPayloadType
NvtxPayloadTypeFloat: NvtxPayloadType
NvtxMessageTypeUnknown: NvtxMessageType
NvtxMessageTypeAscii: NvtxMessageType
NvtxMessageTypeUnicode: NvtxMessageType
NvtxMessageTypeRegistered: NvtxMessageType
NvtxNameFamilyUnknown: NvtxNameFamily
NvtxNameFamilyCategory: NvtxNameFamily
NvtxNameFamilyOsThread: NvtxNameFamily
NvtxNameFamilyCudaDevice: NvtxNameFamily
NvtxNameFamilyCudaContext: NvtxNameFamily
NvtxNameFamilyCudaStream: NvtxNameFamily
NvtxNameFamilyCudaEvent: NvtxNameFamily
NvtxNameFamilyClDevice: NvtxNameFamily
NvtxNameFamilyClContext: NvtxNameFamily
NvtxNameFamilyClCommandQueue: NvtxNameFamily
NvtxNameFamilyClMemObject: NvtxNameFamily
NvtxNameFamilyClSampler: NvtxNameFamily
NvtxNameFamilyClProgram: NvtxNameFamily
NvtxNameFamilyClEvent: NvtxNameFamily
NvtxNameFamilyCudaRtDevice: NvtxNameFamily
NvtxNameFamilyCudaRtStream: NvtxNameFamily
NvtxNameFamilyCudaRtEvent: NvtxNameFamily

class NvtxRequestStateMessage(_message.Message):
    __slots__ = ()
    class TypeInfo(_message.Message):
        __slots__ = ("Category", "Method")
        CATEGORY_FIELD_NUMBER: _ClassVar[int]
        METHOD_FIELD_NUMBER: _ClassVar[int]
        Category: _NvtxCategories_pb2.NvtxCategory
        Method: NvtxMethod
        def __init__(self, Category: _Optional[_Union[_NvtxCategories_pb2.NvtxCategory, str]] = ..., Method: _Optional[_Union[NvtxMethod, str]] = ...) -> None: ...
    def __init__(self) -> None: ...

class NvtxPayload(_message.Message):
    __slots__ = ("PayloadType", "ULLValue", "LLValue", "DoubleValue", "UValue", "IValue", "FValue")
    PAYLOADTYPE_FIELD_NUMBER: _ClassVar[int]
    ULLVALUE_FIELD_NUMBER: _ClassVar[int]
    LLVALUE_FIELD_NUMBER: _ClassVar[int]
    DOUBLEVALUE_FIELD_NUMBER: _ClassVar[int]
    UVALUE_FIELD_NUMBER: _ClassVar[int]
    IVALUE_FIELD_NUMBER: _ClassVar[int]
    FVALUE_FIELD_NUMBER: _ClassVar[int]
    PayloadType: NvtxPayloadType
    ULLValue: int
    LLValue: int
    DoubleValue: float
    UValue: int
    IValue: int
    FValue: float
    def __init__(self, PayloadType: _Optional[_Union[NvtxPayloadType, str]] = ..., ULLValue: _Optional[int] = ..., LLValue: _Optional[int] = ..., DoubleValue: _Optional[float] = ..., UValue: _Optional[int] = ..., IValue: _Optional[int] = ..., FValue: _Optional[float] = ...) -> None: ...

class NvtxMessage(_message.Message):
    __slots__ = ("MessageType", "message", "handle")
    MESSAGETYPE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    MessageType: NvtxMessageType
    message: str
    handle: int
    def __init__(self, MessageType: _Optional[_Union[NvtxMessageType, str]] = ..., message: _Optional[str] = ..., handle: _Optional[int] = ...) -> None: ...

class NvtxColor(_message.Message):
    __slots__ = ("ColorType", "Color")
    COLORTYPE_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    ColorType: NvtxColorType
    Color: int
    def __init__(self, ColorType: _Optional[_Union[NvtxColorType, str]] = ..., Color: _Optional[int] = ...) -> None: ...

class NvtxEventAttributes(_message.Message):
    __slots__ = ("Version", "Category", "Color", "Payload", "Message")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    Version: int
    Category: int
    Color: NvtxColor
    Payload: NvtxPayload
    Message: NvtxMessage
    def __init__(self, Version: _Optional[int] = ..., Category: _Optional[int] = ..., Color: _Optional[_Union[NvtxColor, _Mapping]] = ..., Payload: _Optional[_Union[NvtxPayload, _Mapping]] = ..., Message: _Optional[_Union[NvtxMessage, _Mapping]] = ...) -> None: ...

class NvtxPushPopRange(_message.Message):
    __slots__ = ("Name", "Attributes", "LastApiCallId")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    LASTAPICALLID_FIELD_NUMBER: _ClassVar[int]
    Name: str
    Attributes: NvtxEventAttributes
    LastApiCallId: int
    def __init__(self, Name: _Optional[str] = ..., Attributes: _Optional[_Union[NvtxEventAttributes, _Mapping]] = ..., LastApiCallId: _Optional[int] = ...) -> None: ...

class NvtxStartEndRange(_message.Message):
    __slots__ = ("Id", "Name", "Attributes", "LastApiCallId", "StartTID")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    LASTAPICALLID_FIELD_NUMBER: _ClassVar[int]
    STARTTID_FIELD_NUMBER: _ClassVar[int]
    Id: int
    Name: str
    Attributes: NvtxEventAttributes
    LastApiCallId: int
    StartTID: int
    def __init__(self, Id: _Optional[int] = ..., Name: _Optional[str] = ..., Attributes: _Optional[_Union[NvtxEventAttributes, _Mapping]] = ..., LastApiCallId: _Optional[int] = ..., StartTID: _Optional[int] = ...) -> None: ...

class NvtxPushPopDomain(_message.Message):
    __slots__ = ("Id", "Stack")
    ID_FIELD_NUMBER: _ClassVar[int]
    STACK_FIELD_NUMBER: _ClassVar[int]
    Id: int
    Stack: _containers.RepeatedCompositeFieldContainer[NvtxPushPopRange]
    def __init__(self, Id: _Optional[int] = ..., Stack: _Optional[_Iterable[_Union[NvtxPushPopRange, _Mapping]]] = ...) -> None: ...

class NvtxStartEndDomain(_message.Message):
    __slots__ = ("Id", "Ranges")
    ID_FIELD_NUMBER: _ClassVar[int]
    RANGES_FIELD_NUMBER: _ClassVar[int]
    Id: int
    Ranges: _containers.RepeatedCompositeFieldContainer[NvtxStartEndRange]
    def __init__(self, Id: _Optional[int] = ..., Ranges: _Optional[_Iterable[_Union[NvtxStartEndRange, _Mapping]]] = ...) -> None: ...

class NvtxRegisteredString(_message.Message):
    __slots__ = ("Id", "Value")
    ID_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    Id: int
    Value: str
    def __init__(self, Id: _Optional[int] = ..., Value: _Optional[str] = ...) -> None: ...

class NvtxDomainInfo(_message.Message):
    __slots__ = ("Id", "Name", "Strings", "NameTables")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    STRINGS_FIELD_NUMBER: _ClassVar[int]
    NAMETABLES_FIELD_NUMBER: _ClassVar[int]
    Id: int
    Name: str
    Strings: _containers.RepeatedCompositeFieldContainer[NvtxRegisteredString]
    NameTables: _containers.RepeatedCompositeFieldContainer[NvtxNameTable]
    def __init__(self, Id: _Optional[int] = ..., Name: _Optional[str] = ..., Strings: _Optional[_Iterable[_Union[NvtxRegisteredString, _Mapping]]] = ..., NameTables: _Optional[_Iterable[_Union[NvtxNameTable, _Mapping]]] = ...) -> None: ...

class NvtxThread(_message.Message):
    __slots__ = ("TID", "PushPopDomains")
    TID_FIELD_NUMBER: _ClassVar[int]
    PUSHPOPDOMAINS_FIELD_NUMBER: _ClassVar[int]
    TID: int
    PushPopDomains: _containers.RepeatedCompositeFieldContainer[NvtxPushPopDomain]
    def __init__(self, TID: _Optional[int] = ..., PushPopDomains: _Optional[_Iterable[_Union[NvtxPushPopDomain, _Mapping]]] = ...) -> None: ...

class NvtxNameTable(_message.Message):
    __slots__ = ("Family", "Mappings")
    FAMILY_FIELD_NUMBER: _ClassVar[int]
    MAPPINGS_FIELD_NUMBER: _ClassVar[int]
    Family: NvtxNameFamily
    Mappings: _containers.RepeatedCompositeFieldContainer[NvtxRegisteredString]
    def __init__(self, Family: _Optional[_Union[NvtxNameFamily, str]] = ..., Mappings: _Optional[_Iterable[_Union[NvtxRegisteredString, _Mapping]]] = ...) -> None: ...

class NvtxState(_message.Message):
    __slots__ = ("Domains", "Threads", "StartEndDomains", "DefaultDomain")
    DOMAINS_FIELD_NUMBER: _ClassVar[int]
    THREADS_FIELD_NUMBER: _ClassVar[int]
    STARTENDDOMAINS_FIELD_NUMBER: _ClassVar[int]
    DEFAULTDOMAIN_FIELD_NUMBER: _ClassVar[int]
    Domains: _containers.RepeatedCompositeFieldContainer[NvtxDomainInfo]
    Threads: _containers.RepeatedCompositeFieldContainer[NvtxThread]
    StartEndDomains: _containers.RepeatedCompositeFieldContainer[NvtxStartEndDomain]
    DefaultDomain: int
    def __init__(self, Domains: _Optional[_Iterable[_Union[NvtxDomainInfo, _Mapping]]] = ..., Threads: _Optional[_Iterable[_Union[NvtxThread, _Mapping]]] = ..., StartEndDomains: _Optional[_Iterable[_Union[NvtxStartEndDomain, _Mapping]]] = ..., DefaultDomain: _Optional[int] = ...) -> None: ...

class NvtxReplyStateMessage(_message.Message):
    __slots__ = ("State",)
    class TypeInfo(_message.Message):
        __slots__ = ("Category", "Method")
        CATEGORY_FIELD_NUMBER: _ClassVar[int]
        METHOD_FIELD_NUMBER: _ClassVar[int]
        Category: _NvtxCategories_pb2.NvtxCategory
        Method: NvtxMethod
        def __init__(self, Category: _Optional[_Union[_NvtxCategories_pb2.NvtxCategory, str]] = ..., Method: _Optional[_Union[NvtxMethod, str]] = ...) -> None: ...
    STATE_FIELD_NUMBER: _ClassVar[int]
    State: NvtxState
    def __init__(self, State: _Optional[_Union[NvtxState, _Mapping]] = ...) -> None: ...
