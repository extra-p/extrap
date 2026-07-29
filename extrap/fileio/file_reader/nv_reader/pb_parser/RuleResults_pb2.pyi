import ProfilerSection_pb2 as _ProfilerSection_pb2
import ProfilerCommon_pb2 as _ProfilerCommon_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RuleResultMessageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    None_: _ClassVar[RuleResultMessageType]
    Ok: _ClassVar[RuleResultMessageType]
    Warning: _ClassVar[RuleResultMessageType]
    Error: _ClassVar[RuleResultMessageType]
None_: RuleResultMessageType
Ok: RuleResultMessageType
Warning: RuleResultMessageType
Error: RuleResultMessageType

class RuleResultMessage(_message.Message):
    __slots__ = ("Message", "Type")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    Message: str
    Type: RuleResultMessageType
    def __init__(self, Message: _Optional[str] = ..., Type: _Optional[_Union[RuleResultMessageType, str]] = ...) -> None: ...

class RuleResultProposal(_message.Message):
    __slots__ = ("Identifier",)
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    Identifier: str
    def __init__(self, Identifier: _Optional[str] = ...) -> None: ...

class RuleResultBodyItem(_message.Message):
    __slots__ = ("Message", "Table", "BarChart", "HistogramChart", "LineChart", "Proposal")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TABLE_FIELD_NUMBER: _ClassVar[int]
    BARCHART_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAMCHART_FIELD_NUMBER: _ClassVar[int]
    LINECHART_FIELD_NUMBER: _ClassVar[int]
    PROPOSAL_FIELD_NUMBER: _ClassVar[int]
    Message: RuleResultMessage
    Table: _ProfilerSection_pb2.ProfilerSectionTable
    BarChart: _ProfilerSection_pb2.ProfilerSectionBarChart
    HistogramChart: _ProfilerSection_pb2.ProfilerSectionHistogramChart
    LineChart: _ProfilerSection_pb2.ProfilerSectionLineChart
    Proposal: RuleResultProposal
    def __init__(self, Message: _Optional[_Union[RuleResultMessage, _Mapping]] = ..., Table: _Optional[_Union[_ProfilerSection_pb2.ProfilerSectionTable, _Mapping]] = ..., BarChart: _Optional[_Union[_ProfilerSection_pb2.ProfilerSectionBarChart, _Mapping]] = ..., HistogramChart: _Optional[_Union[_ProfilerSection_pb2.ProfilerSectionHistogramChart, _Mapping]] = ..., LineChart: _Optional[_Union[_ProfilerSection_pb2.ProfilerSectionLineChart, _Mapping]] = ..., Proposal: _Optional[_Union[RuleResultProposal, _Mapping]] = ...) -> None: ...

class RuleResultBody(_message.Message):
    __slots__ = ("Items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    Items: _containers.RepeatedCompositeFieldContainer[RuleResultBodyItem]
    def __init__(self, Items: _Optional[_Iterable[_Union[RuleResultBodyItem, _Mapping]]] = ...) -> None: ...

class RuleResult(_message.Message):
    __slots__ = ("Identifier", "DisplayName", "Body", "SectionIdentifier")
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    DISPLAYNAME_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    SECTIONIDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    Identifier: str
    DisplayName: str
    Body: RuleResultBody
    SectionIdentifier: str
    def __init__(self, Identifier: _Optional[str] = ..., DisplayName: _Optional[str] = ..., Body: _Optional[_Union[RuleResultBody, _Mapping]] = ..., SectionIdentifier: _Optional[str] = ...) -> None: ...

class RuleResults(_message.Message):
    __slots__ = ("RuleResults",)
    RULERESULTS_FIELD_NUMBER: _ClassVar[int]
    RuleResults: _containers.RepeatedCompositeFieldContainer[RuleResult]
    def __init__(self, RuleResults: _Optional[_Iterable[_Union[RuleResult, _Mapping]]] = ...) -> None: ...
