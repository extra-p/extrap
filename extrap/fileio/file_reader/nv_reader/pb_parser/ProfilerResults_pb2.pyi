import ProfilerCommon_pb2 as _ProfilerCommon_pb2
import ProfilerCategories_pb2 as _ProfilerCategories_pb2
import ProfilerStringTable_pb2 as _ProfilerStringTable_pb2
import ProfilerReport_pb2 as _ProfilerReport_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProfilerResultsMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    InvalidResultsMethod: _ClassVar[ProfilerResultsMethod]
    MethodSourceMessage: _ClassVar[ProfilerResultsMethod]
    MethodResultMessage: _ClassVar[ProfilerResultsMethod]
    MethodClearCachedProfilerSourceMessage: _ClassVar[ProfilerResultsMethod]
InvalidResultsMethod: ProfilerResultsMethod
MethodSourceMessage: ProfilerResultsMethod
MethodResultMessage: ProfilerResultsMethod
MethodClearCachedProfilerSourceMessage: ProfilerResultsMethod

class MetricValueMessage(_message.Message):
    __slots__ = ("StringValue", "FloatValue", "DoubleValue", "Uint32Value", "Uint64Value")
    STRINGVALUE_FIELD_NUMBER: _ClassVar[int]
    FLOATVALUE_FIELD_NUMBER: _ClassVar[int]
    DOUBLEVALUE_FIELD_NUMBER: _ClassVar[int]
    UINT32VALUE_FIELD_NUMBER: _ClassVar[int]
    UINT64VALUE_FIELD_NUMBER: _ClassVar[int]
    StringValue: str
    FloatValue: float
    DoubleValue: float
    Uint32Value: int
    Uint64Value: int
    def __init__(self, StringValue: _Optional[str] = ..., FloatValue: _Optional[float] = ..., DoubleValue: _Optional[float] = ..., Uint32Value: _Optional[int] = ..., Uint64Value: _Optional[int] = ...) -> None: ...

class MetricListElementMessage(_message.Message):
    __slots__ = ("CorrelationId", "ElementValue")
    CORRELATIONID_FIELD_NUMBER: _ClassVar[int]
    ELEMENTVALUE_FIELD_NUMBER: _ClassVar[int]
    CorrelationId: MetricValueMessage
    ElementValue: MetricValueMessage
    def __init__(self, CorrelationId: _Optional[_Union[MetricValueMessage, _Mapping]] = ..., ElementValue: _Optional[_Union[MetricValueMessage, _Mapping]] = ...) -> None: ...

class MetricResultMessage(_message.Message):
    __slots__ = ("MetricNameId", "MetricValue", "MetricValueList")
    METRICNAMEID_FIELD_NUMBER: _ClassVar[int]
    METRICVALUE_FIELD_NUMBER: _ClassVar[int]
    METRICVALUELIST_FIELD_NUMBER: _ClassVar[int]
    MetricNameId: int
    MetricValue: MetricValueMessage
    MetricValueList: _containers.RepeatedCompositeFieldContainer[MetricListElementMessage]
    def __init__(self, MetricNameId: _Optional[int] = ..., MetricValue: _Optional[_Union[MetricValueMessage, _Mapping]] = ..., MetricValueList: _Optional[_Iterable[_Union[MetricListElementMessage, _Mapping]]] = ...) -> None: ...

class ProfilerSourceMessage(_message.Message):
    __slots__ = ("Source",)
    class TypeInfo(_message.Message):
        __slots__ = ("Category", "Method")
        CATEGORY_FIELD_NUMBER: _ClassVar[int]
        METHOD_FIELD_NUMBER: _ClassVar[int]
        Category: _ProfilerCategories_pb2.ProfilerCategory
        Method: ProfilerResultsMethod
        def __init__(self, Category: _Optional[_Union[_ProfilerCategories_pb2.ProfilerCategory, str]] = ..., Method: _Optional[_Union[ProfilerResultsMethod, str]] = ...) -> None: ...
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    Source: _ProfilerCommon_pb2.SourceData
    def __init__(self, Source: _Optional[_Union[_ProfilerCommon_pb2.SourceData, _Mapping]] = ...) -> None: ...

class ClearCachedProfilerSourceMessage(_message.Message):
    __slots__ = ()
    class TypeInfo(_message.Message):
        __slots__ = ("Category", "Method")
        CATEGORY_FIELD_NUMBER: _ClassVar[int]
        METHOD_FIELD_NUMBER: _ClassVar[int]
        Category: _ProfilerCategories_pb2.ProfilerCategory
        Method: ProfilerResultsMethod
        def __init__(self, Category: _Optional[_Union[_ProfilerCategories_pb2.ProfilerCategory, str]] = ..., Method: _Optional[_Union[ProfilerResultsMethod, str]] = ...) -> None: ...
    def __init__(self) -> None: ...

class ProfilerVersionInfoMessage(_message.Message):
    __slots__ = ("Provider", "Version")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    Provider: str
    Version: str
    def __init__(self, Provider: _Optional[str] = ..., Version: _Optional[str] = ...) -> None: ...

class ProfilerResultMessage(_message.Message):
    __slots__ = ("ID", "Launch", "MetricResults", "Source", "StringTable", "UnsupportedDevice", "VersionInfo", "SeriesInfo")
    class TypeInfo(_message.Message):
        __slots__ = ("Category", "Method")
        CATEGORY_FIELD_NUMBER: _ClassVar[int]
        METHOD_FIELD_NUMBER: _ClassVar[int]
        Category: _ProfilerCategories_pb2.ProfilerCategory
        Method: ProfilerResultsMethod
        def __init__(self, Category: _Optional[_Union[_ProfilerCategories_pb2.ProfilerCategory, str]] = ..., Method: _Optional[_Union[ProfilerResultsMethod, str]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    LAUNCH_FIELD_NUMBER: _ClassVar[int]
    METRICRESULTS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    STRINGTABLE_FIELD_NUMBER: _ClassVar[int]
    UNSUPPORTEDDEVICE_FIELD_NUMBER: _ClassVar[int]
    VERSIONINFO_FIELD_NUMBER: _ClassVar[int]
    SERIESINFO_FIELD_NUMBER: _ClassVar[int]
    ID: int
    Launch: _ProfilerCommon_pb2.LaunchData
    MetricResults: _containers.RepeatedCompositeFieldContainer[MetricResultMessage]
    Source: _ProfilerCommon_pb2.SourceData
    StringTable: _ProfilerStringTable_pb2.ProfilerStringTable
    UnsupportedDevice: bool
    VersionInfo: _containers.RepeatedCompositeFieldContainer[ProfilerVersionInfoMessage]
    SeriesInfo: _ProfilerReport_pb2.ProfileSeriesInfoMessage
    def __init__(self, ID: _Optional[int] = ..., Launch: _Optional[_Union[_ProfilerCommon_pb2.LaunchData, _Mapping]] = ..., MetricResults: _Optional[_Iterable[_Union[MetricResultMessage, _Mapping]]] = ..., Source: _Optional[_Union[_ProfilerCommon_pb2.SourceData, _Mapping]] = ..., StringTable: _Optional[_Union[_ProfilerStringTable_pb2.ProfilerStringTable, _Mapping]] = ..., UnsupportedDevice: _Optional[bool] = ..., VersionInfo: _Optional[_Iterable[_Union[ProfilerVersionInfoMessage, _Mapping]]] = ..., SeriesInfo: _Optional[_Union[_ProfilerReport_pb2.ProfileSeriesInfoMessage, _Mapping]] = ...) -> None: ...
