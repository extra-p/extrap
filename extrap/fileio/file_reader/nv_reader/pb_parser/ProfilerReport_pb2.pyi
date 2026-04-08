import Nvtx_pb2 as _Nvtx_pb2
import ProfilerStringTable_pb2 as _ProfilerStringTable_pb2
import ProfilerCommon_pb2 as _ProfilerCommon_pb2
import ProfilerReportCommon_pb2 as _ProfilerReportCommon_pb2
import ProfilerSection_pb2 as _ProfilerSection_pb2
import RuleResults_pb2 as _RuleResults_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PlatformType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    Unknown: _ClassVar[PlatformType]
    Windows: _ClassVar[PlatformType]
    Linux: _ClassVar[PlatformType]
    Android: _ClassVar[PlatformType]
    OSX: _ClassVar[PlatformType]
    QNX: _ClassVar[PlatformType]
    Hos: _ClassVar[PlatformType]
Unknown: PlatformType
Windows: PlatformType
Linux: PlatformType
Android: PlatformType
OSX: PlatformType
QNX: PlatformType
Hos: PlatformType

class DeviceProperty(_message.Message):
    __slots__ = ("Key", "Value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    Key: str
    Value: str
    def __init__(self, Key: _Optional[str] = ..., Value: _Optional[str] = ...) -> None: ...

class DeviceProperties(_message.Message):
    __slots__ = ("Properties",)
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    Properties: _containers.RepeatedCompositeFieldContainer[DeviceProperty]
    def __init__(self, Properties: _Optional[_Iterable[_Union[DeviceProperty, _Mapping]]] = ...) -> None: ...

class DeviceAttribute(_message.Message):
    __slots__ = ("Name", "Value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    Name: str
    Value: ProfileMetricValue
    def __init__(self, Name: _Optional[str] = ..., Value: _Optional[_Union[ProfileMetricValue, _Mapping]] = ...) -> None: ...

class DeviceAttributes(_message.Message):
    __slots__ = ("ID", "Name", "Attributes")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    ID: int
    Name: str
    Attributes: _containers.RepeatedCompositeFieldContainer[DeviceAttribute]
    def __init__(self, ID: _Optional[int] = ..., Name: _Optional[str] = ..., Attributes: _Optional[_Iterable[_Union[DeviceAttribute, _Mapping]]] = ...) -> None: ...

class SystemInfo(_message.Message):
    __slots__ = ("OSName", "Build", "Processor", "Architecture", "ComputerName", "Platform")
    OSNAME_FIELD_NUMBER: _ClassVar[int]
    BUILD_FIELD_NUMBER: _ClassVar[int]
    PROCESSOR_FIELD_NUMBER: _ClassVar[int]
    ARCHITECTURE_FIELD_NUMBER: _ClassVar[int]
    COMPUTERNAME_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    OSName: str
    Build: str
    Processor: str
    Architecture: str
    ComputerName: str
    Platform: PlatformType
    def __init__(self, OSName: _Optional[str] = ..., Build: _Optional[str] = ..., Processor: _Optional[str] = ..., Architecture: _Optional[str] = ..., ComputerName: _Optional[str] = ..., Platform: _Optional[_Union[PlatformType, str]] = ...) -> None: ...

class VersionInfo(_message.Message):
    __slots__ = ("Provider", "Version")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    Provider: str
    Version: str
    def __init__(self, Provider: _Optional[str] = ..., Version: _Optional[str] = ...) -> None: ...

class FilterOptions(_message.Message):
    __slots__ = ("KernelRegex", "KernelRegexBase", "KernelId", "SkipCount", "SkipBeforeMatchCount", "CaptureCount")
    KERNELREGEX_FIELD_NUMBER: _ClassVar[int]
    KERNELREGEXBASE_FIELD_NUMBER: _ClassVar[int]
    KERNELID_FIELD_NUMBER: _ClassVar[int]
    SKIPCOUNT_FIELD_NUMBER: _ClassVar[int]
    SKIPBEFOREMATCHCOUNT_FIELD_NUMBER: _ClassVar[int]
    CAPTURECOUNT_FIELD_NUMBER: _ClassVar[int]
    KernelRegex: str
    KernelRegexBase: str
    KernelId: str
    SkipCount: int
    SkipBeforeMatchCount: int
    CaptureCount: int
    def __init__(self, KernelRegex: _Optional[str] = ..., KernelRegexBase: _Optional[str] = ..., KernelId: _Optional[str] = ..., SkipCount: _Optional[int] = ..., SkipBeforeMatchCount: _Optional[int] = ..., CaptureCount: _Optional[int] = ...) -> None: ...

class SamplingOptions(_message.Message):
    __slots__ = ("IntervalAuto", "Interval", "MaxPasses", "BufferSize")
    INTERVALAUTO_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_FIELD_NUMBER: _ClassVar[int]
    MAXPASSES_FIELD_NUMBER: _ClassVar[int]
    BUFFERSIZE_FIELD_NUMBER: _ClassVar[int]
    IntervalAuto: bool
    Interval: int
    MaxPasses: int
    BufferSize: int
    def __init__(self, IntervalAuto: _Optional[bool] = ..., Interval: _Optional[int] = ..., MaxPasses: _Optional[int] = ..., BufferSize: _Optional[int] = ...) -> None: ...

class OtherOptions(_message.Message):
    __slots__ = ("ApplyRules", "Metrics")
    APPLYRULES_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    ApplyRules: bool
    Metrics: str
    def __init__(self, ApplyRules: _Optional[bool] = ..., Metrics: _Optional[str] = ...) -> None: ...

class ProfilerSettings(_message.Message):
    __slots__ = ("EnableNvtx", "DisableProfilingStartStop", "EnableProfilingFromStart", "ActivityType", "FilterOptions", "OtherOptions", "EnabledSections", "ClockControlMode", "SamplingOptions")
    ENABLENVTX_FIELD_NUMBER: _ClassVar[int]
    DISABLEPROFILINGSTARTSTOP_FIELD_NUMBER: _ClassVar[int]
    ENABLEPROFILINGFROMSTART_FIELD_NUMBER: _ClassVar[int]
    ACTIVITYTYPE_FIELD_NUMBER: _ClassVar[int]
    FILTEROPTIONS_FIELD_NUMBER: _ClassVar[int]
    OTHEROPTIONS_FIELD_NUMBER: _ClassVar[int]
    ENABLEDSECTIONS_FIELD_NUMBER: _ClassVar[int]
    CLOCKCONTROLMODE_FIELD_NUMBER: _ClassVar[int]
    SAMPLINGOPTIONS_FIELD_NUMBER: _ClassVar[int]
    EnableNvtx: bool
    DisableProfilingStartStop: bool
    EnableProfilingFromStart: bool
    ActivityType: str
    FilterOptions: FilterOptions
    OtherOptions: OtherOptions
    EnabledSections: str
    ClockControlMode: str
    SamplingOptions: SamplingOptions
    def __init__(self, EnableNvtx: _Optional[bool] = ..., DisableProfilingStartStop: _Optional[bool] = ..., EnableProfilingFromStart: _Optional[bool] = ..., ActivityType: _Optional[str] = ..., FilterOptions: _Optional[_Union[FilterOptions, _Mapping]] = ..., OtherOptions: _Optional[_Union[OtherOptions, _Mapping]] = ..., EnabledSections: _Optional[str] = ..., ClockControlMode: _Optional[str] = ..., SamplingOptions: _Optional[_Union[SamplingOptions, _Mapping]] = ...) -> None: ...

class ReportSessionDetails(_message.Message):
    __slots__ = ("ProcessID", "CreationTime", "HostSystemInfo", "TargetSystemInfo", "DeviceProperties", "DeviceAttributes", "Comments", "VersionInfo", "ExecutableSettings", "ProfilerSettings")
    PROCESSID_FIELD_NUMBER: _ClassVar[int]
    CREATIONTIME_FIELD_NUMBER: _ClassVar[int]
    HOSTSYSTEMINFO_FIELD_NUMBER: _ClassVar[int]
    TARGETSYSTEMINFO_FIELD_NUMBER: _ClassVar[int]
    DEVICEPROPERTIES_FIELD_NUMBER: _ClassVar[int]
    DEVICEATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    COMMENTS_FIELD_NUMBER: _ClassVar[int]
    VERSIONINFO_FIELD_NUMBER: _ClassVar[int]
    EXECUTABLESETTINGS_FIELD_NUMBER: _ClassVar[int]
    PROFILERSETTINGS_FIELD_NUMBER: _ClassVar[int]
    ProcessID: int
    CreationTime: int
    HostSystemInfo: SystemInfo
    TargetSystemInfo: SystemInfo
    DeviceProperties: DeviceProperties
    DeviceAttributes: _containers.RepeatedCompositeFieldContainer[DeviceAttributes]
    Comments: str
    VersionInfo: _containers.RepeatedCompositeFieldContainer[VersionInfo]
    ExecutableSettings: _ProfilerReportCommon_pb2.ExecutableSettings
    ProfilerSettings: ProfilerSettings
    def __init__(self, ProcessID: _Optional[int] = ..., CreationTime: _Optional[int] = ..., HostSystemInfo: _Optional[_Union[SystemInfo, _Mapping]] = ..., TargetSystemInfo: _Optional[_Union[SystemInfo, _Mapping]] = ..., DeviceProperties: _Optional[_Union[DeviceProperties, _Mapping]] = ..., DeviceAttributes: _Optional[_Iterable[_Union[DeviceAttributes, _Mapping]]] = ..., Comments: _Optional[str] = ..., VersionInfo: _Optional[_Iterable[_Union[VersionInfo, _Mapping]]] = ..., ExecutableSettings: _Optional[_Union[_ProfilerReportCommon_pb2.ExecutableSettings, _Mapping]] = ..., ProfilerSettings: _Optional[_Union[ProfilerSettings, _Mapping]] = ...) -> None: ...

class ProfileMetricValue(_message.Message):
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

class ProfileMetricListElement(_message.Message):
    __slots__ = ("CorrelationId", "ElementValue")
    CORRELATIONID_FIELD_NUMBER: _ClassVar[int]
    ELEMENTVALUE_FIELD_NUMBER: _ClassVar[int]
    CorrelationId: ProfileMetricValue
    ElementValue: ProfileMetricValue
    def __init__(self, CorrelationId: _Optional[_Union[ProfileMetricValue, _Mapping]] = ..., ElementValue: _Optional[_Union[ProfileMetricValue, _Mapping]] = ...) -> None: ...

class ProfileMetricResult(_message.Message):
    __slots__ = ("NameId", "MetricValue", "MetricValueList")
    NAMEID_FIELD_NUMBER: _ClassVar[int]
    METRICVALUE_FIELD_NUMBER: _ClassVar[int]
    METRICVALUELIST_FIELD_NUMBER: _ClassVar[int]
    NameId: int
    MetricValue: ProfileMetricValue
    MetricValueList: _containers.RepeatedCompositeFieldContainer[ProfileMetricListElement]
    def __init__(self, NameId: _Optional[int] = ..., MetricValue: _Optional[_Union[ProfileMetricValue, _Mapping]] = ..., MetricValueList: _Optional[_Iterable[_Union[ProfileMetricListElement, _Mapping]]] = ...) -> None: ...

class CommentID(_message.Message):
    __slots__ = ("SectionID",)
    SECTIONID_FIELD_NUMBER: _ClassVar[int]
    SectionID: str
    def __init__(self, SectionID: _Optional[str] = ...) -> None: ...

class Comment(_message.Message):
    __slots__ = ("ID", "DisplayName", "Text")
    ID_FIELD_NUMBER: _ClassVar[int]
    DISPLAYNAME_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    ID: CommentID
    DisplayName: str
    Text: str
    def __init__(self, ID: _Optional[_Union[CommentID, _Mapping]] = ..., DisplayName: _Optional[str] = ..., Text: _Optional[str] = ...) -> None: ...

class ProfileSeriesInfoMessage(_message.Message):
    __slots__ = ("SeriesID", "CombinationStr")
    SERIESID_FIELD_NUMBER: _ClassVar[int]
    COMBINATIONSTR_FIELD_NUMBER: _ClassVar[int]
    SeriesID: int
    CombinationStr: str
    def __init__(self, SeriesID: _Optional[int] = ..., CombinationStr: _Optional[str] = ...) -> None: ...

class ProfileResult(_message.Message):
    __slots__ = ("ThreadID", "APICallID", "ProgramHandle", "KernelID", "KernelMangledName", "KernelFunctionName", "KernelDemangledName", "WorkDimensions", "GlobalWorkOffset", "GlobalWorkSize", "LocalWorkSize", "Comments", "MetricResults", "CreationTime", "Source", "Api", "Sections", "SectionComments", "RuleResults", "UnsupportedDevice", "Nvtx", "ContextID", "StreamID", "SeriesInfo")
    THREADID_FIELD_NUMBER: _ClassVar[int]
    APICALLID_FIELD_NUMBER: _ClassVar[int]
    PROGRAMHANDLE_FIELD_NUMBER: _ClassVar[int]
    KERNELID_FIELD_NUMBER: _ClassVar[int]
    KERNELMANGLEDNAME_FIELD_NUMBER: _ClassVar[int]
    KERNELFUNCTIONNAME_FIELD_NUMBER: _ClassVar[int]
    KERNELDEMANGLEDNAME_FIELD_NUMBER: _ClassVar[int]
    WORKDIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    GLOBALWORKOFFSET_FIELD_NUMBER: _ClassVar[int]
    GLOBALWORKSIZE_FIELD_NUMBER: _ClassVar[int]
    LOCALWORKSIZE_FIELD_NUMBER: _ClassVar[int]
    COMMENTS_FIELD_NUMBER: _ClassVar[int]
    METRICRESULTS_FIELD_NUMBER: _ClassVar[int]
    CREATIONTIME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    API_FIELD_NUMBER: _ClassVar[int]
    SECTIONS_FIELD_NUMBER: _ClassVar[int]
    SECTIONCOMMENTS_FIELD_NUMBER: _ClassVar[int]
    RULERESULTS_FIELD_NUMBER: _ClassVar[int]
    UNSUPPORTEDDEVICE_FIELD_NUMBER: _ClassVar[int]
    NVTX_FIELD_NUMBER: _ClassVar[int]
    CONTEXTID_FIELD_NUMBER: _ClassVar[int]
    STREAMID_FIELD_NUMBER: _ClassVar[int]
    SERIESINFO_FIELD_NUMBER: _ClassVar[int]
    ThreadID: int
    APICallID: int
    ProgramHandle: int
    KernelID: int
    KernelMangledName: str
    KernelFunctionName: str
    KernelDemangledName: str
    WorkDimensions: int
    GlobalWorkOffset: _ProfilerCommon_pb2.Uint64x3
    GlobalWorkSize: _ProfilerCommon_pb2.Uint64x3
    LocalWorkSize: _ProfilerCommon_pb2.Uint64x3
    Comments: str
    MetricResults: _containers.RepeatedCompositeFieldContainer[ProfileMetricResult]
    CreationTime: int
    Source: _ProfilerCommon_pb2.SourceData
    Api: _ProfilerCommon_pb2.APIType
    Sections: _containers.RepeatedCompositeFieldContainer[_ProfilerSection_pb2.ProfilerSection]
    SectionComments: _containers.RepeatedCompositeFieldContainer[Comment]
    RuleResults: _containers.RepeatedCompositeFieldContainer[_RuleResults_pb2.RuleResult]
    UnsupportedDevice: bool
    Nvtx: _Nvtx_pb2.NvtxState
    ContextID: int
    StreamID: int
    SeriesInfo: ProfileSeriesInfoMessage
    def __init__(self, ThreadID: _Optional[int] = ..., APICallID: _Optional[int] = ..., ProgramHandle: _Optional[int] = ..., KernelID: _Optional[int] = ..., KernelMangledName: _Optional[str] = ..., KernelFunctionName: _Optional[str] = ..., KernelDemangledName: _Optional[str] = ..., WorkDimensions: _Optional[int] = ..., GlobalWorkOffset: _Optional[_Union[_ProfilerCommon_pb2.Uint64x3, _Mapping]] = ..., GlobalWorkSize: _Optional[_Union[_ProfilerCommon_pb2.Uint64x3, _Mapping]] = ..., LocalWorkSize: _Optional[_Union[_ProfilerCommon_pb2.Uint64x3, _Mapping]] = ..., Comments: _Optional[str] = ..., MetricResults: _Optional[_Iterable[_Union[ProfileMetricResult, _Mapping]]] = ..., CreationTime: _Optional[int] = ..., Source: _Optional[_Union[_ProfilerCommon_pb2.SourceData, _Mapping]] = ..., Api: _Optional[_Union[_ProfilerCommon_pb2.APIType, str]] = ..., Sections: _Optional[_Iterable[_Union[_ProfilerSection_pb2.ProfilerSection, _Mapping]]] = ..., SectionComments: _Optional[_Iterable[_Union[Comment, _Mapping]]] = ..., RuleResults: _Optional[_Iterable[_Union[_RuleResults_pb2.RuleResult, _Mapping]]] = ..., UnsupportedDevice: _Optional[bool] = ..., Nvtx: _Optional[_Union[_Nvtx_pb2.NvtxState, _Mapping]] = ..., ContextID: _Optional[int] = ..., StreamID: _Optional[int] = ..., SeriesInfo: _Optional[_Union[ProfileSeriesInfoMessage, _Mapping]] = ...) -> None: ...

class ProcessInfo(_message.Message):
    __slots__ = ("ProcessID", "Hostname", "ProcessName")
    PROCESSID_FIELD_NUMBER: _ClassVar[int]
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    PROCESSNAME_FIELD_NUMBER: _ClassVar[int]
    ProcessID: int
    Hostname: str
    ProcessName: str
    def __init__(self, ProcessID: _Optional[int] = ..., Hostname: _Optional[str] = ..., ProcessName: _Optional[str] = ...) -> None: ...

class BlockHeader(_message.Message):
    __slots__ = ("NumSources", "NumResults", "SessionDetails", "StringTable", "PayloadSize", "Process")
    NUMSOURCES_FIELD_NUMBER: _ClassVar[int]
    NUMRESULTS_FIELD_NUMBER: _ClassVar[int]
    SESSIONDETAILS_FIELD_NUMBER: _ClassVar[int]
    STRINGTABLE_FIELD_NUMBER: _ClassVar[int]
    PAYLOADSIZE_FIELD_NUMBER: _ClassVar[int]
    PROCESS_FIELD_NUMBER: _ClassVar[int]
    NumSources: int
    NumResults: int
    SessionDetails: ReportSessionDetails
    StringTable: _ProfilerStringTable_pb2.ProfilerStringTable
    PayloadSize: int
    Process: ProcessInfo
    def __init__(self, NumSources: _Optional[int] = ..., NumResults: _Optional[int] = ..., SessionDetails: _Optional[_Union[ReportSessionDetails, _Mapping]] = ..., StringTable: _Optional[_Union[_ProfilerStringTable_pb2.ProfilerStringTable, _Mapping]] = ..., PayloadSize: _Optional[int] = ..., Process: _Optional[_Union[ProcessInfo, _Mapping]] = ...) -> None: ...

class FileHeader(_message.Message):
    __slots__ = ("Version",)
    VERSION_FIELD_NUMBER: _ClassVar[int]
    Version: int
    def __init__(self, Version: _Optional[int] = ...) -> None: ...
