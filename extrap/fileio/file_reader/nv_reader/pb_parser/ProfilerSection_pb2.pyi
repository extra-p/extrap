import ProfilerMetricOptions_pb2 as _ProfilerMetricOptions_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HWUnitType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    Invalid: _ClassVar[HWUnitType]
    Default: _ClassVar[HWUnitType]
    Gpc: _ClassVar[HWUnitType]
    Tpc: _ClassVar[HWUnitType]
    Sm: _ClassVar[HWUnitType]
    Smsp: _ClassVar[HWUnitType]
    Tex: _ClassVar[HWUnitType]
    Lts: _ClassVar[HWUnitType]
    Ltc: _ClassVar[HWUnitType]
    Fbpa: _ClassVar[HWUnitType]
Invalid: HWUnitType
Default: HWUnitType
Gpc: HWUnitType
Tpc: HWUnitType
Sm: HWUnitType
Smsp: HWUnitType
Tex: HWUnitType
Lts: HWUnitType
Ltc: HWUnitType
Fbpa: HWUnitType

class ProfilerSectionMetricOption(_message.Message):
    __slots__ = ("Name", "Label", "Filter")
    NAME_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    Name: str
    Label: str
    Filter: _ProfilerMetricOptions_pb2.MetricOptionFilter
    def __init__(self, Name: _Optional[str] = ..., Label: _Optional[str] = ..., Filter: _Optional[_Union[_ProfilerMetricOptions_pb2.MetricOptionFilter, _Mapping]] = ...) -> None: ...

class ProfilerSectionMetric(_message.Message):
    __slots__ = ("Name", "Label", "HWUnit", "ShowInstances", "Unit", "Filter", "Options")
    NAME_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    HWUNIT_FIELD_NUMBER: _ClassVar[int]
    SHOWINSTANCES_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    Name: str
    Label: str
    HWUnit: HWUnitType
    ShowInstances: bool
    Unit: str
    Filter: _ProfilerMetricOptions_pb2.MetricOptionFilter
    Options: _containers.RepeatedCompositeFieldContainer[ProfilerSectionMetricOption]
    def __init__(self, Name: _Optional[str] = ..., Label: _Optional[str] = ..., HWUnit: _Optional[_Union[HWUnitType, str]] = ..., ShowInstances: _Optional[bool] = ..., Unit: _Optional[str] = ..., Filter: _Optional[_Union[_ProfilerMetricOptions_pb2.MetricOptionFilter, _Mapping]] = ..., Options: _Optional[_Iterable[_Union[ProfilerSectionMetricOption, _Mapping]]] = ...) -> None: ...

class ProfilerSectionHighlightX(_message.Message):
    __slots__ = ("Metrics",)
    METRICS_FIELD_NUMBER: _ClassVar[int]
    Metrics: _containers.RepeatedCompositeFieldContainer[ProfilerSectionMetric]
    def __init__(self, Metrics: _Optional[_Iterable[_Union[ProfilerSectionMetric, _Mapping]]] = ...) -> None: ...

class ProfilerSectionTable(_message.Message):
    __slots__ = ("Label", "Rows", "Columns", "Order", "Metrics")
    class LayoutOrder(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        RowMajor: _ClassVar[ProfilerSectionTable.LayoutOrder]
        ColumnMajor: _ClassVar[ProfilerSectionTable.LayoutOrder]
    RowMajor: ProfilerSectionTable.LayoutOrder
    ColumnMajor: ProfilerSectionTable.LayoutOrder
    LABEL_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    Label: str
    Rows: int
    Columns: int
    Order: ProfilerSectionTable.LayoutOrder
    Metrics: _containers.RepeatedCompositeFieldContainer[ProfilerSectionMetric]
    def __init__(self, Label: _Optional[str] = ..., Rows: _Optional[int] = ..., Columns: _Optional[int] = ..., Order: _Optional[_Union[ProfilerSectionTable.LayoutOrder, str]] = ..., Metrics: _Optional[_Iterable[_Union[ProfilerSectionMetric, _Mapping]]] = ...) -> None: ...

class ProfilerSectionChartAxisRange(_message.Message):
    __slots__ = ("Min", "Max")
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    Min: int
    Max: int
    def __init__(self, Min: _Optional[int] = ..., Max: _Optional[int] = ...) -> None: ...

class ProfilerSectionChartValueAxis(_message.Message):
    __slots__ = ("Label", "Range", "TickCount", "Size", "Precision")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    TICKCOUNT_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    PRECISION_FIELD_NUMBER: _ClassVar[int]
    Label: str
    Range: ProfilerSectionChartAxisRange
    TickCount: int
    Size: int
    Precision: int
    def __init__(self, Label: _Optional[str] = ..., Range: _Optional[_Union[ProfilerSectionChartAxisRange, _Mapping]] = ..., TickCount: _Optional[int] = ..., Size: _Optional[int] = ..., Precision: _Optional[int] = ...) -> None: ...

class ProfilerSectionChartCategoryAxis(_message.Message):
    __slots__ = ("Label",)
    LABEL_FIELD_NUMBER: _ClassVar[int]
    Label: str
    def __init__(self, Label: _Optional[str] = ...) -> None: ...

class ProfilerSectionChartHistogramAxis(_message.Message):
    __slots__ = ("Label", "BinCount")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    BINCOUNT_FIELD_NUMBER: _ClassVar[int]
    Label: str
    BinCount: int
    def __init__(self, Label: _Optional[str] = ..., BinCount: _Optional[int] = ...) -> None: ...

class ProfilerSectionBarChart(_message.Message):
    __slots__ = ("Label", "Direction", "CategoryAxis", "ValueAxis", "Metrics")
    class Directions(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        Horizontal: _ClassVar[ProfilerSectionBarChart.Directions]
        Vertical: _ClassVar[ProfilerSectionBarChart.Directions]
    Horizontal: ProfilerSectionBarChart.Directions
    Vertical: ProfilerSectionBarChart.Directions
    LABEL_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    CATEGORYAXIS_FIELD_NUMBER: _ClassVar[int]
    VALUEAXIS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    Label: str
    Direction: ProfilerSectionBarChart.Directions
    CategoryAxis: ProfilerSectionChartCategoryAxis
    ValueAxis: ProfilerSectionChartValueAxis
    Metrics: _containers.RepeatedCompositeFieldContainer[ProfilerSectionMetric]
    def __init__(self, Label: _Optional[str] = ..., Direction: _Optional[_Union[ProfilerSectionBarChart.Directions, str]] = ..., CategoryAxis: _Optional[_Union[ProfilerSectionChartCategoryAxis, _Mapping]] = ..., ValueAxis: _Optional[_Union[ProfilerSectionChartValueAxis, _Mapping]] = ..., Metrics: _Optional[_Iterable[_Union[ProfilerSectionMetric, _Mapping]]] = ...) -> None: ...

class ProfilerSectionHistogramChart(_message.Message):
    __slots__ = ("Label", "HistogramAxis", "ValueAxis", "Metric")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAMAXIS_FIELD_NUMBER: _ClassVar[int]
    VALUEAXIS_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    Label: str
    HistogramAxis: ProfilerSectionChartHistogramAxis
    ValueAxis: ProfilerSectionChartValueAxis
    Metric: ProfilerSectionMetric
    def __init__(self, Label: _Optional[str] = ..., HistogramAxis: _Optional[_Union[ProfilerSectionChartHistogramAxis, _Mapping]] = ..., ValueAxis: _Optional[_Union[ProfilerSectionChartValueAxis, _Mapping]] = ..., Metric: _Optional[_Union[ProfilerSectionMetric, _Mapping]] = ...) -> None: ...

class ProfilerSectionLineChart(_message.Message):
    __slots__ = ("Label", "AxisX", "AxisY", "Metrics", "HighlightX")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    AXISX_FIELD_NUMBER: _ClassVar[int]
    AXISY_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    HIGHLIGHTX_FIELD_NUMBER: _ClassVar[int]
    Label: str
    AxisX: ProfilerSectionChartValueAxis
    AxisY: ProfilerSectionChartValueAxis
    Metrics: _containers.RepeatedCompositeFieldContainer[ProfilerSectionMetric]
    HighlightX: ProfilerSectionHighlightX
    def __init__(self, Label: _Optional[str] = ..., AxisX: _Optional[_Union[ProfilerSectionChartValueAxis, _Mapping]] = ..., AxisY: _Optional[_Union[ProfilerSectionChartValueAxis, _Mapping]] = ..., Metrics: _Optional[_Iterable[_Union[ProfilerSectionMetric, _Mapping]]] = ..., HighlightX: _Optional[_Union[ProfilerSectionHighlightX, _Mapping]] = ...) -> None: ...

class ProfilerSectionMemorySharedTable(_message.Message):
    __slots__ = ("Label", "ShowLoads", "ShowStores", "ShowAtomics", "ShowTotals")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SHOWLOADS_FIELD_NUMBER: _ClassVar[int]
    SHOWSTORES_FIELD_NUMBER: _ClassVar[int]
    SHOWATOMICS_FIELD_NUMBER: _ClassVar[int]
    SHOWTOTALS_FIELD_NUMBER: _ClassVar[int]
    Label: str
    ShowLoads: bool
    ShowStores: bool
    ShowAtomics: bool
    ShowTotals: bool
    def __init__(self, Label: _Optional[str] = ..., ShowLoads: _Optional[bool] = ..., ShowStores: _Optional[bool] = ..., ShowAtomics: _Optional[bool] = ..., ShowTotals: _Optional[bool] = ...) -> None: ...

class ProfilerSectionMemoryFirstLevelCacheTable(_message.Message):
    __slots__ = ("Label", "ShowLoads", "ShowStores", "ShowAtomics", "ShowReductions", "ShowGlobal", "ShowLocal", "ShowSurface", "ShowTexture", "ShowTotalLoads", "ShowTotalStores", "ShowTotals")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SHOWLOADS_FIELD_NUMBER: _ClassVar[int]
    SHOWSTORES_FIELD_NUMBER: _ClassVar[int]
    SHOWATOMICS_FIELD_NUMBER: _ClassVar[int]
    SHOWREDUCTIONS_FIELD_NUMBER: _ClassVar[int]
    SHOWGLOBAL_FIELD_NUMBER: _ClassVar[int]
    SHOWLOCAL_FIELD_NUMBER: _ClassVar[int]
    SHOWSURFACE_FIELD_NUMBER: _ClassVar[int]
    SHOWTEXTURE_FIELD_NUMBER: _ClassVar[int]
    SHOWTOTALLOADS_FIELD_NUMBER: _ClassVar[int]
    SHOWTOTALSTORES_FIELD_NUMBER: _ClassVar[int]
    SHOWTOTALS_FIELD_NUMBER: _ClassVar[int]
    Label: str
    ShowLoads: bool
    ShowStores: bool
    ShowAtomics: bool
    ShowReductions: bool
    ShowGlobal: bool
    ShowLocal: bool
    ShowSurface: bool
    ShowTexture: bool
    ShowTotalLoads: bool
    ShowTotalStores: bool
    ShowTotals: bool
    def __init__(self, Label: _Optional[str] = ..., ShowLoads: _Optional[bool] = ..., ShowStores: _Optional[bool] = ..., ShowAtomics: _Optional[bool] = ..., ShowReductions: _Optional[bool] = ..., ShowGlobal: _Optional[bool] = ..., ShowLocal: _Optional[bool] = ..., ShowSurface: _Optional[bool] = ..., ShowTexture: _Optional[bool] = ..., ShowTotalLoads: _Optional[bool] = ..., ShowTotalStores: _Optional[bool] = ..., ShowTotals: _Optional[bool] = ...) -> None: ...

class ProfilerSectionMemorySecondLevelCacheTable(_message.Message):
    __slots__ = ("Label", "ShowLoads", "ShowStores", "ShowAtomics", "ShowReductions", "ShowGlobal", "ShowLocal", "ShowSurface", "ShowTexture", "ShowTotalLoads", "ShowTotalStores", "ShowTotals")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SHOWLOADS_FIELD_NUMBER: _ClassVar[int]
    SHOWSTORES_FIELD_NUMBER: _ClassVar[int]
    SHOWATOMICS_FIELD_NUMBER: _ClassVar[int]
    SHOWREDUCTIONS_FIELD_NUMBER: _ClassVar[int]
    SHOWGLOBAL_FIELD_NUMBER: _ClassVar[int]
    SHOWLOCAL_FIELD_NUMBER: _ClassVar[int]
    SHOWSURFACE_FIELD_NUMBER: _ClassVar[int]
    SHOWTEXTURE_FIELD_NUMBER: _ClassVar[int]
    SHOWTOTALLOADS_FIELD_NUMBER: _ClassVar[int]
    SHOWTOTALSTORES_FIELD_NUMBER: _ClassVar[int]
    SHOWTOTALS_FIELD_NUMBER: _ClassVar[int]
    Label: str
    ShowLoads: bool
    ShowStores: bool
    ShowAtomics: bool
    ShowReductions: bool
    ShowGlobal: bool
    ShowLocal: bool
    ShowSurface: bool
    ShowTexture: bool
    ShowTotalLoads: bool
    ShowTotalStores: bool
    ShowTotals: bool
    def __init__(self, Label: _Optional[str] = ..., ShowLoads: _Optional[bool] = ..., ShowStores: _Optional[bool] = ..., ShowAtomics: _Optional[bool] = ..., ShowReductions: _Optional[bool] = ..., ShowGlobal: _Optional[bool] = ..., ShowLocal: _Optional[bool] = ..., ShowSurface: _Optional[bool] = ..., ShowTexture: _Optional[bool] = ..., ShowTotalLoads: _Optional[bool] = ..., ShowTotalStores: _Optional[bool] = ..., ShowTotals: _Optional[bool] = ...) -> None: ...

class ProfilerSectionMemoryDeviceMemoryTable(_message.Message):
    __slots__ = ("Label", "ShowLoads", "ShowStores", "ShowTotals")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SHOWLOADS_FIELD_NUMBER: _ClassVar[int]
    SHOWSTORES_FIELD_NUMBER: _ClassVar[int]
    SHOWTOTALS_FIELD_NUMBER: _ClassVar[int]
    Label: str
    ShowLoads: bool
    ShowStores: bool
    ShowTotals: bool
    def __init__(self, Label: _Optional[str] = ..., ShowLoads: _Optional[bool] = ..., ShowStores: _Optional[bool] = ..., ShowTotals: _Optional[bool] = ...) -> None: ...

class ProfilerSectionMemoryChart(_message.Message):
    __slots__ = ("Label",)
    LABEL_FIELD_NUMBER: _ClassVar[int]
    Label: str
    def __init__(self, Label: _Optional[str] = ...) -> None: ...

class ProfilerSectionGfxMetricsWidget(_message.Message):
    __slots__ = ("Type", "Label", "Metrics")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    Type: str
    Label: str
    Metrics: _containers.RepeatedCompositeFieldContainer[ProfilerSectionMetric]
    def __init__(self, Type: _Optional[str] = ..., Label: _Optional[str] = ..., Metrics: _Optional[_Iterable[_Union[ProfilerSectionMetric, _Mapping]]] = ...) -> None: ...

class ProfilerSectionHeader(_message.Message):
    __slots__ = ("Rows", "Metrics")
    ROWS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    Rows: int
    Metrics: _containers.RepeatedCompositeFieldContainer[ProfilerSectionMetric]
    def __init__(self, Rows: _Optional[int] = ..., Metrics: _Optional[_Iterable[_Union[ProfilerSectionMetric, _Mapping]]] = ...) -> None: ...

class ProfilerSectionBodyItem(_message.Message):
    __slots__ = ("Table", "BarChart", "HistogramChart", "LineChart", "MemorySharedTable", "MemoryFirstLevelCacheTable", "MemorySecondLevelCacheTable", "MemoryDeviceMemoryTable", "MemoryChart", "GfxMetricsWidget", "Filter")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    BARCHART_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAMCHART_FIELD_NUMBER: _ClassVar[int]
    LINECHART_FIELD_NUMBER: _ClassVar[int]
    MEMORYSHAREDTABLE_FIELD_NUMBER: _ClassVar[int]
    MEMORYFIRSTLEVELCACHETABLE_FIELD_NUMBER: _ClassVar[int]
    MEMORYSECONDLEVELCACHETABLE_FIELD_NUMBER: _ClassVar[int]
    MEMORYDEVICEMEMORYTABLE_FIELD_NUMBER: _ClassVar[int]
    MEMORYCHART_FIELD_NUMBER: _ClassVar[int]
    GFXMETRICSWIDGET_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    Table: ProfilerSectionTable
    BarChart: ProfilerSectionBarChart
    HistogramChart: ProfilerSectionHistogramChart
    LineChart: ProfilerSectionLineChart
    MemorySharedTable: ProfilerSectionMemorySharedTable
    MemoryFirstLevelCacheTable: ProfilerSectionMemoryFirstLevelCacheTable
    MemorySecondLevelCacheTable: ProfilerSectionMemorySecondLevelCacheTable
    MemoryDeviceMemoryTable: ProfilerSectionMemoryDeviceMemoryTable
    MemoryChart: ProfilerSectionMemoryChart
    GfxMetricsWidget: ProfilerSectionGfxMetricsWidget
    Filter: _ProfilerMetricOptions_pb2.MetricOptionFilter
    def __init__(self, Table: _Optional[_Union[ProfilerSectionTable, _Mapping]] = ..., BarChart: _Optional[_Union[ProfilerSectionBarChart, _Mapping]] = ..., HistogramChart: _Optional[_Union[ProfilerSectionHistogramChart, _Mapping]] = ..., LineChart: _Optional[_Union[ProfilerSectionLineChart, _Mapping]] = ..., MemorySharedTable: _Optional[_Union[ProfilerSectionMemorySharedTable, _Mapping]] = ..., MemoryFirstLevelCacheTable: _Optional[_Union[ProfilerSectionMemoryFirstLevelCacheTable, _Mapping]] = ..., MemorySecondLevelCacheTable: _Optional[_Union[ProfilerSectionMemorySecondLevelCacheTable, _Mapping]] = ..., MemoryDeviceMemoryTable: _Optional[_Union[ProfilerSectionMemoryDeviceMemoryTable, _Mapping]] = ..., MemoryChart: _Optional[_Union[ProfilerSectionMemoryChart, _Mapping]] = ..., GfxMetricsWidget: _Optional[_Union[ProfilerSectionGfxMetricsWidget, _Mapping]] = ..., Filter: _Optional[_Union[_ProfilerMetricOptions_pb2.MetricOptionFilter, _Mapping]] = ...) -> None: ...

class ProfilerSectionBody(_message.Message):
    __slots__ = ("Items", "DisplayName")
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    DISPLAYNAME_FIELD_NUMBER: _ClassVar[int]
    Items: _containers.RepeatedCompositeFieldContainer[ProfilerSectionBodyItem]
    DisplayName: str
    def __init__(self, Items: _Optional[_Iterable[_Union[ProfilerSectionBodyItem, _Mapping]]] = ..., DisplayName: _Optional[str] = ...) -> None: ...

class ProfilerSectionMetrics(_message.Message):
    __slots__ = ("Metrics", "Order")
    METRICS_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    Metrics: _containers.RepeatedCompositeFieldContainer[ProfilerSectionMetric]
    Order: int
    def __init__(self, Metrics: _Optional[_Iterable[_Union[ProfilerSectionMetric, _Mapping]]] = ..., Order: _Optional[int] = ...) -> None: ...

class ProfilerSection(_message.Message):
    __slots__ = ("Identifier", "DisplayName", "Order", "Header", "Body", "Metrics", "Description", "Extends")
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    DISPLAYNAME_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    EXTENDS_FIELD_NUMBER: _ClassVar[int]
    Identifier: str
    DisplayName: str
    Order: int
    Header: ProfilerSectionHeader
    Body: _containers.RepeatedCompositeFieldContainer[ProfilerSectionBody]
    Metrics: ProfilerSectionMetrics
    Description: str
    Extends: str
    def __init__(self, Identifier: _Optional[str] = ..., DisplayName: _Optional[str] = ..., Order: _Optional[int] = ..., Header: _Optional[_Union[ProfilerSectionHeader, _Mapping]] = ..., Body: _Optional[_Iterable[_Union[ProfilerSectionBody, _Mapping]]] = ..., Metrics: _Optional[_Union[ProfilerSectionMetrics, _Mapping]] = ..., Description: _Optional[str] = ..., Extends: _Optional[str] = ...) -> None: ...

class ProfilerSections(_message.Message):
    __slots__ = ("Sections",)
    SECTIONS_FIELD_NUMBER: _ClassVar[int]
    Sections: _containers.RepeatedCompositeFieldContainer[ProfilerSection]
    def __init__(self, Sections: _Optional[_Iterable[_Union[ProfilerSection, _Mapping]]] = ...) -> None: ...
