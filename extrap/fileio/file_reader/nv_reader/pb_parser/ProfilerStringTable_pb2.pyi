# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
    FileDescriptor as google___protobuf___descriptor___FileDescriptor,
)

from google.protobuf.internal.containers import (
    RepeatedScalarFieldContainer as google___protobuf___internal___containers___RepeatedScalarFieldContainer,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Iterable as typing___Iterable,
    Optional as typing___Optional,
    Text as typing___Text,
)

from typing import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int


DESCRIPTOR: google___protobuf___descriptor___FileDescriptor = ...

class ProfilerStringTable(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Strings: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text] = ...

    def __init__(self,
        *,
        Strings : typing___Optional[typing___Iterable[typing___Text]] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Strings",b"Strings"]) -> None: ...
type___ProfilerStringTable = ProfilerStringTable