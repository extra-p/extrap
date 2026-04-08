from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ProfilerStringTable(_message.Message):
    __slots__ = ("Strings",)
    STRINGS_FIELD_NUMBER: _ClassVar[int]
    Strings: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, Strings: _Optional[_Iterable[str]] = ...) -> None: ...
