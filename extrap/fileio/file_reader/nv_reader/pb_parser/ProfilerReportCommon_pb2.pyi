from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ExecutableSettings(_message.Message):
    __slots__ = ("ExecutablePath", "WorkDirectory", "CmdlineAgruments", "Environment")
    EXECUTABLEPATH_FIELD_NUMBER: _ClassVar[int]
    WORKDIRECTORY_FIELD_NUMBER: _ClassVar[int]
    CMDLINEAGRUMENTS_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    ExecutablePath: str
    WorkDirectory: str
    CmdlineAgruments: str
    Environment: str
    def __init__(self, ExecutablePath: _Optional[str] = ..., WorkDirectory: _Optional[str] = ..., CmdlineAgruments: _Optional[str] = ..., Environment: _Optional[str] = ...) -> None: ...
