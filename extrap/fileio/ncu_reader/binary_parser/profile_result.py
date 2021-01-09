from kaitaistruct import KaitaiStream
from memory_profiler import profile

from extrap.fileio.ncu_reader.pb_parser.ProfilerReport_pb2 import ProfileResult as PbProfileResult


class ProfileResult:
    def __init__(self, stream: KaitaiStream):
        self.raw_data = stream.read_bytes_full()
