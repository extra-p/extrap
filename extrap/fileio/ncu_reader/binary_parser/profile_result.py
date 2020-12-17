from kaitaistruct import KaitaiStream

from extrap.fileio.ncu_reader.pb_parser.ProfilerReport_pb2 import ProfileResult as PbProfileResult


class ProfileResult:
    def __init__(self, stream: KaitaiStream):
        raw_data = stream.read_bytes_full()
        self.data = PbProfileResult()
        self.data.ParseFromString(raw_data)
