from kaitaistruct import KaitaiStream

from extrap.fileio.ncu_reader.pb_parser.ProfilerResults_pb2 import ProfilerSourceMessage


class ProfileSource:
    def __init__(self, stream: KaitaiStream):
        self.data = ProfilerSourceMessage()
        self.data.ParseFromString(stream.read_bytes_full())
