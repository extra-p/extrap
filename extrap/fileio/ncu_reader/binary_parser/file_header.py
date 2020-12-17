from kaitaistruct import KaitaiStream

from extrap.fileio.ncu_reader.pb_parser.ProfilerReport_pb2 import FileHeader as PbFileHeader


class FileHeader():
    def __init__(self, stream: KaitaiStream):
        self.data = PbFileHeader()
        self.data.ParseFromString(stream.read_bytes_full())
