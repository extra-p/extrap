from kaitaistruct import KaitaiStream

from extrap.fileio.ncu_reader.pb_parser.ProfilerReport_pb2 import BlockHeader as PbBlockHeader


class BlockHeader():
    def __init__(self, stream: KaitaiStream):
        self.data = PbBlockHeader()
        self.data.ParseFromString(stream.read_bytes_full())

    @property
    def payload_size(self):
        return self.data.PayloadSize

    @property
    def num_sources(self):
        return self.data.NumSources

    @property
    def num_results(self):
        return self.data.NumResults
