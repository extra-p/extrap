# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from kaitaistruct import KaitaiStream

from extrap.fileio.file_reader.nv_reader.pb_parser.ProfilerReport_pb2 import BlockHeader as PbBlockHeader


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
