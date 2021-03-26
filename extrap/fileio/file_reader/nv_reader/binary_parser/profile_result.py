# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from kaitaistruct import KaitaiStream

from extrap.fileio.file_reader.nv_reader.pb_parser.ProfilerReport_pb2 import ProfileResult as PbProfileResult


class ProfileResult:
    def __init__(self, stream: KaitaiStream):
        self.raw_data = stream.read_bytes_full()

    def parse(self) -> PbProfileResult:
        data = PbProfileResult()
        data.ParseFromString(self.raw_data)
        return data
