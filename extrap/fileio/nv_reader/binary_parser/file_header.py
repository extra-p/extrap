# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from kaitaistruct import KaitaiStream

from extrap.fileio.nv_reader.pb_parser.ProfilerReport_pb2 import FileHeader as PbFileHeader


class FileHeader():
    def __init__(self, stream: KaitaiStream):
        self.data = PbFileHeader()
        self.data.ParseFromString(stream.read_bytes_full())
