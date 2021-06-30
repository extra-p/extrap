# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Optional

from kaitaistruct import KaitaiStream

from extrap.fileio.file_reader.nv_reader.pb_parser.ProfilerCommon_pb2 import SourceData
from extrap.fileio.file_reader.nv_reader.pb_parser.ProfilerResults_pb2 import ProfilerSourceMessage


class ProfileSource:
    def __init__(self, stream: KaitaiStream):
        self.raw_data = stream.read_bytes_full()

    def parse(self) -> Optional[SourceData]:
        data = ProfilerSourceMessage()
        data.ParseFromString(self.raw_data)
        if data.IsInitialized():
            return data.Source
        else:
            return None
