# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

from pkg_resources import parse_version
import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO


if parse_version(kaitaistruct.__version__) < parse_version('0.9'):
    raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

from . import file_header
from . import profile_result
from . import profile_source
from . import block_header
class NsightCuprofReport(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.magic = self._io.read_bytes(4)
        if not self.magic == b"\x4E\x56\x52\x00":
            raise kaitaistruct.ValidationNotEqualError(b"\x4E\x56\x52\x00", self.magic, self._io, u"/seq/0")
        self.sizeof_header = self._io.read_u4le()
        _raw_header = self._io.read_bytes(self.sizeof_header)
        _io__raw_header = KaitaiStream(BytesIO(_raw_header))
        self.header = file_header.FileHeader(_io__raw_header)
        self.blocks = []
        i = 0
        while not self._io.is_eof():
            self.blocks.append(NsightCuprofReport.Block(self._io, self, self._root))
            i += 1


    class IBlockHeader(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            pass

        @property
        def num_sources(self):
            if hasattr(self, '_m_num_sources'):
                return self._m_num_sources if hasattr(self, '_m_num_sources') else None

            self._m_num_sources = 0
            return self._m_num_sources if hasattr(self, '_m_num_sources') else None

        @property
        def num_results(self):
            if hasattr(self, '_m_num_results'):
                return self._m_num_results if hasattr(self, '_m_num_results') else None

            self._m_num_results = 0
            return self._m_num_results if hasattr(self, '_m_num_results') else None

        @property
        def payload_size(self):
            if hasattr(self, '_m_payload_size'):
                return self._m_payload_size if hasattr(self, '_m_payload_size') else None

            self._m_payload_size = 0
            return self._m_payload_size if hasattr(self, '_m_payload_size') else None


    class PayloadResult(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.sizeof_payload = self._io.read_u4le()
            _raw_entry = self._io.read_bytes(self.sizeof_payload)
            _io__raw_entry = KaitaiStream(BytesIO(_raw_entry))
            self.entry = profile_result.ProfileResult(_io__raw_entry)


    class PayloadSource(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.sizeof_payload = self._io.read_u4le()
            _raw_entry = self._io.read_bytes(self.sizeof_payload)
            _io__raw_entry = KaitaiStream(BytesIO(_raw_entry))
            self.entry = profile_source.ProfileSource(_io__raw_entry)


    class PayloadEntries(KaitaiStruct):
        def __init__(self, num_sources, num_results, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.num_sources = num_sources
            self.num_results = num_results
            self._read()

        def _read(self):
            self.sources = [None] * (self.num_sources)
            for i in range(self.num_sources):
                self.sources[i] = NsightCuprofReport.PayloadSource(self._io, self, self._root)

            self.results = [None] * (self.num_results)
            for i in range(self.num_results):
                self.results[i] = NsightCuprofReport.PayloadResult(self._io, self, self._root)



    class Block(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.sizeof_header = self._io.read_u4le()
            _raw_header = self._io.read_bytes(self.sizeof_header)
            _io__raw_header = KaitaiStream(BytesIO(_raw_header))
            self.header = block_header.BlockHeader(_io__raw_header)
            _raw_payload = self._io.read_bytes(self.header.payload_size)
            _io__raw_payload = KaitaiStream(BytesIO(_raw_payload))
            self.payload = NsightCuprofReport.PayloadEntries(self.header.num_sources, self.header.num_results, _io__raw_payload, self, self._root)



