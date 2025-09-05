# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO


if getattr(kaitaistruct, 'API_VERSION', (0, 9)) < (0, 9):
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
        self.len_header = self._io.read_u4le()
        _raw_header = self._io.read_bytes(self.len_header)
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
                return self._m_num_sources

            self._m_num_sources = 0
            return getattr(self, '_m_num_sources', None)

        @property
        def num_results(self):
            if hasattr(self, '_m_num_results'):
                return self._m_num_results

            self._m_num_results = 0
            return getattr(self, '_m_num_results', None)

        @property
        def payload_size(self):
            if hasattr(self, '_m_payload_size'):
                return self._m_payload_size

            self._m_payload_size = 0
            return getattr(self, '_m_payload_size', None)


    class PayloadResult(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.len_entry = self._io.read_u4le()
            _raw_entry = self._io.read_bytes(self.len_entry)
            _io__raw_entry = KaitaiStream(BytesIO(_raw_entry))
            self.entry = profile_result.ProfileResult(_io__raw_entry)


    class PayloadSource(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.len_entry = self._io.read_u4le()
            _raw_entry = self._io.read_bytes(self.len_entry)
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
            self.sources = []
            for i in range(self.num_sources):
                self.sources.append(NsightCuprofReport.PayloadSource(self._io, self, self._root))

            self.results = []
            for i in range(self.num_results):
                self.results.append(NsightCuprofReport.PayloadResult(self._io, self, self._root))



    class Block(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.len_header = self._io.read_u4le()
            _raw_header = self._io.read_bytes(self.len_header)
            _io__raw_header = KaitaiStream(BytesIO(_raw_header))
            self.header = block_header.BlockHeader(_io__raw_header)
            _raw_payload = self._io.read_bytes(self.header.payload_size)
            _io__raw_payload = KaitaiStream(BytesIO(_raw_payload))
            self.payload = NsightCuprofReport.PayloadEntries(self.header.num_sources, self.header.num_results, _io__raw_payload, self, self._root)



