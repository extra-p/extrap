# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import struct
from zipfile import ZipFile

import numpy as np
from numpy.lib import format

from extrap.util.exceptions import FileFormatError


class ValueWriter:

    def __init__(self, zipfile: ZipFile):
        self._zipfile = zipfile
        self.max_memory_size = 1024 * 1024 * 1024
        self.NUMPY_HEADER_SIZE = 128

        self.index = 0
        self.chunk_index = 0
        self.values = []
        self.byte_size = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

    def write_values(self, values: np.ndarray):
        self.values.append(values)
        self.byte_size += values.itemsize * values.size + self.NUMPY_HEADER_SIZE

        current = self.index
        self.index += 1
        if self.byte_size >= self.max_memory_size:
            self.flush()

        return self.chunk_index, current

    def write_to_file(self):

        pos = 0
        index = []

        with self._zipfile.open("values/" + str(self.chunk_index) + ".vals", 'w') as file:
            old_write = file.write

            def write_with_pos_count(b: bytes):
                nonlocal pos
                written = old_write(b)
                pos += written
                return written

            file.write = write_with_pos_count

            file.write(b"Extra-P VALUES\0\0")

            for val in self.values:
                index.append(pos)
                format.write_array(file, val)

        with self._zipfile.open("values/" + str(self.chunk_index) + ".indx", 'w') as file:
            old_write = file.write

            def write_with_pos_count(b: bytes):
                nonlocal pos
                written = old_write(b)
                pos += written
                return written

            file.write = write_with_pos_count

            file.write(b"Extra-P INDEX\0\0\0")
            file.write(struct.pack('<Q', len(index)))

            for pos in index:
                file.write(struct.pack('<Q', pos))

    def flush(self):
        if self.index == 0:
            return
        self.write_to_file()
        self.index = 0
        self.chunk_index += 1
        self.values.clear()
        self.byte_size = 0


class ValueReader:
    def __init__(self, zipfile: ZipFile):
        self._zipfile = zipfile

    def read_values(self, chunk_index: int, index: int):
        with self._zipfile.open("values/" + str(chunk_index) + ".indx", 'r') as file:
            magic_number = file.read(16)
            if magic_number != b"Extra-P INDEX\0\0\0":
                raise FileFormatError("Could not read index file to retrieve values.")
            file.seek(struct.calcsize('<Q') * (index + 1) + 16)
            start_position = struct.unpack_from('<Q', file.read(struct.calcsize('<Q')))[0]

        with self._zipfile.open("values/" + str(chunk_index) + ".vals", 'r') as file:
            magic_number = file.read(16)
            if magic_number != b"Extra-P VALUES\0\0":
                raise FileFormatError("Could not read value file to retrieve values.")
            file.seek(start_position)
            return format.read_array(file)
