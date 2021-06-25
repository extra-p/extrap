#  This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
#  Copyright (c) 2021, Technical University of Darmstadt, Germany
#
#  This software may be modified and distributed under the terms of a BSD-style license.
#  See the LICENSE file in the base directory for details.

import unittest

from extrap.fileio.file_reader.perf_taint_reader import PerfTaintReader
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.exceptions import FileFormatError


class PerfTaintTest(unittest.TestCase):
    def test_loading(self):
        reader = PerfTaintReader()
        reader.scaling_type = 'weak'
        self.assertRaises(FileFormatError, reader.read_experiment, 'data/perf_taint/lulesh')
        experiment = reader.read_experiment('data/perf_taint/lulesh/lulesh.ll.json')

    def test_model(self):
        reader = PerfTaintReader()
        reader.scaling_type = 'weak'
        experiment = reader.read_experiment('data/perf_taint/lulesh/lulesh.ll.json')
        ModelGenerator(experiment).model_all()


if __name__ == '__main__':
    unittest.main()
