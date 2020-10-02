# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import tempfile
import unittest

from extrap.fileio.experiment_io import write_experiment, read_experiment
from extrap.fileio.text_file_reader import read_text_file
from extrap.modelers.model_generator import ModelGenerator


class TestMultiParameterAfterModeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = read_text_file("data/text/two_parameter_3.txt")
        ModelGenerator(cls.experiment).model_all()
        with tempfile.TemporaryFile() as tmp:
            write_experiment(cls.experiment, tmp)
            cls.reconstructed = read_experiment(tmp)

    def test_setup(self):
        self.setUpClass()

    def test_parameters(self):
        self.assertListEqual(self.experiment.parameters, self.reconstructed.parameters)
        pass

    def test_measurements(self):
        self.assertDictEqual(self.experiment.measurements, self.reconstructed.measurements)

    def test_coordinates(self):
        self.assertListEqual(self.experiment.coordinates, self.reconstructed.coordinates)

    def test_callpaths(self):
        self.assertListEqual(self.experiment.callpaths, self.reconstructed.callpaths)

    def test_metrics(self):
        self.assertListEqual(self.experiment.metrics, self.reconstructed.metrics)

    def test_call_tree(self):
        self.assertEqual(self.experiment.call_tree, self.reconstructed.call_tree)

    def test_modelers(self):
        self.assertListEqual(self.experiment.modelers, self.reconstructed.modelers)

    def test_scaling(self):
        self.assertEqual(self.experiment.scaling, self.reconstructed.scaling)


if __name__ == '__main__':
    unittest.main()
