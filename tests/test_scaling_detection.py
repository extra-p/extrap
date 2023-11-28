# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

from extrap.fileio import io_helper
from extrap.fileio.file_reader.text_file_reader import TextFileReader


class TestWeakScalingTextFiles(unittest.TestCase):
    def test_read_one_parameter(self):
        reader = TextFileReader()
        for file in ["data/text/one_parameter_1.txt", "data/text/one_parameter_2.txt", "data/text/one_parameter_3.txt",
                     "data/text/one_parameter_4.txt", "data/text/one_parameter_5.txt", "data/text/one_parameter_6.txt",
                     "data/text/one_parameter_7.txt"]:
            experiment = reader.read_experiment(file)
            self.assertEqual([0], io_helper.check_for_strong_scaling(experiment))

    def test_read_two_parameter(self):
        reader = TextFileReader()
        for file in ["data/text/two_parameter_1.txt", "data/text/two_parameter_2.txt", "data/text/two_parameter_3.txt",
                     "data/text/two_parameter_4.txt", "data/text/two_parameter_5.txt", "data/text/two_parameter_6.txt"]:
            experiment = reader.read_experiment(file)
            self.assertEqual([0, 0], io_helper.check_for_strong_scaling(experiment))

    def test_read_three_parameter(self):
        reader = TextFileReader()
        for file in ["data/text/three_parameter_1.txt", "data/text/three_parameter_2.txt",
                     "data/text/three_parameter_3.txt"]:
            experiment = reader.read_experiment(file)
            self.assertEqual([0, 0, 0], io_helper.check_for_strong_scaling(experiment))


class TestStrongScalingTextFiles(unittest.TestCase):
    def test_read_one_parameter(self):
        reader = TextFileReader()
        experiment = reader.read_experiment("data/text/strong_scaling/one_parameter_1.txt")
        self.assertEqual([1], io_helper.check_for_strong_scaling(experiment))
        reader = TextFileReader()
        experiment = reader.read_experiment("data/text/strong_scaling/one_parameter_2.txt")
        self.assertEqual([1], io_helper.check_for_strong_scaling(experiment))
        reader = TextFileReader()
        experiment = reader.read_experiment("data/text/strong_scaling/one_parameter_6.txt")
        self.assertEqual([1], io_helper.check_for_strong_scaling(experiment))

    def test_read_two_parameter(self):
        reader = TextFileReader()
        experiment = reader.read_experiment("data/text/strong_scaling/two_parameter_1.txt")
        self.assertEqual([0, 1], io_helper.check_for_strong_scaling(experiment))
        reader = TextFileReader()
        experiment = reader.read_experiment("data/text/strong_scaling/two_parameter_2.txt")
        self.assertEqual([1, 0], io_helper.check_for_strong_scaling(experiment))
