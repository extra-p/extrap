# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio.talpas_file_reader import read_talpas_file
from extrap.util.exceptions import InvalidExperimentError


class TestTalpasFiles(unittest.TestCase):

    def test_read_1(self):
        experiment = read_talpas_file("data/talpas/talpas_1.txt")
        x = Parameter('x')
        self.assertListEqual(experiment.parameters, [x])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(20),
            Coordinate(30),
            Coordinate(40),
            Coordinate(50),
            Coordinate(60)
        ])
        self.assertListEqual(experiment.metrics, [
            Metric('time')
        ])
        self.assertListEqual(experiment.callpaths, [
            Callpath('compute')
        ])

    def test_read_2(self):
        experiment = read_talpas_file("data/talpas/talpas_2.txt")
        x, y = Parameter('x'), Parameter('y')
        self.assertListEqual(experiment.parameters, [x, y])

    def test_read_3(self):
        experiment = read_talpas_file("data/talpas/talpas_3.txt")

    def test_read_4(self):
        experiment = read_talpas_file("data/talpas/talpas_4.txt")

    def test_read_5(self):
        experiment = read_talpas_file("data/talpas/talpas_5.txt")

    def test_read_6(self):
        experiment = read_talpas_file("data/talpas/talpas_6.txt")

    def test_read_7(self):
        experiment = read_talpas_file("data/talpas/talpas_7.txt")

    def test_read_8(self):
        experiment = read_talpas_file("data/talpas/talpas_8.txt")

    def test_read_9(self):
        experiment = read_talpas_file("data/talpas/talpas_9.txt")

    def test_read_10(self):
        self.assertRaises(InvalidExperimentError, read_talpas_file, "data/talpas/talpas_10_neg.txt")
