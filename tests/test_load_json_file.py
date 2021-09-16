# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import itertools
import unittest

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio.file_reader.json_file_reader import JsonFileReader
from extrap.util.exceptions import InvalidExperimentError


# noinspection PyUnusedLocal
class TestJsonFiles(unittest.TestCase):

    def test_read_1(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = JsonFileReader().read_experiment("data/json/input_1.JSON")
        self.assertListEqual(experiment.parameters, [Parameter('x')])
        self.assertListEqual([p.id for p in experiment.parameters], [0])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(4),
            Coordinate(8),
            Coordinate(16),
            Coordinate(32),
            Coordinate(64)
        ])
        self.assertListEqual(experiment.metrics, [
            Metric('time')
        ])
        self.assertListEqual(experiment.callpaths, [
            Callpath('sweep')
        ])

    def test_read_2(self):
        experiment = JsonFileReader().read_experiment("data/json/input_2.JSON")

    def test_read_3(self):
        experiment = JsonFileReader().read_experiment("data/json/input_3.JSON")

    def test_read_4(self):
        experiment = JsonFileReader().read_experiment("data/json/input_4.JSON")

    def test_read_5(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = JsonFileReader().read_experiment("data/json/input_5.JSON")
        self.assertListEqual(experiment.parameters, [Parameter('x'), Parameter('y')])
        self.assertListEqual([p.id for p in experiment.parameters], [0, 1])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(4.0, 10.0), Coordinate(4.0, 20.0), Coordinate(4.0, 30.0), Coordinate(4.0, 40.0),
            Coordinate(4.0, 50.0),
            Coordinate(8.0, 10.0), Coordinate(8.0, 20.0), Coordinate(8.0, 30.0), Coordinate(8.0, 40.0),
            Coordinate(8.0, 50.0),
            Coordinate(16.0, 10.0), Coordinate(16.0, 20.0), Coordinate(16.0, 30.0), Coordinate(16.0, 40.0),
            Coordinate(16.0, 50.0),
            Coordinate(32.0, 10.0), Coordinate(32.0, 20.0), Coordinate(32.0, 30.0), Coordinate(32.0, 40.0),
            Coordinate(32.0, 50.0),
            Coordinate(64.0, 10.0), Coordinate(64.0, 20.0), Coordinate(64.0, 30.0), Coordinate(64.0, 40.0),
            Coordinate(64.0, 50.0)
        ])

    def test_read_6(self):
        experiment = JsonFileReader().read_experiment("data/json/input_6.JSON")

    def test_read_7(self):
        experiment = JsonFileReader().read_experiment("data/json/input_7.JSON")

    def test_read_8(self):
        experiment = JsonFileReader().read_experiment("data/json/input_8.JSON")

    def test_read_9(self):
        experiment = JsonFileReader().read_experiment("data/json/input_9.JSON")

    def test_read_10(self):
        experiment = JsonFileReader().read_experiment("data/json/input_10.JSON")

    def test_read_11(self):
        self.assertRaises(InvalidExperimentError, JsonFileReader().read_experiment, "data/json/input_11.JSON")

    def test_read_12(self):
        experiment = JsonFileReader().read_experiment("data/json/input_12.JSON")
        self.assertListEqual(experiment.parameters, [Parameter('x'), Parameter('y')])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(4.0, 10.0), Coordinate(4.0, 20.0), Coordinate(4.0, 30.0), Coordinate(4.0, 40.0),
            Coordinate(4.0, 50.0),
            Coordinate(8.0, 10.0), Coordinate(8.0, 20.0), Coordinate(8.0, 30.0), Coordinate(8.0, 40.0),
            Coordinate(8.0, 50.0),
            Coordinate(16.0, 10.0), Coordinate(16.0, 20.0), Coordinate(16.0, 30.0), Coordinate(16.0, 40.0),
            Coordinate(16.0, 50.0),
            Coordinate(32.0, 10.0), Coordinate(32.0, 20.0), Coordinate(32.0, 30.0), Coordinate(32.0, 40.0),
            Coordinate(32.0, 50.0),
            Coordinate(64.0, 10.0), Coordinate(64.0, 20.0), Coordinate(64.0, 30.0), Coordinate(64.0, 40.0),
            Coordinate(64.0, 50.0)
        ])


class TestNewJsonFiles(unittest.TestCase):
    def test_read_1(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = JsonFileReader().read_experiment("data/json/new/input1.json")
        self.assertListEqual(experiment.parameters, [Parameter('x')])
        self.assertListEqual([p.id for p in experiment.parameters], [0])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(4),
            Coordinate(8),
            Coordinate(16),
            Coordinate(32),
            Coordinate(64)
        ])
        self.assertListEqual(experiment.metrics, [
            Metric('time')
        ])
        self.assertListEqual(experiment.callpaths, [
            Callpath('sweep')
        ])

    def test_read_2(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = JsonFileReader().read_experiment("data/json/new/input2.json")
        x = Parameter('x')
        y = Parameter('y')
        self.assertListEqual(experiment.parameters, [x, y])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(4, 10),
            Coordinate(8, 20),
            Coordinate(16, 30),
            Coordinate(32, 40),
            Coordinate(64, 50)
        ])
        self.assertListEqual(experiment.metrics, [
            Metric('time'), Metric('visits')
        ])
        self.assertListEqual(experiment.callpaths, [
            Callpath('sweep'), Callpath('sweep2')
        ])
