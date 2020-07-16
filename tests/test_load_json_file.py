import itertools
import unittest
from fileio.json_file_reader import read_json_file
from entities.callpath import Callpath
from entities.parameter import Parameter
from entities.coordinate import Coordinate
from entities.metric import Metric
import timeit

from util.exceptions import InvalidExperimentError


class TestJsonFiles(unittest.TestCase):

    def test_read_1(self):
        experiment = read_json_file("data/json/input_1.JSON")
        x = Parameter('x')
        self.assertListEqual(experiment.parameters, [x])
        self.assertListEqual([p.id for p in experiment.parameters], [0])
        self.assertListEqual(experiment.coordinates, [
            Coordinate([(x, 4)]),
            Coordinate([(x, 8)]),
            Coordinate([(x, 16)]),
            Coordinate([(x, 32)]),
            Coordinate([(x, 64)])
        ])
        self.assertListEqual(experiment.metrics, [
            Metric('time')
        ])
        self.assertListEqual(experiment.callpaths, [
            Callpath('sweep')
        ])

    def test_read_2(self):
        experiment = read_json_file("data/json/input_2.JSON")

    def test_read_3(self):
        experiment = read_json_file("data/json/input_3.JSON")

    def test_read_4(self):
        experiment = read_json_file("data/json/input_4.JSON")

    def test_read_5(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = read_json_file("data/json/input_5.JSON")
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
        experiment = read_json_file("data/json/input_6.JSON")

    def test_read_7(self):
        experiment = read_json_file("data/json/input_7.JSON")

    def test_read_8(self):
        experiment = read_json_file("data/json/input_8.JSON")

    def test_read_9(self):
        experiment = read_json_file("data/json/input_9.JSON")

    def test_read_10(self):
        experiment = read_json_file("data/json/input_10.JSON")

    def test_read_11(self):
        self.assertRaises(InvalidExperimentError, read_json_file, "data/json/input_11.JSON")

    def test_read_12(self):
        experiment = read_json_file("data/json/input_12.JSON")
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
