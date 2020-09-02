import itertools
import unittest

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio.json_file_reader import read_json_file
from extrap.fileio.jsonlines_file_reader import read_jsonlines_file


# noinspection PyUnusedLocal
class TestJsonLinesFiles(unittest.TestCase):

    def test_read_1(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = read_jsonlines_file("data/jsonlines/test1.jsonl")
        self.assertListEqual([Parameter('x'), Parameter('y')], experiment.parameters)
        self.assertListEqual([0, 1], [p.id for p in experiment.parameters])
        self.assertListEqual([Coordinate(x, y) for x in range(1, 5 + 1) for y in range(1, 5 + 1)],
                             experiment.coordinates)
        self.assertListEqual([
            Metric('metr')
        ], experiment.metrics)
        self.assertListEqual([
            Callpath('<root>')
        ], experiment.callpaths)

    def test_read_1_json(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = read_json_file("data/jsonlines/test1.jsonl")
        self.assertListEqual([Parameter('x'), Parameter('y')], experiment.parameters)
        self.assertListEqual([0, 1], [p.id for p in experiment.parameters])
        self.assertListEqual([Coordinate(x, y) for x in range(1, 5 + 1) for y in range(1, 5 + 1)],
                             experiment.coordinates)
        self.assertListEqual([
            Metric('metr')
        ], experiment.metrics)
        self.assertListEqual([
            Callpath('<root>')
        ], experiment.callpaths)

    def test_read_2(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = read_jsonlines_file("data/jsonlines/test2.jsonl")
        self.assertListEqual([Parameter('p'), Parameter('n')], experiment.parameters)
        self.assertListEqual([0, 1], [p.id for p in experiment.parameters])
        self.assertListEqual([Coordinate(x, y) for x in [16, 32, 64, 128, 256] for y in [100, 200, 300, 400, 500]],
                             experiment.coordinates)
        self.assertListEqual([
            Metric('metr')
        ], experiment.metrics)
        self.assertListEqual([
            Callpath('<root>')
        ], experiment.callpaths)

    def test_matrix3p(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = read_jsonlines_file("data/jsonlines/matrix_3p.jsonl")
        self.assertListEqual([Parameter('x'), Parameter('y'), Parameter('z')], experiment.parameters)
        self.assertListEqual([0, 1, 2], [p.id for p in experiment.parameters])
        self.assertListEqual(
            [Coordinate(x, 1, 1) for x in range(1, 5 + 1)] +
            [Coordinate(1, x, 1) for x in range(2, 5 + 1)] +
            [Coordinate(1, 1, x) for x in range(2, 5 + 1)],
            experiment.coordinates)
        self.assertListEqual([
            Metric('metr')
        ], experiment.metrics)
        self.assertListEqual([
            Callpath('<root>')
        ], experiment.callpaths)

    def test_matrix4p(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = read_jsonlines_file("data/jsonlines/matrix_4p.jsonl")
        self.assertListEqual([Parameter('a'), Parameter('b'), Parameter('c'), Parameter('d')], experiment.parameters)
        self.assertListEqual([0, 1, 2, 3], [p.id for p in experiment.parameters])
        self.assertListEqual(
            [Coordinate(x, 1, 1, 1) for x in range(1, 5 + 1)] +
            [Coordinate(1, x, 1, 1) for x in range(2, 5 + 1)] +
            [Coordinate(1, 1, x, 1) for x in range(2, 5 + 1)] +
            [Coordinate(1, 1, 1, x) for x in range(2, 5 + 1)],
            experiment.coordinates)
        self.assertListEqual([
            Metric('metr')
        ], experiment.metrics)
        self.assertListEqual([
            Callpath('<root>')
        ], experiment.callpaths)

    def test_sparse_matrix2p(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = read_jsonlines_file("data/jsonlines/sparse_matrix_2p.jsonl")
        self.assertListEqual([Parameter('x'), Parameter('y')], experiment.parameters)
        self.assertListEqual([0, 1], [p.id for p in experiment.parameters])
        self.assertListEqual(
            [Coordinate(20, 1), Coordinate(30, 1), Coordinate(30, 2), Coordinate(40, 1), Coordinate(40, 2),
             Coordinate(40, 3), Coordinate(50, 1), Coordinate(50, 2), Coordinate(50, 3), Coordinate(50, 4),
             Coordinate(60, 1), Coordinate(60, 2), Coordinate(60, 3), Coordinate(60, 4), Coordinate(60, 5),
             Coordinate(70, 2), Coordinate(70, 3), Coordinate(70, 4), Coordinate(70, 5), Coordinate(80, 3),
             Coordinate(80, 4), Coordinate(80, 5), Coordinate(90, 4), Coordinate(90, 5), Coordinate(100, 5)],
            experiment.coordinates)
        self.assertListEqual([
            Metric('metr')
        ], experiment.metrics)
        self.assertListEqual([
            Callpath('<root>')
        ], experiment.callpaths)

    def test_input_1(self):
        experiment = read_jsonlines_file("data/jsonlines/input_1.jsonl")

    def test_incomplete_matrix_2p(self):
        experiment = read_jsonlines_file("data/jsonlines/incomplete_matrix_2p.jsonl")

    def test_cross_matrix_2p(self):
        experiment = read_jsonlines_file("data/jsonlines/cross_matrix_2p.jsonl")

    def test_complete_matrix_2p(self):
        experiment = read_jsonlines_file("data/jsonlines/complete_matrix_2p.jsonl")

    def test_extended1(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = read_jsonlines_file("data/jsonlines/extended1.jsonl")
        self.assertListEqual([Parameter('x'), Parameter('y')], experiment.parameters)
        self.assertListEqual([0, 1], [p.id for p in experiment.parameters])
        self.assertListEqual([Coordinate(x, y) for x in range(1, 5 + 1) for y in range(1, 5 + 1)],
                             experiment.coordinates)
        self.assertListEqual([
            Metric('metr')
        ], experiment.metrics)
        self.assertListEqual([
            Callpath('test')
        ], experiment.callpaths)

    def test_extended2(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = read_jsonlines_file("data/jsonlines/extended2.jsonl")
        self.assertListEqual([Parameter('x'), Parameter('y')], experiment.parameters)
        self.assertListEqual([0, 1], [p.id for p in experiment.parameters])
        self.assertListEqual([Coordinate(x, y) for x in range(1, 3 + 1) for y in range(1, 3 + 1)],
                             experiment.coordinates)
        self.assertListEqual([
            Metric('metr')
        ], experiment.metrics)
        self.assertListEqual([
            Callpath('test')
        ], experiment.callpaths)

    def test_reduced(self):
        Parameter.ID_COUNTER = itertools.count()
        experiment = read_jsonlines_file("data/jsonlines/reduced.jsonl")
        self.assertListEqual([Parameter('x'), Parameter('y')], experiment.parameters)
        self.assertListEqual([0, 1], [p.id for p in experiment.parameters])
        self.assertListEqual([Coordinate(x, y) for x in range(1, 5 + 1) for y in range(1, 5 + 1)],
                             experiment.coordinates)
        self.assertListEqual([
            Metric('<default>')
        ], experiment.metrics)
        self.assertListEqual([
            Callpath('<root>')
        ], experiment.callpaths)
