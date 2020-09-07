import unittest
from random import shuffle

import numpy as np

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.functions import MultiParameterFunction
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.terms import CompoundTerm, MultiParameterTerm
from extrap.fileio.jsonlines_file_reader import read_jsonlines_file
from extrap.fileio.text_file_reader import read_text_file
from extrap.modelers.model_generator import ModelGenerator
from extrap.modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler
from tests.test_modeling import TestCaseWithFunctionAssertions


class TestFindFirstMeasurements(unittest.TestCase):

    def test_2parameters_basic(self):
        experiment = read_text_file('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]

        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 2)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate((20,)),
            Coordinate((30,)),
            Coordinate((40,)),
            Coordinate((50,)),
            Coordinate((60,))
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate((1,)),
            Coordinate((2,)),
            Coordinate((3,)),
            Coordinate((4,)),
            Coordinate((5,))
        ])

    def test_2parameters_reversed(self):
        experiment = read_text_file('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        measurements = list(reversed(measurements))

        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 2)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate((60,)),
            Coordinate((50,)),
            Coordinate((40,)),
            Coordinate((30,)),
            Coordinate((20,))
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate((5,)),
            Coordinate((4,)),
            Coordinate((3,)),
            Coordinate((2,)),
            Coordinate((1,))
        ])

    def test_2parameters_random(self):
        experiment = read_text_file('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        for _ in range(len(measurements)):
            shuffle(measurements)

            f_msm = modeler.find_first_measurement_points(measurements)

            self.assertEqual(len(f_msm), 2)
            self.assertSetEqual(set(m.coordinate for m in f_msm[0]),
                                {Coordinate(20), Coordinate(30), Coordinate(40), Coordinate(50), Coordinate(60)})
            self.assertSetEqual(set(m.coordinate for m in f_msm[1]),
                                {Coordinate(1), Coordinate(2), Coordinate(3), Coordinate(4), Coordinate(5)})

    def test_3parameters_basic(self):
        experiment = read_text_file('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 3)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate(20),
            Coordinate(30),
            Coordinate(40),
            Coordinate(50),
            Coordinate(60)
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate(1),
            Coordinate(2),
            Coordinate(3),
            Coordinate(4),
            Coordinate(5)
        ])
        self.assertListEqual([m.coordinate for m in f_msm[2]], [
            Coordinate(100),
            Coordinate(200),
            Coordinate(300),
            Coordinate(400),
            Coordinate(500)
        ])

    def test_3parameters_reversed(self):
        experiment = read_text_file('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        measurements = list(reversed(measurements))

        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 3)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate(60),
            Coordinate(50),
            Coordinate(40),
            Coordinate(30),
            Coordinate(20)
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate(5),
            Coordinate(4),
            Coordinate(3),
            Coordinate(2),
            Coordinate(1)
        ])
        self.assertListEqual([m.coordinate for m in f_msm[2]], [
            Coordinate(500),
            Coordinate(400),
            Coordinate(300),
            Coordinate(200),
            Coordinate(100)
        ])

    def test_3parameters_random(self):
        experiment = read_text_file('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        for _ in range(len(measurements)):
            shuffle(measurements)

            f_msm = modeler.find_first_measurement_points(measurements)

            self.assertEqual(len(f_msm), 3)
            self.assertSetEqual(set(m.coordinate for m in f_msm[0]), {
                Coordinate(20),
                Coordinate(30),
                Coordinate(40),
                Coordinate(50),
                Coordinate(60)
            })
            self.assertSetEqual(set(m.coordinate for m in f_msm[1]), {
                Coordinate(1),
                Coordinate(2),
                Coordinate(3),
                Coordinate(4),
                Coordinate(5)
            })
            self.assertSetEqual(set(m.coordinate for m in f_msm[2]), {
                Coordinate(100),
                Coordinate(200),
                Coordinate(300),
                Coordinate(400),
                Coordinate(500)
            })


class TestSparseModeling(TestCaseWithFunctionAssertions):
    def test_1(self):
        experiment = read_jsonlines_file('data/jsonlines/test1.jsonl')
        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        model_generator.model_all()

    def test_2(self):
        experiment = read_jsonlines_file('data/jsonlines/test2.jsonl')
        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        model_generator.model_all()

    def test_input_1(self):
        experiment = read_jsonlines_file('data/jsonlines/input_1.jsonl')
        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        model_generator.model_all()

    def test_complete_matrix_2p(self):
        experiment = read_jsonlines_file('data/jsonlines/complete_matrix_2p.jsonl')
        modeler = MultiParameterModeler()
        modeler.single_parameter_point_selection = 'all'
        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        model_generator.model_all()

    def test_cross_matrix_2p(self):
        experiment = read_jsonlines_file('data/jsonlines/cross_matrix_2p.jsonl')
        modeler = MultiParameterModeler()
        modeler.single_parameter_point_selection = 'all'
        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        self.assertWarns(UserWarning, model_generator.model_all)

    def test_modeling(self):
        exponents = [(0, 1, 1), (0, 1, 2), (1, 4, 0), (1, 3, 0), (1, 4, 1), (1, 3, 1), (1, 4, 2), (1, 3, 2),
                     (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 3, 0), (3, 4, 0), (2, 3, 1), (3, 4, 1), (4, 5, 0),
                     (2, 3, 2), (3, 4, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (5, 4, 0), (5, 4, 1), (4, 3, 0),
                     (4, 3, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2), (5, 3, 0), (7, 4, 0), (2, 1, 0), (2, 1, 1),
                     (2, 1, 2), (9, 4, 0), (7, 3, 0), (5, 2, 0), (5, 2, 1), (5, 2, 2), (8, 3, 0), (11, 4, 0),
                     (3, 1, 0), (3, 1, 1)]
        for expo1, expo2 in zip(exponents, exponents[1:]):
            termX = CompoundTerm.create(*expo1)
            termY = CompoundTerm.create(*expo2)
            term = MultiParameterTerm((0, termX), (1, termY))
            term.coefficient = 10
            function = MultiParameterFunction(term)
            function.constant_coefficient = 200
            points = [np.array([2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 2, 4, 8, 16, 32]),
                      np.array([2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32])]

            values = function.evaluate(np.array(points))
            measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(zip(*points), values)]
            modeler = MultiParameterModeler()

            models = modeler.model([measurements])
            self.assertEqual(1, len(models))
            self.assertApproxFunction(function, models[0].hypothesis.function)

    def test_modeling_plus(self):
        exponents = [(0, 1, 1), (0, 1, 2), (1, 4, 0), (1, 3, 0), (1, 4, 1), (1, 3, 1), (1, 4, 2), (1, 3, 2),
                     (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 3, 0), (3, 4, 0), (2, 3, 1), (3, 4, 1), (4, 5, 0),
                     (2, 3, 2), (3, 4, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (5, 4, 0), (5, 4, 1), (4, 3, 0),
                     (4, 3, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2), (5, 3, 0), (7, 4, 0), (2, 1, 0), (2, 1, 1),
                     (2, 1, 2), (9, 4, 0), (7, 3, 0), (5, 2, 0), (5, 2, 1), (5, 2, 2), (8, 3, 0), (11, 4, 0),
                     (3, 1, 0), (3, 1, 1)]
        for expo1, expo2 in zip(exponents, exponents[1:]):
            termX = CompoundTerm.create(*expo1)
            termY = CompoundTerm.create(*expo2)
            term1 = MultiParameterTerm((0, termX))
            term1.coefficient = 10
            term2 = MultiParameterTerm((1, termY))
            term2.coefficient = 20
            function = MultiParameterFunction(term1, term2)
            function.constant_coefficient = 200
            points = [np.array([2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 2, 4, 8, 16, 32]),
                      np.array([2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32])]

            values = function.evaluate(np.array(points))
            measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(zip(*points), values)]
            modeler = MultiParameterModeler()

            models = modeler.model([measurements])
            self.assertEqual(1, len(models))
            self.assertApproxFunction(function, models[0].hypothesis.function)
