# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import itertools
import unittest
from random import shuffle

import numpy as np

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.functions import MultiParameterFunction
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.terms import CompoundTerm, MultiParameterTerm
from extrap.fileio.file_reader.jsonlines_file_reader import read_jsonlines_file
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.modelers.model_generator import ModelGenerator
from extrap.modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler
from tests.modelling_testcase import TestCaseWithFunctionAssertions


class TestFindFirstMeasurements(unittest.TestCase):

    def test_2parameters_basic(self):
        experiment = TextFileReader().read_experiment('data/text/two_parameter_1.txt')

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
        experiment = TextFileReader().read_experiment('data/text/two_parameter_1.txt')

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
        experiment = TextFileReader().read_experiment('data/text/two_parameter_1.txt')

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
        experiment = TextFileReader().read_experiment('data/text/three_parameter_1.txt')

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
        experiment = TextFileReader().read_experiment('data/text/three_parameter_1.txt')

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
        experiment = TextFileReader().read_experiment('data/text/three_parameter_1.txt')

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
        model_generator = ModelGenerator(experiment, modeler)
        # create models from data
        model_generator.model_all()

    def test_cross_matrix_2p(self):
        experiment = read_jsonlines_file('data/jsonlines/cross_matrix_2p.jsonl')
        modeler = MultiParameterModeler()
        modeler.single_parameter_point_selection = 'all'
        # initialize model generator
        model_generator = ModelGenerator(experiment, modeler)
        # create models from data
        self.assertWarns(UserWarning, model_generator.model_all)

    def test_band_matrix_2p(self):
        experiment = read_jsonlines_file('data/jsonlines/band_matrix_2p.jsonl')
        modeler = MultiParameterModeler()
        modeler.single_parameter_point_selection = 'all'
        # initialize model generator
        model_generator = ModelGenerator(experiment, modeler)
        # create models from data
        self.assertWarns(UserWarning, model_generator.model_all)

    def test_matrix_3p(self):
        experiment = read_jsonlines_file('data/jsonlines/matrix_3p.jsonl')
        modeler = MultiParameterModeler()
        modeler.single_parameter_point_selection = 'all'
        # initialize model generator
        model_generator = ModelGenerator(experiment, modeler)
        # create models from data
        self.assertWarns(UserWarning, model_generator.model_all)

    def test_3parameters_bands_incomplete(self):
        experiment = read_jsonlines_file('data/jsonlines/matrix_3p_bands_incomplete.jsonl')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('<root>'), Metric('metr'))]

        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 3)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate(c) for c in [1, 3, 4, 5, 6]
        ])
        self.assertListEqual([0] + [1] * 4, [m.mean for m in f_msm[0]])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate(c) for c in range(1, 5 + 1)
        ])
        self.assertListEqual([0] + [2] * 4, [m.mean for m in f_msm[1]])
        self.assertListEqual([m.coordinate for m in f_msm[2]], [
            Coordinate(c) for c in range(1, 5 + 1)
        ])
        self.assertListEqual([0] + [4] * 4, [m.mean for m in f_msm[2]])

        measurements.reverse()

        f_msm = modeler.find_best_measurement_points(measurements)

        self.assertEqual(len(f_msm), 3)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate(c) for c in reversed([1, 3, 4, 5, 6])
        ])
        self.assertListEqual([1] * 4 + [0], [m.mean for m in f_msm[0]])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate(c) for c in [6, 5, 4, 3, 2]
        ])
        self.assertListEqual([3] * 5, [m.mean for m in f_msm[1]])
        self.assertListEqual([m.coordinate for m in f_msm[2]], [
            Coordinate(c) for c in reversed(range(1, 5 + 1))
        ])
        self.assertListEqual([4] * 4 + [0], [m.mean for m in f_msm[2]])

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

    def test_modeling_3p(self):
        exponents = [(0, 1, 1), (0, 1, 2), (1, 4, 0), (1, 3, 0), (1, 4, 1), (1, 3, 1), (1, 4, 2), (1, 3, 2),
                     (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 3, 0), (3, 4, 0), (2, 3, 1), (3, 4, 1), (4, 5, 0),
                     (2, 3, 2), (3, 4, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (5, 4, 0), (5, 4, 1), (4, 3, 0),
                     (4, 3, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2), (5, 3, 0), (7, 4, 0), (2, 1, 0), (2, 1, 1),
                     (2, 1, 2), (9, 4, 0), (7, 3, 0), (5, 2, 0), (5, 2, 1), (5, 2, 2), (8, 3, 0), (11, 4, 0),
                     (3, 1, 0), (3, 1, 1)]
        points = np.array(list(zip(*itertools.product([2, 4, 8, 16, 32], repeat=3))))
        for expo1, expo2, expo3 in zip(exponents, exponents[1:], exponents[2:]):
            termX = CompoundTerm.create(*expo1)
            termY = CompoundTerm.create(*expo2)
            termZ = CompoundTerm.create(*expo3)
            term = MultiParameterTerm((0, termX), (1, termY), (2, termZ))
            term.coefficient = 10
            function = MultiParameterFunction(term)
            function.constant_coefficient = 200

            values = function.evaluate(points)
            measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(zip(*points), values)]
            modeler = MultiParameterModeler()

            models = modeler.model([measurements])
            self.assertEqual(1, len(models))
            self.assertApproxFunction(function, models[0].hypothesis.function, places=5)

    def test_modeling_3p_plus(self):
        exponents = [(0, 1, 1), (0, 1, 2), (1, 4, 0), (1, 3, 0), (1, 4, 1), (1, 3, 1), (1, 4, 2), (1, 3, 2),
                     (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 3, 0), (3, 4, 0), (2, 3, 1), (3, 4, 1), (4, 5, 0),
                     (2, 3, 2), (3, 4, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (5, 4, 0), (5, 4, 1), (4, 3, 0),
                     (4, 3, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2), (5, 3, 0), (7, 4, 0), (2, 1, 0), (2, 1, 1),
                     (2, 1, 2), (9, 4, 0), (7, 3, 0), (5, 2, 0), (5, 2, 1), (5, 2, 2), (8, 3, 0), (11, 4, 0),
                     (3, 1, 0), (3, 1, 1)]
        points = np.array(list(zip(*itertools.product([2, 4, 8, 16, 32], repeat=3))))
        for expo1, expo2, expo3 in zip(exponents, exponents[1:], exponents[2:]):
            termX = CompoundTerm.create(*expo1)
            termY = CompoundTerm.create(*expo2)
            termZ = CompoundTerm.create(*expo3)
            term1 = MultiParameterTerm((0, termX))
            term1.coefficient = 10
            term2 = MultiParameterTerm((1, termY))
            term2.coefficient = 20
            term3 = MultiParameterTerm((2, termZ))
            term3.coefficient = 30
            function = MultiParameterFunction(term1, term2, term3)
            function.constant_coefficient = 200

            values = function.evaluate(points)
            measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(zip(*points), values)]
            modeler = MultiParameterModeler()

            models = modeler.model([measurements])
            self.assertEqual(1, len(models))
            self.assertApproxFunction(function, models[0].hypothesis.function)

    def test_modeling_3p_mul_plus(self):
        exponents = [(0, 1, 1), (0, 1, 2), (1, 4, 0), (1, 3, 0), (1, 4, 1), (1, 3, 1), (1, 4, 2), (1, 3, 2),
                     (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 3, 0), (3, 4, 0), (2, 3, 1), (3, 4, 1), (4, 5, 0),
                     (2, 3, 2), (3, 4, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (5, 4, 0), (5, 4, 1), (4, 3, 0),
                     (4, 3, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2), (5, 3, 0), (7, 4, 0), (2, 1, 0), (2, 1, 1),
                     (2, 1, 2), (9, 4, 0), (7, 3, 0), (5, 2, 0), (5, 2, 1), (5, 2, 2), (8, 3, 0), (11, 4, 0),
                     (3, 1, 0), (3, 1, 1)]
        points = np.array(list(zip(*itertools.product([2, 4, 8, 16, 32], repeat=3))))
        for expo1, expo2, expo3 in zip(exponents, exponents[1:], exponents[2:]):
            termX = CompoundTerm.create(*expo1)
            termY = CompoundTerm.create(*expo2)
            termZ = CompoundTerm.create(*expo3)
            term1 = MultiParameterTerm((0, termX))
            term1.coefficient = 100
            term2 = MultiParameterTerm((1, termY), (2, termZ))
            term2.coefficient = 2
            function = MultiParameterFunction(term1, term2)
            function.constant_coefficient = 200

            values = function.evaluate(points)
            measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(zip(*points), values)]
            modeler = MultiParameterModeler()

            models = modeler.model([measurements])
            self.assertEqual(1, len(models))
            self.assertApproxFunction(function, models[0].hypothesis.function)

    def test_modeling_4p(self):
        exponents = [(0, 1, 1), (0, 1, 2), (1, 4, 0), (1, 3, 0), (1, 4, 1), (1, 3, 1), (1, 4, 2), (1, 3, 2),
                     (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 3, 0), (3, 4, 0), (2, 3, 1), (3, 4, 1), (4, 5, 0),
                     (2, 3, 2), (3, 4, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (5, 4, 0), (5, 4, 1), (4, 3, 0),
                     (4, 3, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2), (5, 3, 0), (7, 4, 0), (2, 1, 0), (2, 1, 1),
                     (2, 1, 2), (9, 4, 0), (7, 3, 0), (5, 2, 0), (5, 2, 1), (5, 2, 2), (8, 3, 0), (11, 4, 0),
                     (3, 1, 0), (3, 1, 1)]
        points = np.array(list(zip(*itertools.product([2, 4, 8, 10, 12], repeat=4))))
        for expo1, expo2, expo3, expo4 in zip(exponents, exponents[1:], exponents[2:], exponents[3:]):
            termX = CompoundTerm.create(*expo1)
            termY = CompoundTerm.create(*expo2)
            termZ = CompoundTerm.create(*expo3)
            termW = CompoundTerm.create(*expo4)
            term = MultiParameterTerm((0, termX), (1, termY), (2, termZ), (3, termW))
            term.coefficient = 10
            function = MultiParameterFunction(term)
            function.constant_coefficient = 20000

            values = function.evaluate(points)
            measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(zip(*points), values)]
            modeler = MultiParameterModeler()

            models = modeler.model([measurements])
            self.assertEqual(1, len(models))
            self.assertApproxFunction(function, models[0].hypothesis.function)


class TestFindBestMeasurements(unittest.TestCase):

    def test_2parameters_basic(self):
        experiment = TextFileReader().read_experiment('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]

        f_msm = modeler.find_best_measurement_points(measurements)

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
        experiment = TextFileReader().read_experiment('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        measurements = list(reversed(measurements))

        f_msm = modeler.find_best_measurement_points(measurements)

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
        experiment = TextFileReader().read_experiment('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        for _ in range(len(measurements)):
            shuffle(measurements)

            f_msm = modeler.find_best_measurement_points(measurements)

            self.assertEqual(len(f_msm), 2)
            self.assertSetEqual(set(m.coordinate for m in f_msm[0]),
                                {Coordinate(20), Coordinate(30), Coordinate(40), Coordinate(50), Coordinate(60)})
            self.assertSetEqual(set(m.coordinate for m in f_msm[1]),
                                {Coordinate(1), Coordinate(2), Coordinate(3), Coordinate(4), Coordinate(5)})

    def test_3parameters_basic(self):
        experiment = TextFileReader().read_experiment('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        f_msm = modeler.find_best_measurement_points(measurements)

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
        experiment = TextFileReader().read_experiment('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        measurements = list(reversed(measurements))

        f_msm = modeler.find_best_measurement_points(measurements)

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
        experiment = TextFileReader().read_experiment('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        for _ in range(len(measurements)):
            shuffle(measurements)

            f_msm = modeler.find_best_measurement_points(measurements)

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

    def test_3parameters_sparse(self):
        experiment = read_jsonlines_file('data/jsonlines/matrix_3p.jsonl')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('<root>'), Metric('metr'))]
        for _ in range(len(measurements)):
            shuffle(measurements)

            f_msm = modeler.find_best_measurement_points(measurements)

            self.assertEqual(len(f_msm), 3)
            self.assertSetEqual(set(m.coordinate for m in f_msm[0]), {
                Coordinate(1),
                Coordinate(2),
                Coordinate(3),
                Coordinate(4),
                Coordinate(5)
            })
            self.assertListEqual([1] * 5, [m.mean for m in f_msm[0]])
            self.assertSetEqual(set(m.coordinate for m in f_msm[1]), {
                Coordinate(1),
                Coordinate(2),
                Coordinate(3),
                Coordinate(4),
                Coordinate(5)
            })
            self.assertListEqual([1] * 5, [m.mean for m in f_msm[1]])
            self.assertSetEqual(set(m.coordinate for m in f_msm[2]), {
                Coordinate(1),
                Coordinate(2),
                Coordinate(3),
                Coordinate(4),
                Coordinate(5)
            })
            self.assertListEqual([1] * 5, [m.mean for m in f_msm[2]])

    def test_3parameters_bands(self):
        experiment = read_jsonlines_file('data/jsonlines/matrix_3p_bands.jsonl')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('<root>'), Metric('metr'))]

        f_msm = modeler.find_best_measurement_points(measurements)

        self.assertEqual(len(f_msm), 3)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate(1),
            Coordinate(2),
            Coordinate(3),
            Coordinate(4),
            Coordinate(5)
        ])
        self.assertListEqual([0] + [1] * 4, [m.mean for m in f_msm[0]])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate(1),
            Coordinate(2),
            Coordinate(3),
            Coordinate(4),
            Coordinate(5)
        ])
        self.assertListEqual([0.5] + [2.5] * 4, [m.mean for m in f_msm[1]])
        self.assertListEqual([m.coordinate for m in f_msm[2]], [
            Coordinate(1),
            Coordinate(2),
            Coordinate(3),
            Coordinate(4),
            Coordinate(5)
        ])
        self.assertListEqual([0] + [4] * 4, [m.mean for m in f_msm[2]])

    def test_3parameters_bands_incomplete(self):
        experiment = read_jsonlines_file('data/jsonlines/matrix_3p_bands_incomplete.jsonl')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('<root>'), Metric('metr'))]

        f_msm = modeler.find_best_measurement_points(measurements)

        self.assertEqual(len(f_msm), 3)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate(c) for c in [1, 3, 4, 5, 6]
        ])
        self.assertListEqual([0] + [1] * 4, [m.mean for m in f_msm[0]])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate(c) for c in range(1, 5 + 1)
        ])
        self.assertListEqual([0] + [2] * 4, [m.mean for m in f_msm[1]])
        self.assertListEqual([m.coordinate for m in f_msm[2]], [
            Coordinate(c) for c in range(1, 5 + 1)
        ])
        self.assertListEqual([0] + [4] * 4, [m.mean for m in f_msm[2]])

        measurements.reverse()

        f_msm = modeler.find_best_measurement_points(measurements)

        self.assertEqual(len(f_msm), 3)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate(c) for c in reversed([1, 3, 4, 5, 6])
        ])
        self.assertListEqual([1] * 4 + [0], [m.mean for m in f_msm[0]])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate(c) for c in [6, 5, 4, 3, 2]
        ])
        self.assertListEqual([3] * 5, [m.mean for m in f_msm[1]])
        self.assertListEqual([m.coordinate for m in f_msm[2]], [
            Coordinate(c) for c in reversed(range(1, 5 + 1))
        ])
        self.assertListEqual([4] * 4 + [0], [m.mean for m in f_msm[2]])
