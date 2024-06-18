# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
import importlib.util
import unittest

import numpy as np

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.functions import SingleParameterFunction
from extrap.entities.hypotheses import ConstantHypothesis
from extrap.entities.hypotheses import SingleParameterHypothesis
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.terms import CompoundTerm
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.modelers.model_generator import ModelGenerator
from extrap.modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler
from extrap.modelers.single_parameter import adaptive
from extrap.modelers.single_parameter.adaptive import AdaptiveModeler


class TestAdaptiveModeling(unittest.TestCase):

    def setUp(self):
        adaptive_modeler_package = importlib.util.find_spec('extrap_adaptive_modeler')
        if adaptive_modeler_package is None:
            self.skipTest("Adaptive modeling plugin is not installed.")

    def test_single_parameter_modeling(self):
        experiment = TextFileReader().read_experiment('data/text/one_parameter_6.txt')

        modeler = AdaptiveModeler()
        modeler.preset = 2
        # initialize model generator
        model_generator = ModelGenerator(experiment, modeler)

        # create models from data
        model_generator.model_all()

        models = experiment.modelers[0].models
        cp0 = Callpath('met1'), Metric('')
        # self.assertIsInstance(models[cp0].hypothesis, ConstantHypothesis)

        cp1 = Callpath('met2'), Metric('')
        self.assertIsInstance(models[cp1].hypothesis, ConstantHypothesis)
        self.assertAlmostEqual(models[cp1].hypothesis.function.constant_coefficient, 68.20036)

        cp2 = Callpath('met3'), Metric('')
        self.assertIsInstance(models[cp2].hypothesis, ConstantHypothesis)
        self.assertAlmostEqual(models[cp2].hypothesis.function.constant_coefficient, 71.21204)

        cp3 = Callpath('met4'), Metric('')
        self.assertIsInstance(models[cp3].hypothesis, SingleParameterHypothesis)
        self.assertEqual(len(models[cp3].hypothesis.function.compound_terms), 1)
        self.assertEqual(len(models[cp3].hypothesis.function.compound_terms[0].simple_terms), 1)
        self.assertEqual(models[cp3].hypothesis.function.compound_terms[0].simple_terms[0].term_type, 'polynomial')
        self.assertAlmostEqual(models[cp3].hypothesis.function.compound_terms[0].simple_terms[0].exponent, 2.0)

    def test_default_multi_parameter_modeling(self):
        import logging
        logging.basicConfig(level=logging.DEBUG)
        files = ['data/text/two_parameter_1.txt',
                 "data/text/three_parameter_1.txt",
                 "data/text/three_parameter_2.txt",
                 "data/text/three_parameter_3.txt"]
        for f in files:
            experiment = TextFileReader().read_experiment(f)

            modeler = MultiParameterModeler()
            modeler.single_parameter_modeler = AdaptiveModeler()
            modeler.single_parameter_modeler.preset = 2

            # initialize model generator
            model_generator = ModelGenerator(experiment, modeler)

            # create models from data
            model_generator.model_all()

    def test_modeling(self):
        for exponents in [(3, 4, 0), (1, 1, 1), (3, 2, 2), (2, 1, 0)]:
            term = CompoundTerm.create(*exponents)
            term.coefficient = 10
            function = SingleParameterFunction(term)
            function.constant_coefficient = 200
            points = [2, 4, 8, 16, 32]

            values = function.evaluate(np.array(points))
            measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(points, values)]
            points = [10, 20, 30, 40, 50]

            values = function.evaluate(np.array(points))
            measurements2 = [Measurement(Coordinate(p), None, None, v) for p, v in zip(points, values)]
            modeler = AdaptiveModeler()

            models = modeler.model([measurements, measurements2])
            self.assertEqual(2, len(models))

    def test_noise_category(self):
        modeler = AdaptiveModeler()
        modeler.noise_aware = False
        for nc in adaptive._NOISE_CATEGORIES:
            self.assertEqual(0.2, modeler._noise_category(nc))
        modeler.noise_aware = True
        for nc in adaptive._NOISE_CATEGORIES:
            self.assertEqual(nc, modeler._noise_category(nc))
        for nc in adaptive._NOISE_CATEGORIES:
            self.assertEqual(nc, modeler._noise_category(np.array([nc])))
