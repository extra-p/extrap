# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extrap.entities.callpath import Callpath
from extrap.entities.hypotheses import ConstantHypothesis
from extrap.entities.hypotheses import SingleParameterHypothesis
from extrap.entities.metric import Metric
from extrap.fileio.text_file_reader import read_text_file
from extrap.modelers.model_generator import ModelGenerator
from tests.modelling_testcase import TestCaseWithFunctionAssertions


class TestModeling(TestCaseWithFunctionAssertions):

    def test_default_single_parameter_modeling(self):
        experiment = read_text_file('data/text/one_parameter_6.txt')

        # initialize model generator
        model_generator = ModelGenerator(experiment)

        # create models from data
        model_generator.model_all()

        models = experiment.modelers[0].models
        cp0 = Callpath('met1'), Metric('')
        self.assertIsInstance(models[cp0].hypothesis, ConstantHypothesis)
        self.assertAlmostEqual(models[cp0].hypothesis.function.constant_coefficient, 4.068)

        cp1 = Callpath('met2'), Metric('')
        self.assertIsInstance(models[cp1].hypothesis, SingleParameterHypothesis)
        self.assertEqual(len(models[cp1].hypothesis.function.compound_terms), 1)
        self.assertEqual(len(models[cp1].hypothesis.function.compound_terms[0].simple_terms), 1)
        self.assertEqual(models[cp1].hypothesis.function.compound_terms[0].simple_terms[0].term_type, 'polynomial')
        self.assertAlmostEqual(models[cp1].hypothesis.function.compound_terms[0].simple_terms[0].exponent, 2.0)

        cp2 = Callpath('met3'), Metric('')
        self.assertIsInstance(models[cp2].hypothesis, SingleParameterHypothesis)
        self.assertEqual(len(models[cp2].hypothesis.function.compound_terms), 1)
        self.assertEqual(len(models[cp2].hypothesis.function.compound_terms[0].simple_terms), 1)
        self.assertEqual(models[cp2].hypothesis.function.compound_terms[0].simple_terms[0].term_type, 'polynomial')
        self.assertAlmostEqual(models[cp2].hypothesis.function.compound_terms[0].simple_terms[0].exponent, 2.0)

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
            experiment = read_text_file(f)

            # initialize model generator
            model_generator = ModelGenerator(experiment)
            # create models from data
            model_generator.model_all()

    def test_two_parameter_modeling(self):
        import logging
        logging.basicConfig(level=logging.DEBUG)
        experiment = read_text_file("data/text/two_parameter_3.txt")

        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        model_generator.model_all()

        first = next(iter(experiment.modelers[0].models.values()))

        for model in experiment.modelers[0].models.values():
            self.assertApproxFunction(first.hypothesis.function, model.hypothesis.function)
            self.assertEqual(first.hypothesis, model.hypothesis)
