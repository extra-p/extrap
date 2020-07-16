import copy
import unittest
from typing import Dict

from entities.callpath import Callpath
from entities.experiment import Experiment
from entities.functions import SingleParameterFunction
from entities.hypotheses import ConstantHypothesis, SingleParameterHypothesis
from entities.metric import Metric
from entities.model import Model
from entities.parameter import Parameter
from entities.terms import SimpleTerm, CompoundTerm
from fileio.text_file_reader import read_text_file
from modelers.model_generator import ModelGenerator
from modelers.single_parameter.basic import SingleParameterModeler
from modelers.single_parameter.refining import RefiningModeler


class TestRefiningModeler(unittest.TestCase):

    def test_general(self):
        experiment = read_text_file('data/text/one_parameter_6.txt')
        # initialize model generator
        model_generator = ModelGenerator(experiment, RefiningModeler())

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
