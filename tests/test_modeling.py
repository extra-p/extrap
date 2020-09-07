import unittest
from operator import itemgetter

from extrap.entities.callpath import Callpath
from extrap.entities.hypotheses import ConstantHypothesis
from extrap.entities.hypotheses import SingleParameterHypothesis
from extrap.entities.metric import Metric
from extrap.entities.terms import CompoundTerm, SimpleTerm, MultiParameterTerm
from extrap.fileio.text_file_reader import read_text_file
from extrap.modelers.model_generator import ModelGenerator


class TestModeling(unittest.TestCase):

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


class TestCaseWithFunctionAssertions(unittest.TestCase):
    def assertApprox(self, function, other, places=5):
        import math
        diff = abs(other - function)
        reference = min(abs(function), abs(other))
        if reference != 0:
            nondecimal_places = int(math.log10(reference))
            diff_scaled = diff / (10 ** nondecimal_places)
        else:
            diff_scaled = diff
        diff_rounded = round(diff_scaled, places)
        self.assertTrue(diff_rounded == 0, msg=f"{other} != {function} in {places} places")

    def assertApproxFunction(self, function, other, **kwargs):
        if len(kwargs) == 0:
            kwargs['places'] = 5
        self.assertApprox(function.constant_coefficient, other.constant_coefficient, **kwargs)
        self.assertEqual(len(function.compound_terms), len(other.compound_terms))
        for tt, to in zip(function.compound_terms, other.compound_terms):
            self.assertApproxTerm(tt, to, **kwargs)

    def assertApproxTerm(self, tt: CompoundTerm, to: CompoundTerm, **kwargs):
        self.assertApprox(tt.coefficient, to.coefficient, **kwargs)
        if isinstance(tt, CompoundTerm):
            self.assertEqual(len(tt.simple_terms), len(to.simple_terms))
            for stt, sto in zip(tt.simple_terms, to.simple_terms):
                self.assertApproxSimpleTerm(stt, sto, **kwargs)
        elif isinstance(tt, MultiParameterTerm):
            self.assertEqual(len(tt.parameter_term_pairs), len(to.parameter_term_pairs))
            for stt, sto in zip(sorted(tt.parameter_term_pairs, key=itemgetter(0)),
                                sorted(to.parameter_term_pairs, key=itemgetter(0))):
                self.assertEqual(stt[0], sto[0])
                self.assertApproxTerm(stt[1], sto[1], **kwargs)

    def assertApproxSimpleTerm(self, stt: SimpleTerm, sto: SimpleTerm, **kwargs):
        self.assertEqual(stt.term_type, sto.term_type)
        self.assertApprox(stt.exponent, sto.exponent, **kwargs)
