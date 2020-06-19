import unittest
from fileio.text_file_reader import read_text_file
from entities.model_generator import ModelGenerator
from entities.constant_hypothesis import ConstantHypothesis
from entities.single_parameter_hypothesis import SingleParameterHypothesis
from entities.experiment import Experiment


class Test_TestModeling(unittest.TestCase):

    def test_basic_single_parameter_modeling(self):
        experiment = read_text_file('data/text/one_parameter_6.txt')

        # initialize model generator
        model_generator = ModelGenerator(experiment)

        # create models from data
        experiment: Experiment = model_generator.model_all(False)

        models = experiment.modeler[0].models
        self.assertIsInstance(models[0].hypothesis, ConstantHypothesis)
        self.assertAlmostEqual(models[0].hypothesis.function.constant_coefficient, 4.068)

        self.assertIsInstance(models[1].hypothesis,  SingleParameterHypothesis)
        self.assertEqual(len(models[1].hypothesis.function.compound_terms), 1)
        self.assertEqual(len(models[1].hypothesis.function.compound_terms[0].simple_terms), 1)
        self.assertEqual(models[1].hypothesis.function.compound_terms[0].simple_terms[0].term_type, 'polynomial')
        self.assertAlmostEqual(models[1].hypothesis.function.compound_terms[0].simple_terms[0].exponent, 2.0)

        self.assertIsInstance(models[2].hypothesis,  SingleParameterHypothesis)
        self.assertEqual(len(models[2].hypothesis.function.compound_terms), 1)
        self.assertEqual(len(models[2].hypothesis.function.compound_terms[0].simple_terms), 1)
        self.assertEqual(models[2].hypothesis.function.compound_terms[0].simple_terms[0].term_type, 'polynomial')
        self.assertAlmostEqual(models[2].hypothesis.function.compound_terms[0].simple_terms[0].exponent, 2.0)

        self.assertIsInstance(models[3].hypothesis,  SingleParameterHypothesis)
        self.assertEqual(len(models[3].hypothesis.function.compound_terms), 1)
        self.assertEqual(len(models[3].hypothesis.function.compound_terms[0].simple_terms), 1)
        self.assertEqual(models[3].hypothesis.function.compound_terms[0].simple_terms[0].term_type, 'polynomial')
        self.assertAlmostEqual(models[3].hypothesis.function.compound_terms[0].simple_terms[0].exponent, 2.0)
