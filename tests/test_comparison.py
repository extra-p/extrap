from unittest import TestCase

import numpy as np

from extrap.comparison.experiment_comparison import ComparisonExperiment
from extrap.comparison.matcher import MinimumMatcher
from extrap.fileio.text_file_reader import read_text_file
from extrap.modelers.model_generator import ModelGenerator


class TestComparison(TestCase):
    def test_identity_comparison(self):
        experiment1 = read_text_file('data/text/one_parameter_6.txt')
        ModelGenerator(experiment1).model_all()
        experiment2 = read_text_file('data/text/one_parameter_6.txt')
        ModelGenerator(experiment2).model_all()
        experiment = ComparisonExperiment(experiment1, experiment2, MinimumMatcher())
        model = next(iter(experiment.modelers[0].models.values()))
        model.hypothesis.function.evaluate(np.array([12, 22, 32, 42]))
