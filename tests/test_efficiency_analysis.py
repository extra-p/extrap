# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2026, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

import sympy
from sympy import Rational

from extrap.entities.callpath import Callpath
from extrap.entities.experiment import Experiment
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.functions import MultiParameterFunction, ConstantFunction
from extrap.entities.hypotheses import ConstantHypothesis, MultiParameterHypothesis
from extrap.entities.measurement import Measure
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.entities.parameter import Parameter
from extrap.entities.terms import MultiParameterTerm, CompoundTerm
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.modelers.model_generator import ModelGenerator
from extrap.modelers.postprocessing.analysis.parallel_efficiency_analysis import ParallelEfficiencyAnalysis
from extrap.util.sympy_functions import log2


class TestEfficiencyAnalysis(unittest.TestCase):
    def test_efficiency_from_file(self):
        experiment = TextFileReader().read_experiment("data/text/two_parameter_7.txt")
        mg = ModelGenerator(experiment)
        mg.model_all()
        analysis = ParallelEfficiencyAnalysis(experiment)
        analysis.resource_parameter = Parameter('p')
        model_set = mg.post_process(analysis)

        for (callpath, _), model in model_set.models.items():
            if callpath.name.startswith("flat"):
                self.assertIsInstance(model.hypothesis, ConstantHypothesis)
                self.assertEqual(model.hypothesis.function.constant_coefficient, 1)
            elif callpath.name.startswith("main"):
                self.assertIsInstance(model.hypothesis, MultiParameterHypothesis)
                self.assertIsInstance(model.hypothesis.function, ComputationFunction)
                simplified = sympy.nsimplify(model.hypothesis.function.sympy_function)
                self.assertEqual(sympy.nsimplify((1 + 0.05 * ComputationFunction.get_param(1) ** Rational(3, 2)) / (
                        1 + 0.05 * ComputationFunction.get_param(
                    1) ** Rational(3, 2) * ComputationFunction.get_param(0))), simplified)

            elif callpath.name.startswith("compute"):
                self.assertIsInstance(model.hypothesis, MultiParameterHypothesis)
                self.assertIsInstance(model.hypothesis.function, ComputationFunction)
                self.assertEqual(sympy.nsimplify(2 / (
                        2 + 0.884 * ComputationFunction.get_param(1) * log2(ComputationFunction.get_param(0)))
                                                 ), sympy.nsimplify(model.hypothesis.function.sympy_function))
            else:
                self.fail("Unknown callpath: %s" % callpath)

    def test_efficiency(self):
        experiment = Experiment()
        experiment.parameters = [Parameter('p'), Parameter('q'), Parameter('s')]
        mg = ModelGenerator(experiment)
        mg.models = {
            (Callpath('test1'), Metric('met1')): Model(
                MultiParameterHypothesis(MultiParameterFunction(MultiParameterTerm((1, CompoundTerm.create(2, 1, 0)))),
                                         Measure.MEAN)),
            (Callpath('test2'), Metric('met1')): Model(
                MultiParameterHypothesis(MultiParameterFunction(MultiParameterTerm((0, CompoundTerm.create(2, 1, 0)))),
                                         Measure.MEAN)),
            (Callpath('test3'), Metric('met1')): Model(
                MultiParameterHypothesis(MultiParameterFunction(MultiParameterTerm((1, CompoundTerm.create(2, 1, 0))),
                                                                MultiParameterTerm((0, CompoundTerm.create(2, 1, 0)))),
                                         Measure.MEAN)),
            (Callpath('test4'), Metric('met1')): Model(
                MultiParameterHypothesis(MultiParameterFunction(
                    MultiParameterTerm((1, CompoundTerm.create(2, 1, 0)), (0, CompoundTerm.create(2, 1, 0)))),
                    Measure.MEAN)),
            (Callpath('test5'), Metric('met1')): Model(
                MultiParameterHypothesis(MultiParameterFunction(
                    MultiParameterTerm((1, CompoundTerm.create(2, 1, 0)), (2, CompoundTerm.create(2, 1, 0)),
                                       (0, CompoundTerm.create(2, 1, 0)))),
                    Measure.MEAN))
            ,
            (Callpath('test6'), Metric('met1')): Model(ConstantHypothesis(ConstantFunction(0),
                                                                          Measure.MEAN))
            ,
            (Callpath('test7'), Metric('met1')): Model(ConstantHypothesis(ConstantFunction(1),
                                                                          Measure.MEAN))
        }
        analysis = ParallelEfficiencyAnalysis(experiment)
        analysis.resource_parameter = Parameter('p')
        model_set = mg.post_process(analysis)

        model = model_set.models[(Callpath('test1'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ComputationFunction)
        self.assertEqual(model.hypothesis.function.sympy_function, 1)

        model = model_set.models[(Callpath('test2'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ComputationFunction)
        self.assertEqual(model.hypothesis.function.sympy_function, 1 / ComputationFunction.get_param(0) ** 2)

        model = model_set.models[(Callpath('test3'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ComputationFunction)
        self.assertEqual(model.hypothesis.function.sympy_function, (1 + ComputationFunction.get_param(1) ** 2) / (
                ComputationFunction.get_param(0) ** 2 + ComputationFunction.get_param(1) ** 2))

        model = model_set.models[(Callpath('test4'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ComputationFunction)
        self.assertEqual(model.hypothesis.function.sympy_function, 1 / ComputationFunction.get_param(0) ** 2)

        model = model_set.models[(Callpath('test5'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ComputationFunction)
        self.assertEqual(model.hypothesis.function.sympy_function, 1 / ComputationFunction.get_param(0) ** 2)

        model = model_set.models[(Callpath('test6'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ConstantFunction)
        self.assertEqual(model.hypothesis.function.constant_coefficient, 1)

        model = model_set.models[(Callpath('test7'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ConstantFunction)
        self.assertEqual(model.hypothesis.function.constant_coefficient, 1)

    def test_efficiency_with_log(self):
        experiment = Experiment()
        experiment.parameters = [Parameter('p'), Parameter('q'), Parameter('s')]
        mg = ModelGenerator(experiment)
        mg.models = {
            (Callpath('test1a'), Metric('met1')): Model(
                MultiParameterHypothesis(MultiParameterFunction(MultiParameterTerm((1, CompoundTerm.create(2, 1, 1)))),
                                         Measure.MEAN)),
            (Callpath('test2a'), Metric('met1')): Model(
                MultiParameterHypothesis(MultiParameterFunction(MultiParameterTerm((0, CompoundTerm.create(2, 1, 1)))),
                                         Measure.MEAN)),
            (Callpath('test3a'), Metric('met1')): Model(
                MultiParameterHypothesis(MultiParameterFunction(MultiParameterTerm((1, CompoundTerm.create(2, 1, 1))),
                                                                MultiParameterTerm((0, CompoundTerm.create(2, 1, 1)))),
                                         Measure.MEAN)),
            (Callpath('test4a'), Metric('met1')): Model(
                MultiParameterHypothesis(MultiParameterFunction(
                    MultiParameterTerm((1, CompoundTerm.create(2, 1, 1)), (0, CompoundTerm.create(2, 1, 1)))),
                    Measure.MEAN)),
            (Callpath('test5a'), Metric('met1')): Model(
                MultiParameterHypothesis(MultiParameterFunction(
                    MultiParameterTerm((1, CompoundTerm.create(2, 1, 1)), (2, CompoundTerm.create(2, 1, 1)),
                                       (0, CompoundTerm.create(2, 1, 1)))),
                    Measure.MEAN))
        }

        for m in mg.models.values():
            m.hypothesis.function.constant_coefficient = 5

        analysis = ParallelEfficiencyAnalysis(experiment)
        analysis.resource_parameter = Parameter('p')
        model_set = mg.post_process(analysis)

        model = model_set.models[(Callpath('test1a'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ComputationFunction)
        self.assertEqual(model.hypothesis.function.sympy_function, 1)

        model = model_set.models[(Callpath('test2a'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ComputationFunction)
        self.assertEqual(model.hypothesis.function.sympy_function,
                         5 / (5 + ComputationFunction.get_param(0) ** 2 * log2(ComputationFunction.get_param(0))))

        model = model_set.models[(Callpath('test3a'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ComputationFunction)
        self.assertEqual(model.hypothesis.function.sympy_function, (
                (5 + ComputationFunction.get_param(1) ** 2 * log2(ComputationFunction.get_param(1)))
                / (5 + ComputationFunction.get_param(0) ** 2 * log2(ComputationFunction.get_param(0))
                   + ComputationFunction.get_param(1) ** 2 * log2(ComputationFunction.get_param(1)))))

        model = model_set.models[(Callpath('test4a'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ComputationFunction)
        self.assertEqual(5 / (5 + ComputationFunction.get_param(0) ** 2 * log2(
            ComputationFunction.get_param(0)) * ComputationFunction.get_param(1) ** 2 * log2(
            ComputationFunction.get_param(1))),
                         model.hypothesis.function.sympy_function)

        model = model_set.models[(Callpath('test5a'), Metric('met1'))]
        self.assertIsInstance(model.hypothesis.function, ComputationFunction)
        self.assertEqual(5 / (5 + ComputationFunction.get_param(0) ** 2 * log2(
            ComputationFunction.get_param(0)) * ComputationFunction.get_param(1) ** 2 * log2(
            ComputationFunction.get_param(1)) * ComputationFunction.get_param(2) ** 2 * log2(
            ComputationFunction.get_param(2))), model.hypothesis.function.sympy_function)
