# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import math
import unittest
from math import log2

import numpy
import numpy as np

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.functions import SingleParameterFunction, SegmentedFunction
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.model import Model, SegmentedModel
from extrap.entities.parameter import Parameter
from extrap.entities.terms import CompoundTerm
from extrap.modelers.model_generator import ModelGenerator
from extrap.modelers.single_parameter.segmented import SegmentedModeler
from extrap.util.progress_bar import ProgressBar
from tests.modelling_testcase import TestCaseWithFunctionAssertions


def create_experiment(g, f, parameter_values, changing_point):
    parameter = Parameter("p")
    metric = Metric("runtime")
    callpath = Callpath("main")
    experiment = Experiment()
    experiment.add_callpath(callpath)
    experiment.add_metric(metric)
    experiment.add_parameter(parameter)
    for i in range(len(parameter_values)):
        coordinate = Coordinate(parameter_values[i])
        experiment.add_coordinate(coordinate)
        p = parameter_values[i]
        if p >= changing_point:
            metric_value = g(p)
        else:
            metric_value = f(p)
        experiment.add_measurement(Measurement(coordinate, callpath, metric, metric_value))
    return experiment


def get_segmented_model(experiment):
    model_generator = ModelGenerator(experiment, modeler="SEGMENTED", name="Segmented", use_median=True)
    with ProgressBar(desc='Generating models', disable=True) as pbar:
        model_generator.model_all(pbar)
    modeler = experiment.modelers[0]
    models = modeler.models
    model = models[(Callpath("main"), Metric("runtime"))]
    functions = []
    if isinstance(model, SegmentedModel):
        for m in model.segment_models:
            func = m.hypothesis.function
            functions.append(func)
    elif isinstance(model, Model):
        func = model.hypothesis.function
        functions.append(func)
    else:
        raise NotImplementedError()
    return functions


class TestSegmentedModeler(TestCaseWithFunctionAssertions):

    def test_segmented_data_one(self):
        parameter_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        f = lambda p: p ** 2
        g = lambda p: 30 + p
        changing_point = 6

        experiment = create_experiment(g, f, parameter_values, changing_point)
        functions = get_segmented_model(experiment)

        self.assertEqual(len(functions), 2)

        term = CompoundTerm.create(2, 0)
        function = SingleParameterFunction(term)
        function.constant_coefficient = 0

        term2 = CompoundTerm.create(1, 0)
        term2.coefficient = 1.0
        function2 = SingleParameterFunction(term2)
        function2.constant_coefficient = 30

        self.assertApproxFunction(function, functions[0])
        self.assertApproxFunction(function2, functions[1])

    def test_segmented_data_one_minimal_number_of_measurements(self):
        parameter_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        f = lambda p: p ** 2
        g = lambda p: 50 + 4 * p
        changing_point = 5

        experiment = create_experiment(g, f, parameter_values, changing_point)
        functions = get_segmented_model(experiment)

        term = CompoundTerm.create(2, 0)
        function = SingleParameterFunction(term)
        function.constant_coefficient = 0

        term2 = CompoundTerm.create(1, 0)
        term2.coefficient = 4
        function2 = SingleParameterFunction(term2)
        function2.constant_coefficient = 50

        self.assertApproxFunction(function, functions[0])
        self.assertApproxFunction(function2, functions[1])

    def test_segmented_data_two(self):
        parameter_values = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
        f = lambda p: log2(p) ** 1
        g = lambda p: p ** 2
        changing_point = 22

        experiment = create_experiment(g, f, parameter_values, changing_point)
        functions = get_segmented_model(experiment)

        self.assertEqual(len(functions), 2)

        term = CompoundTerm.create(0, 1, c=1)
        function = SingleParameterFunction(term)
        function.constant_coefficient = 0

        term = CompoundTerm.create(2, 0)
        term.coefficient = 1.0
        function2 = SingleParameterFunction(term)
        function2.constant_coefficient = 0

        self.assertApproxFunction(function, functions[0])
        self.assertApproxFunction(function2, functions[1])

    def test_no_segmentation(self):
        parameter_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        f = lambda p: p ** 2 * log2(p) ** 1
        g = lambda p: p ** 2 * log2(p) ** 1
        changing_point = 50

        experiment = create_experiment(g, f, parameter_values, changing_point)
        functions = get_segmented_model(experiment)

        term = CompoundTerm.create(2, 1, c=1)
        term.coefficient = 1.0
        function = SingleParameterFunction(term)
        function.constant_coefficient = 0

        self.assertApproxFunction(function, functions[0])

    def test_exmple_two(self):
        parameter_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        f = lambda p: 67 + 63 * p ** (1 / 4) * log2(p) ** 2
        g = lambda p: 93 + 64 * p ** (5 / 3)
        changing_point = 6

        experiment = create_experiment(g, f, parameter_values, changing_point)
        functions = get_segmented_model(experiment)

        self.assertEqual(len(functions), 2)

        term = CompoundTerm.create(1, 4, c=2)
        term.coefficient = 63
        function = SingleParameterFunction(term)
        function.constant_coefficient = 67

        term = CompoundTerm.create(5, 3, 0)
        term.coefficient = 64
        function2 = SingleParameterFunction(term)
        function2.constant_coefficient = 93

        self.assertApproxFunction(function, functions[0])
        self.assertApproxFunction(function2, functions[1])

    def test_const_square_segmentation(self):
        measurements = [Measurement(Coordinate(500), None, None, [4.1, 3.9, 4.0, 4.0, 4.1]),
                        Measurement(Coordinate(1000), None, None, [4.1, 3.9, 4.0, 4.0, 4.1]),
                        Measurement(Coordinate(2000), None, None, [4.1, 3.9, 4.0, 4.0, 4.1]),
                        Measurement(Coordinate(4000), None, None, [16, 15.999, 16.01, 16.01, 15.99]),
                        Measurement(Coordinate(8000), None, None, [64, 64, 64, 64.01, 63.99]),
                        Measurement(Coordinate(16000), None, None, [256.01, 255.99, 256, 256]),
                        ]
        model = SegmentedModeler().model([measurements])
        self.assertEqual(1, len(model))
        self.assertFalse(isinstance(model[0].hypothesis.function, SegmentedFunction))

        modeler = SegmentedModeler()
        modeler.min_measurement_points = 3

        model = modeler.model([measurements])
        self.assertEqual(1, len(model))
        self.assertTrue(isinstance(model[0].hypothesis.function, SegmentedFunction))

    def test_linear_square_segmentation(self):
        measurements = [Measurement(Coordinate(c), None, None, [c * 2]) for c in range(7)] + [
            Measurement(Coordinate(c), None, None, [c ** 2 + 7 * 2]) for c in range(7, 15)]

        modeler = SegmentedModeler()

        model = modeler.model([measurements])
        self.assertEqual(1, len(model))
        function = model[0].hypothesis.function
        self.assertTrue(isinstance(function, SegmentedFunction))
        for m in measurements:
            self.assertApprox(m.mean, function.evaluate(m.coordinate))

    def test_unordered_measurements(self):
        m1 = [Measurement(Coordinate(c), None, None, [c * 2]) for c in range(7)]
        m2 = [Measurement(Coordinate(c), None, None, [c ** 2 + 7 * 2]) for c in range(7, 15)]

        measurements = [val for pair in zip(m1, m2) for val in pair]

        modeler = SegmentedModeler()

        model = modeler.model([measurements])
        self.assertEqual(1, len(model))
        function = model[0].hypothesis.function
        self.assertTrue(isinstance(function, SegmentedFunction))
        for m in measurements:
            self.assertApprox(m.mean, function.evaluate(m.coordinate))

    def test_segmented_function(self):
        segments = [SingleParameterFunction(CompoundTerm.create(1, 1, 0)),
                    SingleParameterFunction(CompoundTerm.create(2, 1, 0)),
                    SingleParameterFunction(CompoundTerm.create(3, 1, 0))]
        change_points = [(-math.inf, 1), (1, 5), (5, math.inf)]

        class SegmentedTestFunction(SegmentedFunction):
            MAX_NUM_SEGMENTS = 3

        f = SegmentedTestFunction(segments, change_points)
        res = f.evaluate(np.array([0.5, 1, 2, 8]))
        numpy.testing.assert_array_equal(np.array([0.5, 1, 4, 512]), res)


if __name__ == '__main__':
    unittest.main()
