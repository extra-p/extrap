# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from math import log2
from extrap.entities.parameter import Parameter
from extrap.entities.metric import Metric
from extrap.entities.callpath import Callpath
from extrap.entities.experiment import Experiment
from extrap.entities.coordinate import Coordinate
from extrap.entities.measurement import Measurement
from extrap.util.progress_bar import ProgressBar
from extrap.modelers.model_generator import ModelGenerator
from extrap.entities.model import Model
from tests.modelling_testcase import TestCaseWithFunctionAssertions
import unittest
from extrap.entities.functions import SingleParameterFunction
from extrap.entities.terms import CompoundTerm


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
            metric_value = eval(g)
        else:
            metric_value = eval(f)
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
    if isinstance(model, Model):
        func = model.hypothesis.function
        functions.append(func)
    else:
        for m in model:
            func = m.hypothesis.function
            functions.append(func)
    return functions


class TestSegmentedModeler(TestCaseWithFunctionAssertions):
    
    def test_segmented_data_one(self):
        parameter_values = [1,2,3,4,5,6,7,8,9,10]
        f = "p**2"
        g = "30+p"
        changing_point = 6

        experiment = create_experiment(g, f, parameter_values, changing_point)
        functions = get_segmented_model(experiment)

        term = CompoundTerm.create(2,0)
        function = SingleParameterFunction(term)
        function.constant_coefficient = -5.753052622747037e-16

        term2 = CompoundTerm.create(1,0)
        term2.coefficient = 1.0
        function2 = SingleParameterFunction(term2)
        function2.constant_coefficient = 29.999999999999975

        self.assertApproxFunction(function, functions[0])
        self.assertApproxFunction(function2, functions[1])
        
    def test_segmented_data_two(self):
        parameter_values = [4,8,12,16,20,24,28,32,36,40]
        f = "log2(p)**1"
        g = "p**2"
        changing_point = 22

        experiment = create_experiment(g, f, parameter_values, changing_point)
        functions = get_segmented_model(experiment)

        term = CompoundTerm.create(0,1,c=1)
        function = SingleParameterFunction(term)
        function.constant_coefficient = -1.1467601473192458e-16

        term = CompoundTerm.create(2,0)
        term.coefficient = 1.0
        function2 = SingleParameterFunction(term)
        function2.constant_coefficient = 4.258612552374011e-13

        self.assertApproxFunction(function, functions[0])
        self.assertApproxFunction(function2, functions[1])
        
    def test_no_segmentation(self):
        parameter_values = [10,20,30,40,50,60,70,80,90,100]
        f = "p**2*log2(p)**1"
        g = "p**2*log2(p)**1"
        changing_point = 50

        experiment = create_experiment(g, f, parameter_values, changing_point)
        functions = get_segmented_model(experiment)

        term = CompoundTerm.create(2,1,c=1)
        term.coefficient = 1.0
        function = SingleParameterFunction(term)
        function.constant_coefficient = 1.2179330821573644e-11

        self.assertApproxFunction(function, functions[0])

if __name__ == '__main__':
    unittest.main()