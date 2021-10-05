# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Union, Sequence

import numpy
import numpy as np

from extrap.entities.callpath import Callpath
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.modelers.aggregation import AggregatedModel
from extrap.modelers.aggregation.abstract_binary_aggregation import BinaryAggregationFunction, BinaryAggregation, \
    BinaryAggregationFunctionSchema


class MaxAggregationFunction(BinaryAggregationFunction):

    def evaluate(self, parameter_value):
        rest = iter(self.raw_terms)
        function_value = next(rest).evaluate(parameter_value)

        if isinstance(parameter_value, numpy.ndarray):
            shape = parameter_value.shape
            if len(shape) == 2:
                shape = (shape[1],)
            function_value += numpy.zeros(shape, dtype=float)

        for t in rest:
            function_value = np.maximum(function_value, t.evaluate(parameter_value))
        return function_value

    def aggregate(self):
        self.constant_coefficient = 0
        self.compound_terms = []

    def to_string(self, *parameters: Union[str, Parameter]):
        function_string = "Max(" + str(self.constant_coefficient)
        for t in self.raw_terms:
            function_string += ' , '
            function_string += t.to_string(*parameters)
        function_string += ")"
        return function_string


class MaxAggregationFunctionSchema(BinaryAggregationFunctionSchema):
    def create_object(self):
        return MaxAggregationFunction(None)


class MaxAggregation(BinaryAggregation):
    NAME = 'Max'

    def binary_operator(self, a, b):
        return np.maximum(a, b)

    def aggregate_model(self, agg_models, callpath: Callpath, measurements: Sequence[Measurement], metric: Metric):
        function = MaxAggregationFunction([m.hypothesis.function for m in agg_models])
        hypothesis_type = type(agg_models[0].hypothesis)
        hypothesis = hypothesis_type(function, agg_models[0].hypothesis._use_median)
        hypothesis.compute_cost(measurements)
        model = AggregatedModel(hypothesis, callpath, metric)
        return model

    TAG_DISABLED = 'agg__disabled__max'
    TAG_USAGE_DISABLED = 'agg__usage_disabled__max'
    _OPERATION_NAME = 'max'
