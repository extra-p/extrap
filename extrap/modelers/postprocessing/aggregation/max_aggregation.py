# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Sequence

import numpy as np
import sympy
from marshmallow import pre_dump

from extrap.entities.callpath import Callpath
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.modelers.postprocessing.aggregation import AggregatedModel
from extrap.modelers.postprocessing.aggregation.abstract_binary_aggregation import BinaryAggregationFunction, \
    BinaryAggregation, \
    BinaryAggregationFunctionSchema, BinaryAggregationSchema


class MaxAggregationFunction(BinaryAggregationFunction):

    def aggregate(self):
        if not self.raw_terms:
            return
        sym_functions = []
        _, self._ftype = self._determine_params(self.raw_terms[0])
        for function in self.raw_terms:
            _params, ftype = self._determine_params(function)
            if self._ftype != ftype:
                raise ValueError("You cannot process single and multi parameter functions to one function.")
            self._ftype &= ftype
            if isinstance(function, ComputationFunction):
                sym_functions.append(function.sympy_function)
            else:
                sym_functions.append(function.evaluate(_params))
        self.sympy_function = sympy.Max(*sym_functions)


class MaxAggregationFunctionSchema(BinaryAggregationFunctionSchema):
    def create_object(self):
        return MaxAggregationFunction([])

    @pre_dump
    def intercept(self, data, **kwargs):
        return data


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


class MaxAggregationSchema(BinaryAggregationSchema):
    def create_object(self):
        return MaxAggregation(None)
