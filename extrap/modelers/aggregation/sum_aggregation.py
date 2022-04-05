# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Sequence

import sympy

from extrap.entities.callpath import Callpath
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.modelers.aggregation import AggregatedModel
from extrap.modelers.aggregation.abstract_binary_aggregation import BinaryAggregationFunction, BinaryAggregation, \
    BinaryAggregationFunctionSchema


class SumAggregationFunction(BinaryAggregationFunction):
    def __init__(self, *function_terms):
        super().__init__(*function_terms)

    def aggregate(self):
        if not self.raw_terms:
            return
        _, self._ftype = self._determine_params(self.raw_terms[0])
        preprocessed_terms = []
        for function in self.raw_terms:
            _params, ftype = self._determine_params(function)
            if self._ftype != ftype:
                raise ValueError("You cannot aggregate single and multi parameter functions to one function.")
            self._ftype &= ftype
            if isinstance(function, ComputationFunction):
                preprocessed_terms.append(function.sympy_function)
            else:
                preprocessed_terms.append(function.evaluate(_params))
        self.sympy_function = sympy.Add(*preprocessed_terms)


class SumAggregationFunctionSchema(BinaryAggregationFunctionSchema):
    def create_object(self):
        return SumAggregationFunction([])


class SumAggregation(BinaryAggregation):
    NAME = 'Sum'

    def binary_operator(self, a, b):
        return a + b

    def aggregate_model(self, agg_models, callpath: Callpath, measurements: Sequence[Measurement], metric: Metric):
        function = SumAggregationFunction([m.hypothesis.function for m in agg_models])
        hypothesis_type = type(agg_models[0].hypothesis)
        hypothesis = hypothesis_type(function, agg_models[0].hypothesis._use_median)
        hypothesis.compute_cost(measurements)
        model = AggregatedModel(hypothesis, callpath, metric)
        return model

    TAG_DISABLED = 'agg__disabled__sum'
    TAG_USAGE_DISABLED = 'agg__usage_disabled__sum'
    _OPERATION_NAME = 'sum'
