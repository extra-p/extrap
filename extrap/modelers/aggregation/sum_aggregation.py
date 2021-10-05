# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Union, Sequence, Mapping

import numpy

from extrap.entities.callpath import Callpath
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.entities.terms import CompoundTerm, MultiParameterTerm
from extrap.modelers.aggregation import AggregatedModel
from extrap.modelers.aggregation.abstract_binary_aggregation import BinaryAggregationFunction, BinaryAggregation, \
    BinaryAggregationFunctionSchema


class SumAggregationFunction(BinaryAggregationFunction):

    def evaluate(self, parameter_value):
        if isinstance(parameter_value, numpy.ndarray):
            shape = parameter_value.shape
            if len(shape) == 2:
                shape = (shape[1],)
            function_value = numpy.full(shape, self.constant_coefficient, dtype=float)
        else:
            function_value = self.constant_coefficient
        for t in self.compound_terms:
            if isinstance(t, MultiParameterTerm):
                function_value += t.evaluate(parameter_value)
            else:
                if hasattr(parameter_value, '__len__') and (
                        len(parameter_value) == 1 or isinstance(parameter_value, Mapping)):
                    value = parameter_value[0]
                else:
                    value = parameter_value
                function_value += t.evaluate(value)

        return function_value

    def aggregate(self):
        self.constant_coefficient = 0
        self.compound_terms = []
        term_map = {}

        for t in self.raw_terms:
            # aggregate constant term
            self.constant_coefficient += t.constant_coefficient

            # find immutable simple term combinations
            for x in t.compound_terms:
                if isinstance(x, MultiParameterTerm):
                    key = 'Multi' + ', '.join(str(y[0]) + y[1].to_string() for y in x.parameter_term_pairs)
                elif hasattr(x, 'simple_terms'):
                    key = ', '.join(y.to_string() for y in x.simple_terms)
                else:
                    key = 'Compat' + x.to_string()
                if term_map.keys().__contains__(key):
                    term_map[key].append(x)
                else:
                    term_map[key] = [x]

        # Aggregate a compound term for each immutable simple term combination
        for key in term_map.keys():
            if key.startswith('Compat'):
                self.add_compound_term(term_map[key][0])
                continue
            if key.startswith('Multi'):
                term = MultiParameterTerm()
                term.parameter_term_pairs = term_map[key][0].parameter_term_pairs
            else:
                term = CompoundTerm()
                term.simple_terms = term_map[key][0].simple_terms
            term.coefficient = 0
            for coeff in term_map[key]:
                term.coefficient += coeff.coefficient
            self.add_compound_term(term)

    def to_string(self, *parameters: Union[str, Parameter]):
        function_string = str(self.constant_coefficient)
        for t in self.compound_terms:
            function_string += ' + '
            function_string += t.to_string(*parameters)
        return function_string


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
