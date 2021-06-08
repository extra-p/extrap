import numpy as np

from extrap.entities.terms import CompoundTerm
from extrap.modelers.aggregation.abstract_binary_aggregation import BinaryAggregationFunction
import numpy
from typing import Callable, Union, List, Tuple, Dict, Sequence, Mapping
from extrap.entities.parameter import Parameter


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
