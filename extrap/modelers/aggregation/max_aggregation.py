from extrap.entities.terms import CompoundTerm
from extrap.modelers.aggregation.abstract_binary_aggregation import BinaryAggregationFunction
import numpy
from typing import Callable, Union, List, Tuple, Dict, Sequence, Mapping
from extrap.entities.parameter import Parameter


class MaxAggregationFunction(BinaryAggregationFunction):

    def evaluate(self, parameter_value):
        if hasattr(parameter_value, '__len__') and (len(parameter_value) == 1 or isinstance(parameter_value, Mapping)):
            parameter_value = parameter_value[0]

        if isinstance(parameter_value, numpy.ndarray):
            shape = parameter_value.shape
            if len(shape) == 2:
                shape = (shape[1],)
            function_value = numpy.full(shape, self.constant_coefficient, dtype=float)
        else:
            function_value = self.constant_coefficient

        for t in self.raw_terms:
            function_value += t.evaluate(parameter_value)

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
