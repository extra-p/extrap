# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import abc
import copy
import operator
from abc import ABC
from collections import defaultdict
from inspect import signature
from itertools import permutations
from numbers import Number, Real
from typing import Union, Sequence, Callable, Iterable, Mapping, cast

from extrap.entities.hypotheses import Hypothesis
from extrap.entities.model import Model
from extrap.util.extension_loader import load_extensions

REQUIRED_METRICS_NAME = '_required_metrics'

try:
    from typing import Protocol
except ImportError:
    class Protocol:
        pass

import numpy

from extrap.entities.functions import Function, ConstantFunction
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.util.classproperty import classproperty


class CalculationElement(Protocol):
    def __add__(self, other) -> CalculationElement:
        pass

    def __sub__(self, other) -> CalculationElement:
        pass

    def __mul__(self, other) -> CalculationElement:
        pass

    def __truediv__(self, other) -> CalculationElement:
        pass

    def __radd__(self, other) -> CalculationElement:
        pass

    def __rmul__(self, other) -> CalculationElement:
        pass

    def __rsub__(self, other) -> CalculationElement:
        pass

    def __rtruediv__(self, other) -> CalculationElement:
        pass

    def __neg__(self) -> CalculationElement:
        pass


class CalculationFunction(Function):

    def __init__(self, *function: Function):
        super().__init__(*function)
        self._function = function[0]

    def to_string(self, *parameters: Union[str, Parameter]):
        result = self._function.to_string(*parameters)
        if self.constant_coefficient != 0:
            result = str(self.constant_coefficient) + " + " + result
        return result

    def evaluate(self, parameter_value: Union[Number, numpy.ndarray, Mapping[int, Union[Number, numpy.ndarray]],
                                              Sequence[Union[Number, numpy.ndarray]]]) -> Union[Number, numpy.ndarray]:
        return self._function.evaluate(parameter_value) + self.constant_coefficient

    def __repr__(self):
        return f"CalculatedFunction({self.to_string()})"

    def __add__(self, other):
        if isinstance(other, Function):
            target, other = self.unwrap_functions(self, other)
            if isinstance(other, ConstantFunction):
                return self.__add__(other.constant_coefficient)
            elif isinstance(target, ConstantFunction):
                result = copy.copy(other)
                result.constant_coefficient += target.constant_coefficient
                return CalculationFunction(result)
            return CalculatedFunctionAddition(target, other)
        elif isinstance(other, Real):
            if type(self) == CalculationFunction:
                assert self.constant_coefficient == 0
                result = copy.copy(self._function)
                result.constant_coefficient += other
                return CalculationFunction(result)
            else:
                result = copy.copy(self)
                result.constant_coefficient += other
                return result
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Function):
            target, other = self.unwrap_functions(self, other)
            if isinstance(other, ConstantFunction):
                return self.__sub__(other.constant_coefficient)
            return CalculatedFunctionSubtraction(target, other)
        elif isinstance(other, Real):
            if type(self) == CalculationFunction:
                assert self.constant_coefficient == 0
                result = copy.copy(self._function)
                result.constant_coefficient -= other
                return CalculationFunction(result)
            else:
                result = copy.copy(self)
                result.constant_coefficient -= other
                return result
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Function):
            target, other = self.unwrap_functions(self, other)
            if isinstance(other, ConstantFunction) and other.constant_coefficient == 0:
                return CalculationFunction(other)
            elif isinstance(target, ConstantFunction) and target.constant_coefficient == 0:
                return self
            return CalculatedFunctionMultiplication(target, other)
        elif isinstance(other, Real):
            if other == 0:
                return CalculationFunction(ConstantFunction(0))
            target = self.unwrap_functions(self)
            if isinstance(target, ConstantFunction) and target.constant_coefficient == 0:
                return self
            return CalculatedFunctionFactor(other, target)


        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Function):
            target, other = self.unwrap_functions(self, other)
            if isinstance(other, ConstantFunction):
                return self.__truediv__(other.constant_coefficient)
            return CalculatedFunctionDivision(target, other)
        elif isinstance(other, Real):
            if type(self) == CalculationFunction:
                assert self.constant_coefficient == 0
                return CalculatedFunctionFactor(1 / other, self._function)
            else:
                return CalculatedFunctionFactor(1 / other, self)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Function):
            other, target = self.unwrap_functions(other, self)
            if isinstance(other, ConstantFunction):
                return self.__add__(other.constant_coefficient)
            return CalculatedFunctionAddition(other, target)
        return self.__add__(other)

    def __rmul__(self, other):
        if isinstance(other, Function):
            other, target = self.unwrap_functions(other, self)
            if isinstance(other, ConstantFunction) and other.constant_coefficient == 0:
                return CalculationFunction(other)
            return CalculatedFunctionMultiplication(other, target)
        return self.__mul__(other)

    def __rsub__(self, other):
        if isinstance(other, Function):
            return CalculatedFunctionSubtraction(*self.unwrap_functions(other, self))
        else:
            return (self * -1).__add__(other)

    def __rtruediv__(self, other):
        if isinstance(other, Function):
            return CalculatedFunctionDivision(*self.unwrap_functions(other, self))
        elif isinstance(other, Real):
            if type(self) == CalculationFunction:
                assert self.constant_coefficient == 0
                return CalculatedFunctionDivision(ConstantFunction(other), self._function)
            else:
                return CalculatedFunctionDivision(ConstantFunction(other), self)
        else:
            return NotImplemented

    def __neg__(self):
        return self * -1

    @staticmethod
    def unwrap_functions(target, other=None):
        if type(target) == CalculationFunction:
            assert target.constant_coefficient == 0
            target = cast(CalculationFunction, target)._function
        if type(other) == CalculationFunction:
            assert other.constant_coefficient == 0
            other = cast(CalculationFunction, other)._function
        if other is None:
            return target
        return target, other


__ValueType = Union[Number, numpy.ndarray]


class CalculatedFunctionOperator(CalculationFunction, ABC):
    _operator: Callable[[__ValueType, __ValueType], __ValueType]
    _operator_name: str
    _is_prefix_operator: bool = False

    def evaluate(self, parameter_value: Union[__ValueType, Mapping[int, __ValueType],
                                              Sequence[__ValueType]]) -> __ValueType:
        rest = iter(self)
        function_value = next(rest).evaluate(parameter_value)
        for t in rest:
            function_value = self._operator(function_value, t.evaluate(parameter_value))
        function_value += self.constant_coefficient
        return function_value

    def to_string(self, *parameters: Union[str, Parameter]):
        if self._is_prefix_operator:
            result = self._operator_name + '(' + ', '.join(
                [f.to_string(*parameters) for f in self.compound_terms]) + ')'
        else:
            result = '(' + (')' + self._operator_name + '(').join(
                [f.to_string(*parameters) for f in self.compound_terms]) + ')'
        if self.constant_coefficient != 0:
            result = str(self.constant_coefficient) + " + " + result
        return result


class CalculatedFunctionAddition(CalculatedFunctionOperator):
    _operator = operator.add
    _operator_name = ' + '


class CalculatedFunctionSubtraction(CalculatedFunctionOperator):
    _operator = operator.sub
    _operator_name = ' - '


class CalculatedFunctionMultiplication(CalculatedFunctionOperator):
    _operator = operator.mul
    _operator_name = ' * '


class CalculatedFunctionFactor(CalculationFunction):
    def __init__(self, coefficient, function: Function):
        super(CalculatedFunctionFactor, self).__init__(function)
        self.coefficient = coefficient

    def evaluate(self, parameter_value: Union[__ValueType, Mapping[int, Union[__ValueType]],
                                              Sequence[Union[__ValueType]]]) -> Union[__ValueType]:
        return self.coefficient * self._function.evaluate(parameter_value) + self.constant_coefficient

    def to_string(self, *parameters: Union[str, Parameter]):
        result = str(self.coefficient) + ' * (' + self._function.to_string(*parameters) + ')'
        if self.constant_coefficient != 0:
            result = str(self.constant_coefficient) + " + " + result
        return result


class CalculatedFunctionDivision(CalculatedFunctionOperator):
    _operator = operator.truediv
    _operator_name = ' / '


class ConversionMetrics:
    def __init__(self, *required_metric: str):
        self.required_metrics = [Metric(name) for name in required_metric]

    def __call__(self, f):
        setattr(f, REQUIRED_METRICS_NAME, self.required_metrics)
        return f


class AbstractMetricConverter(ABC):
    Value = Union[Measurement, Function]

    _conversions = []

    @abc.abstractmethod
    @classproperty
    def NAME(cls) -> str:
        pass

    new_metric: Metric

    def __init__(self):
        self._order_mapping = []

    def __init_subclass__(cls, **kwargs):
        cls.new_metric = Metric(cls.NAME)
        cls._conversions = cls._conversions.copy()
        for method_name in dir(cls):
            method = getattr(cls, method_name, None)
            if method:
                metrics = getattr(method, REQUIRED_METRICS_NAME, None)
                if metrics:
                    sig = signature(method)
                    if len(metrics) != len(sig.parameters) - 1:
                        raise SyntaxError("For each metric there must exist exactly one parameter, besides self.")
                    cls._conversions.append(method)
        if len(cls._conversions) < 2:
            raise TypeError("At least two conversion functions must be defined.")
        super().__init_subclass__(**kwargs)

    def check_and_initialise(self, *metric_set):
        conversions = self._determine_permutation(metric_set)
        if conversions:
            self._order_mapping = [p for p, _ in conversions]
            return True
        return False

    @classmethod
    def _determine_permutation(cls, metric_set):
        if len(metric_set) != len(cls._conversions):
            return False
        for conversions in permutations(enumerate(cls._conversions)):
            if all(
                    all(m in metric_set[i] for m in getattr(conversion_function, REQUIRED_METRICS_NAME))
                    for i, (_, conversion_function) in
                    enumerate(conversions)
            ):
                return conversions
        return None

    @classmethod
    def try_create(cls, *metric_set):
        conversions = cls._determine_permutation(metric_set)
        if conversions:
            result = cls()
            result._order_mapping = [p for p, _ in conversions]
            return result
        return None

    def get_required_metrics(self, i):
        conversion = self._conversions[self._order_mapping[i]]
        return getattr(conversion, REQUIRED_METRICS_NAME)

    def get_conversion(self, i) -> Callable[[AbstractMetricConverter, *CalculationElement], Value]:
        conversion = self._conversions[self._order_mapping[i]]
        return conversion

    def convert_measurements(self, i, measurement_sets: Iterable[Sequence[Measurement]]):
        conversion_function = self.get_conversion(i)
        if len(self.get_required_metrics(i)) == 1:
            result = [conversion_function(self, measurement) for measurement in next(iter(measurement_sets))]
        else:
            measurements_ordered = defaultdict(list)
            for measurements in measurement_sets:
                for m in measurements:
                    measurements_ordered[m.coordinate].append(m)

            result = [conversion_function(self, *measurements) for measurements in measurements_ordered.values()]
        for m in result:
            m.metric = self.new_metric
        return result

    def convert_models(self, i, model_sets: Iterable[Model], measurements):
        conversion_function = self.get_conversion(i)
        functions = [CalculationFunction(m.hypothesis.function) for m in model_sets]
        function = conversion_function(self, *functions)
        if type(function) == CalculationFunction:
            function = CalculationFunction.unwrap_functions(function)
        model = next(iter(model_sets))
        hypothesis_class = Hypothesis.infer_best_type([m.hypothesis for m in model_sets])
        hypothesis = hypothesis_class(function, model.hypothesis.use_median)
        hypothesis.compute_cost(measurements)
        if hasattr(hypothesis, 'compute_adjusted_rsquared'):
            _, constant_cost = Hypothesis.calculate_constant_indicators(measurements, model.hypothesis.use_median)
            hypothesis.compute_adjusted_rsquared(constant_cost, measurements)
        result = Model(hypothesis, model.callpath, self.new_metric)
        result.measurements = measurements
        return result


all_conversions = load_extensions(__path__, __name__, AbstractMetricConverter)
