import copy
import operator
from abc import ABC
from numbers import Number, Real
from typing import Union, Mapping, Sequence, cast, Callable

import numpy
from marshmallow import fields

from extrap.comparison.entities.calculation_element import CalculationElement
from extrap.entities.functions import Function, ConstantFunction, FunctionSchema
from extrap.entities.parameter import Parameter
from extrap.util.serialization_schema import NumberField


class CalculationFunction(Function, CalculationElement):

    def __init__(self, *function: Function):
        super().__init__(*function)
        if function:
            self._function = function[0]
        else:
            self._function = None

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


class CalculationFunctionSchema(FunctionSchema):
    def create_object(self):
        return CalculationFunction()

    _function = fields.Nested(FunctionSchema)


_ValueType = Union[Number, numpy.ndarray]


class CalculatedFunctionOperator(CalculationFunction, ABC):
    _operator: Callable[[_ValueType, _ValueType], _ValueType]
    _operator_name: str
    _is_prefix_operator: bool = False

    def evaluate(self, parameter_value: Union[_ValueType, Mapping[int, _ValueType],
                                              Sequence[_ValueType]]) -> _ValueType:
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


class CalculatedFunctionOperatorSchema(CalculationFunctionSchema):
    def create_object(self):
        return NotImplemented, CalculatedFunctionOperator


class CalculatedFunctionAddition(CalculatedFunctionOperator):
    _operator = operator.add
    _operator_name = ' + '


class CalculatedFunctionAdditionSchema(CalculatedFunctionOperatorSchema):
    def create_object(self):
        return CalculatedFunctionAddition()


class CalculatedFunctionSubtraction(CalculatedFunctionOperator):
    _operator = operator.sub
    _operator_name = ' - '


class CalculatedFunctionSubtractionSchema(CalculatedFunctionOperatorSchema):
    def create_object(self):
        return CalculatedFunctionSubtraction()


class CalculatedFunctionMultiplication(CalculatedFunctionOperator):
    _operator = operator.mul
    _operator_name = ' * '


class CalculatedFunctionMultiplicationSchema(CalculatedFunctionOperatorSchema):
    def create_object(self):
        return CalculatedFunctionMultiplication()


class CalculatedFunctionFactor(CalculationFunction):
    def __init__(self, coefficient, function: Function):
        if function == NotImplemented:
            super(CalculatedFunctionFactor, self).__init__()
        else:
            super(CalculatedFunctionFactor, self).__init__(function)
        self.coefficient = coefficient

    def evaluate(self, parameter_value: Union[_ValueType, Mapping[int, Union[_ValueType]],
                                              Sequence[Union[_ValueType]]]) -> Union[_ValueType]:
        return self.coefficient * self._function.evaluate(parameter_value) + self.constant_coefficient

    def to_string(self, *parameters: Union[str, Parameter]):
        result = str(self.coefficient) + ' * (' + self._function.to_string(*parameters) + ')'
        if self.constant_coefficient != 0:
            result = str(self.constant_coefficient) + " + " + result
        return result


class CalculatedFunctionFactorSchema(CalculatedFunctionOperatorSchema):
    coefficient = NumberField()

    def create_object(self):
        return CalculatedFunctionFactor(0, NotImplemented)


class CalculatedFunctionDivision(CalculatedFunctionOperator):
    _operator = operator.truediv
    _operator_name = ' / '


class CalculatedFunctionDivisionSchema(CalculatedFunctionOperatorSchema):
    def create_object(self):
        return CalculatedFunctionDivision()
