# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import numbers
from numbers import Number
from typing import List, Mapping, Union, Sequence

import numpy
import sympy
from marshmallow import fields

from extrap.entities.parameter import Parameter
from extrap.entities.terms import CompoundTerm, MultiParameterTerm, CompoundTermSchema, MultiParameterTermSchema
from extrap.util.serialization_schema import BaseSchema, NumberField

_TermType = Union[CompoundTerm, MultiParameterTerm]


class Function:
    def __init__(self, *compound_terms: _TermType):
        """
        Initialize a Function object.
        """
        self.constant_coefficient = 0
        self.compound_terms: List[_TermType] = list(compound_terms)

    def add_compound_term(self, compound_term: _TermType):
        """
        Add a compound term to the function.
        """
        self.compound_terms.append(compound_term)

    def reset_coefficients(self):
        self.constant_coefficient = 0
        for t in self.compound_terms:
            t.reset_coefficients()

    def __iadd__(self, compound_term: _TermType):
        self.add_compound_term(compound_term)
        return self

    def evaluate(self, parameter_value: Union[
        Number, numpy.ndarray, Mapping[int, Union[Number, numpy.ndarray]],
        Sequence[Union[Number, numpy.ndarray, sympy.Symbol]]]) -> Union[Number, numpy.ndarray, sympy.Basic]:
        """
        Evaluate the function according to the given value and return the result.

        If the input is an ndarray the following rules apply:
        If the ndarray is one-dimensional, each element is interpreted as if this was one number input to this function.
        The output has the same shape as the input.
        If the ndarry is two-dimensional the first dimension is interpreted as the different parameters.
        The second dimension is interpreted as individual values.
        The output is one-dimensional with the same length as the second dimension.
        """

        if isinstance(parameter_value, numpy.ndarray):
            shape = parameter_value.shape
            if len(shape) == 2:
                shape = (shape[1],)
            function_value = numpy.full(shape, self.constant_coefficient, dtype=float)
        else:
            function_value = self.constant_coefficient
        for t in self.compound_terms:
            function_value += t.evaluate(parameter_value)
        return function_value

    def to_string(self, *parameters: Union[str, Parameter]):
        """
        Return a string representation of the function.
        """
        function_string = str(self.constant_coefficient)
        for t in self.compound_terms:
            function_string += ' + '
            function_string += t.to_string(*parameters)
        return function_string

    def __repr__(self):
        return f"Function({self.to_string('p')})"

    def __iter__(self):
        return iter(self.compound_terms)

    def __getitem__(self, i):
        return self.compound_terms[i]

    def __str__(self):
        return self.to_string()

    def __eq__(self, other):
        if not isinstance(other, Function):
            return NotImplemented
        elif self is other:
            return True
        else:
            return self.__dict__ == other.__dict__

    def partial_compare(self, other: Function) -> Union[tuple[numbers.Number], numbers.Number]:
        """
        Compares this function to another function. The comparison happens per parameter, so if the result is the same
        for all parameters that are not equal only one result is returned.
        If the results for the parameters contradict each other all of them are returned in a tuple.

        The comparison is based on the calculation of the limit.

        :param other: The function that is compared with this function.
        :return: A tuple of comparison results if the comparisons per parameter do not agree.
                 If the comparisons agree, only on result is returned.
                 A comparison result is a number that is either positive, negative or 0.
                 If this function is greater than the other function a positive number is returned.
                 If both functions are equal 0 is returned.
                 If this function is lower than the other function a negative number is returned.
        """
        from extrap.entities.function_computation import ComputationFunction
        if not isinstance(other, Function):
            return NotImplemented
        elif self is other:
            return 0
        else:
            return ComputationFunction(self).partial_compare(other)


class TermlessFunction(Function):

    def __init__(self):
        super(TermlessFunction, self).__init__()
        self.add_compound_term = None
        self.__iadd__ = None

    @property
    def compound_terms(self):
        return []

    @compound_terms.setter
    def compound_terms(self, val):
        if val:
            raise NotImplementedError()


class ConstantFunction(TermlessFunction):
    """
    This class represents a constant function.
    """

    def __init__(self, constant_coefficient=1):
        super().__init__()
        self.constant_coefficient = constant_coefficient

    def to_string(self, *_):
        """
        Returns a string representation of the constant function.
        """
        return str(self.constant_coefficient)

    def partial_compare(self, other):
        if isinstance(other, ConstantFunction):
            return self.constant_coefficient - other.constant_coefficient
        elif isinstance(other, SingleParameterFunction):
            return other.lead_order_term.coefficient
        else:
            super().partial_compare(other)


class SingleParameterFunction(Function):
    """
    This class represents a single parameter function
    """
    compound_terms: List[CompoundTerm]

    def __init__(self, *compound_terms: CompoundTerm):
        super().__init__(*compound_terms)

    def evaluate(self, parameter_value):
        if hasattr(parameter_value, '__len__') and (len(parameter_value) == 1 or isinstance(parameter_value, Mapping)):
            parameter_value = parameter_value[0]
        return super().evaluate(parameter_value)

    @property
    def lead_order_term(self):
        max_exponents = [0, 0]
        max_term = None
        for cterm in self.compound_terms:
            term_types_order = ['polynomial', 'logarithm']
            exponents = [0, 0]
            for term in cterm.simple_terms:
                for i, tt in enumerate(term_types_order):
                    if term.term_type == tt:
                        exponents[i] += term.exponent
            if exponents > max_exponents:
                max_exponents = exponents
                max_term = cterm
        return max_term


class MultiParameterFunction(Function):
    compound_terms: List[MultiParameterTerm]

    def __init__(self, *compound_terms: MultiParameterTerm):
        super().__init__(*compound_terms)

    def __repr__(self):
        return f"MultiParameterFunction({self.to_string()})"


class FunctionSchema(BaseSchema):
    constant_coefficient = NumberField()
    compound_terms: List[_TermType] = fields.List(fields.Nested(CompoundTermSchema))  # Not really correct


class TermlessFunctionSchema(FunctionSchema):
    compound_terms = fields.Constant([], load_only=True)

    def create_object(self):
        return NotImplemented, FunctionSchema


class ConstantFunctionSchema(TermlessFunctionSchema):

    def create_object(self):
        return ConstantFunction()


class SingleParameterFunctionSchema(FunctionSchema):
    compound_terms: List[CompoundTerm] = fields.List(fields.Nested(CompoundTermSchema))

    def create_object(self):
        return SingleParameterFunction()


class MultiParameterFunctionSchema(FunctionSchema):
    compound_terms: List[MultiParameterTerm] = fields.List(fields.Nested(MultiParameterTermSchema))

    def create_object(self):
        return MultiParameterFunction()
