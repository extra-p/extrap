# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import math
import numbers
import warnings
from itertools import chain
from typing import List, Mapping, Union, Sequence
from numbers import Number
from typing import List, Mapping, Union
from typing import Sequence

import numpy
import numpy as np
import sympy
from marshmallow import fields

from extrap.entities.parameter import Parameter
from extrap.entities.terms import CompoundTerm, MultiParameterTerm, CompoundTermSchema, MultiParameterTermSchema, \
    SegmentedTerm
from extrap.entities.terms import CompoundTerm, MultiParameterTerm, CompoundTermSchema, MultiParameterTermSchema
from extrap.util.formatting_helper import format_number_html
from extrap.util.latex_formatting import frmt_scientific_coefficient
from extrap.util.serialization_schema import BaseSchema, NumberField, NumpyField, VariantSchemaField
from extrap.util.string_formats import FunctionFormats

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
        warnings.warn("This operator is deprecated use add_compound_term instead.", DeprecationWarning)
        if not isinstance(compound_term, (CompoundTerm, MultiParameterTerm)):
            return NotImplemented
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

    def to_string(self, *parameters: Union[str, Parameter], format: FunctionFormats = None):
        """
        Return a string representation of the function.
        """
        if format == FunctionFormats.LATEX:
            return self.to_latex_string(*parameters)

        term_list = (t.to_string(*parameters, format=format) for t in self.compound_terms)
        if self.constant_coefficient != 0 or not self.compound_terms:
            term_list = chain([str(self.constant_coefficient)], term_list)

        joiner = ' + '
        if format == FunctionFormats.PYTHON:
            joiner = '+'
        function_string = joiner.join(term_list)

        if format == FunctionFormats.PYTHON:
            function_string = function_string.replace('+-', '-')
        return function_string

    def to_html(self, *parameters: Union[str, Parameter]):
        """
        Return a html representation of the function.
        """
        function_string = ' + '.join(t.to_html(*parameters) for t in self.compound_terms)
        if self.constant_coefficient != 0:
            coefficient_string = format_number_html(self.constant_coefficient)
            if coefficient_string[0] == '-':
                function_string = function_string + ' - ' + coefficient_string[1:]
            else:
                function_string = function_string + ' + ' + coefficient_string
        return function_string

    def to_latex_string(self, *parameters: Union[str, Parameter]):
        """
        Return a math string (using latex encoding) representation of the function.
        """
        new_coefficients = []
        new_coefficients.append(frmt_scientific_coefficient(self.constant_coefficient))
        for t in self.compound_terms:
            new_coefficients.append(frmt_scientific_coefficient(t.coefficient))
        function_string = new_coefficients[0]
        coeff_counter = 1
        for t in self.compound_terms:
            if isinstance(t, MultiParameterTerm) is True:
                sub_terms = t.parameter_term_pairs
            elif isinstance(t, CompoundTerm) is True:
                sub_terms = t.simple_terms
            new_term = new_coefficients[coeff_counter]
            for sub_term in sub_terms:
                if type(sub_term) is tuple:
                    new_term = new_term + "*" + sub_term[1].to_string(parameter=parameters[sub_term[0]],
                                                                      format=FunctionFormats.PYTHON)
                    new_term = new_term.replace("log2(" + str(parameters[sub_term[0]]) + ")",
                                                "\\log_2{" + str(parameters[sub_term[0]]) + "}")
                else:
                    new_term = new_term + "*" + sub_term.to_string(parameter=parameters[0],
                                                                   format=FunctionFormats.PYTHON)
                    new_term = new_term.replace("log2(" + str(parameters[0]) + ")",
                                                "\\log_2{" + str(parameters[0]) + "}")
            new_term = new_term.replace("**", "^")
            new_term = new_term.replace("*", "\\cdot ")
            new_term = new_term.replace("(", "{")
            new_term = new_term.replace(")", "}")
            if new_term[0] != "-":
                function_string += "+"
            function_string += new_term
            coeff_counter += 1
        return "$" + function_string + "$"

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
            if self.__dict__.keys() != other.__dict__.keys():
                return False

            for k in self.__dict__:
                if np.any(self.__dict__[k] != other.__dict__[k]):
                    return False

            return True

    def partial_compare(self, other: Function) -> Union[tuple[numbers.Number], numbers.Number]:
        """
        Compares this function to another function. The comparison happens per parameter, so if the result is the same
        for all parameters that are not equal only one result is returned.
        If the results for the parameters contradict each other all of them are returned in a tuple.

        The comparison is based on the calculation of the limit.

        :param other: The function that is compared with this function.
        :return: A tuple of comparison results if the comparisons per parameter do not agree.
                 If the comparisons agree, only one result is returned.
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

    def to_string(self, *_, format: FunctionFormats = None):
        """
        Returns a string representation of the constant function.
        """
        return str(self.constant_coefficient)

    def to_html(self, *_):
        """
        Returns a html representation of the constant function.
        """
        return format_number_html(self.constant_coefficient)

    def partial_compare(self, other):
        if isinstance(other, ConstantFunction):
            return self.constant_coefficient - other.constant_coefficient
        elif isinstance(other, SingleParameterFunction):
            return other.lead_order_term.coefficient
        else:
            return super().partial_compare(other)


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


class SegmentedFunction(SingleParameterFunction):
    MAX_NUM_SEGMENTS = 2

    def __init__(self, segments: list[SingleParameterFunction],
                 intervals: Sequence[tuple[numbers.Number, numbers.Number]]):
        super().__init__()
        self.add_compound_term = None
        self.__iadd__ = None
        if len(segments) > self.MAX_NUM_SEGMENTS:
            raise ValueError(f"Only {self.MAX_NUM_SEGMENTS} are allowed.")
        if len(segments) != len(intervals):
            raise ValueError("Number of intervals must be equal to the number of segments")
        self.segments = segments
        self.intervals = np.array(intervals, dtype=float)

    @property
    def constant_coefficient(self):
        return 0

    @constant_coefficient.setter
    def constant_coefficient(self, value):
        if value != 0:
            raise NotImplementedError()

    @property
    def compound_terms(self):
        return [SegmentedTerm(self.segments, self.intervals)]

    @compound_terms.setter
    def compound_terms(self, value):
        if value:
            raise NotImplementedError()

    def reset_coefficients(self):
        for v in self.segments:
            v.reset_coefficients()

    def evaluate(self, parameter_value):
        if not self.segments:
            return super().evaluate(parameter_value)

        if hasattr(parameter_value, '__len__') and (len(parameter_value) == 1 or isinstance(parameter_value, Mapping)):
            parameter_value = parameter_value[0]

        if isinstance(parameter_value, np.ndarray):
            function_value = np.ndarray(parameter_value.shape, dtype=float)
            int_start = self.intervals[:, 0]
            int_end = self.intervals[:, 1]
            int_start_mask = int_start.reshape(1, -1) <= parameter_value.reshape(-1, 1)
            int_end_mask = parameter_value.reshape(-1, 1) <= int_end.reshape(1, -1)
            mask = int_start_mask & int_end_mask
            match = np.argmax(mask, axis=1)
            for i, segment in enumerate(self.segments):
                function_value[match == i] = segment.evaluate(parameter_value[match == i])
            function_value[~np.any(mask, axis=1)] = math.nan
            return function_value
        else:
            for (int_start, int_end), segment in zip(self.intervals, self.segments):
                if int_start <= parameter_value <= int_end:
                    return segment.evaluate(parameter_value)
            return math.nan

    def to_string(self, *parameters: Union[str, Parameter], format: FunctionFormats = None):
        """
        Return a string representation of the function.
        """
        if format == FunctionFormats.LATEX:
            return self.to_latex_string(*parameters)

        elif format == FunctionFormats.PYTHON:
            function_string = self.segments[0].to_string(*parameters, format=format)
            function_string += f" if {parameters[0]}<={self.intervals[0][1]} else"
            function_string += self.segments[1].to_string(*parameters, format=format)
            if self.intervals[0][1] != self.intervals[1][0]:
                function_string += f" if {parameters[0]}>={self.intervals[1][0]} else math.nan"

        else:
            function_string = self.segments[0].to_string(*parameters, format=format)
            function_string += f" for {parameters[0]}<={self.intervals[0][1]}\n"
            function_string += self.segments[1].to_string(*parameters, format=format)
            function_string += f" for {parameters[0]}>={self.intervals[1][0]}"

        return function_string

    def to_latex_string(self, *parameters: Union[str, Parameter]):
        """
        Return a math string (using latex encoding) representation of the function.
        """
        function_string = self.segments[0].to_latex_string(*parameters)
        function_string += f" for ${parameters[0]}<={self.intervals[0][1]}$\n"
        function_string += self.segments[1].to_latex_string(*parameters)
        function_string += f" for ${parameters[0]}>={self.intervals[1][0]}$"
        return function_string


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


class SegmentedFunctionSchema(SingleParameterFunctionSchema):
    compound_terms = fields.Constant([], load_only=True)
    constant_coefficient = fields.Constant(0, load_only=True)
    segments = fields.List(VariantSchemaField(SingleParameterFunctionSchema, ConstantFunctionSchema))
    intervals = NumpyField()

    def create_object(self):
        return SegmentedFunction([], [])
