# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import math
from abc import ABC, abstractmethod
from itertools import chain
from numbers import Real
from typing import Tuple, List, Union, Mapping

import numpy as np
from marshmallow import fields, validate

from extrap.entities.coordinate import Coordinate
from extrap.entities.fraction import Fraction
from extrap.entities.parameter import Parameter
from extrap.util.serialization_schema import Schema, NumberField, NumpyField, VariantSchemaField
from extrap.util.string_formats import FunctionFormats

DEFAULT_PARAM_NAMES = (
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o')


class Term(ABC):

    def __init__(self):
        self.coefficient = 1

    @abstractmethod
    def to_string(self, *, format: FunctionFormats = None):
        raise NotImplementedError

    def reset_coefficients(self):
        self.coefficient = 1

    def __repr__(self):
        return f"Term({self.to_string()})"

    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        elif self is other:
            return True
        else:
            return self.coefficient == other.coefficient


class SingleParameterTerm(Term, ABC):
    @abstractmethod
    def evaluate(self, parameter_value):
        raise NotImplementedError

    def __mul__(self, other):
        return CompoundTerm(self, other)

    @abstractmethod
    def to_string(self, parameter: Union[Parameter, str] = 'p', *, format: FunctionFormats = None):
        raise NotImplementedError


class SimpleTerm(SingleParameterTerm):

    def __init__(self, term_type, exponent: Real):
        super().__init__()
        del self.coefficient
        self.term_type = term_type
        self.exponent = exponent

    @property
    def exponent(self):
        return self._exponent

    @exponent.setter
    def exponent(self, value):
        self._exponent = value
        self._float_exponent = float(value)

    @property
    def term_type(self):
        return self._term_type

    @term_type.setter
    def term_type(self, val):
        self._term_type = val
        if self._term_type == "polynomial":
            self.evaluate = self._evaluate_polynomial
        elif self._term_type == "logarithm":
            self.evaluate = self._evaluate_logarithm

    def reset_coefficients(self):
        pass

    def to_string(self, parameter='p', *, format: FunctionFormats = None):
        if self._term_type == "polynomial":
            if format == FunctionFormats.PYTHON:
                return f"{parameter}**({self.exponent})"
            elif format == FunctionFormats.LATEX:
                return f"{{{parameter}}}^{{{self.exponent}}}"
            return f"{parameter}^({self.exponent})"
        elif self._term_type == "logarithm":
            if format == FunctionFormats.PYTHON:
                return f"log2({parameter})**({self.exponent})"
            elif format == FunctionFormats.LATEX:
                return f"\\log2{{{parameter}}}^{{{self.exponent}}}"
            return f"log2({parameter})^({self.exponent})"

    def _evaluate_polynomial(self, parameter_value):
        return parameter_value ** self._float_exponent

    def _evaluate_logarithm(self, parameter_value):
        log = np.log2(parameter_value)
        log **= self._float_exponent
        return log

    def evaluate(self, parameter_value):
        # is dispatched during object creation
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, SimpleTerm):
            return False
        elif self is other:
            return True
        else:
            return (self.exponent == other.exponent and
                    self._term_type == other._term_type)


class CompoundTerm(SingleParameterTerm):

    def __init__(self, *terms):
        super().__init__()
        self.simple_terms: List[SimpleTerm] = list(terms)

    def add_simple_term(self, simple_term):
        self.simple_terms.append(simple_term)

    def evaluate(self, parameter_value):
        function_value = self.coefficient
        for t in self.simple_terms:
            function_value *= t.evaluate(parameter_value)
        return function_value

    def to_string(self, parameter='p', *, format: FunctionFormats = None):
        term_list = (t.to_string(parameter, format=format) for t in self.simple_terms)
        if self.coefficient != 1 or not self.simple_terms:
            term_list = chain([str(self.coefficient)], term_list)

        joiner = ' * '
        if format == FunctionFormats.PYTHON:
            joiner = '*'
        elif format == FunctionFormats.LATEX:
            joiner = '\\cdot '
        function_string = joiner.join(term_list)
        return function_string

    def __imul__(self, term: SimpleTerm):
        self.simple_terms.append(term)
        return self

    @staticmethod
    def create(a, b, c=None):
        if c is None:
            f, c = a, b
        else:
            f = Fraction(a, b)

        compound_term = CompoundTerm()
        if a != 0:
            compound_term *= SimpleTerm("polynomial", f)
        if c != 0:
            compound_term *= SimpleTerm("logarithm", c)
        return compound_term

    def __eq__(self, other):
        if not isinstance(other, CompoundTerm):
            return False
        elif self is other:
            return True
        else:
            return (self.coefficient == other.coefficient and
                    self.simple_terms == other.simple_terms)


class SegmentedTerm(CompoundTerm):

    def __init__(self, segments, intervals):
        super().__init__()
        self.simple_terms: List[SimpleTerm] = []
        self.segments = segments
        self.intervals = intervals

    def reset_coefficients(self):
        pass

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

    def to_string(self, parameter='p', *, format: FunctionFormats = None):
        """
        Return a string representation of the function.
        """

        if format == FunctionFormats.LATEX:
            return self.to_latex_string(parameter)

        elif format == FunctionFormats.PYTHON:
            function_string = "(" + self.segments[0].to_string(parameter, format=format)
            function_string += f" if {parameter}<={self.intervals[0][1]} else"
            function_string += self.segments[1].to_string(parameter, format=format)
            if self.intervals[0][1] != self.intervals[1][0]:
                function_string += f" if {parameter}>={self.intervals[1][0]} else math.nan)"

        else:
            function_string = "{" + self.segments[0].to_string(parameter, format=format)
            function_string += f" for {parameter}<={self.intervals[0][1]}; "
            function_string += self.segments[1].to_string(parameter, format=format)
            function_string += f" for {parameter}>={self.intervals[1][0]}}}"

        return function_string

    def to_latex_string(self, parameter):
        """
        Return a math string (using latex encoding) representation of the function.
        """
        function_string = "(" + self.segments[0].to_latex_string(parameter).replace("$", "")
        function_string += f" for {parameter}<={self.intervals[0][1]}\n"
        function_string += self.segments[1].to_latex_string(*parameter).replace("$", "")
        function_string += f" for {parameter}>={self.intervals[1][0]})"
        return function_string


class MultiParameterTerm(Term):

    def __init__(self, *terms: Tuple[int, SingleParameterTerm]):
        super().__init__()
        if len(terms) > 0 and not isinstance(terms[0], Tuple):
            raise TypeError('Argument must be a pair of parameter index and term.')
        self.parameter_term_pairs = list(terms)

    def add_parameter_term_pair(self, parameter_term_pair: Tuple[int, SingleParameterTerm]):
        self.parameter_term_pairs.append(parameter_term_pair)

    def reset_coefficients(self):
        super().reset_coefficients()
        for _, t in self.parameter_term_pairs:
            t.reset_coefficients()

    def evaluate(self, parameter_values: Union[Tuple[float], Coordinate]):
        function_value = self.coefficient
        for param, term in self.parameter_term_pairs:
            parameter_value = parameter_values[param]
            function_value *= term.evaluate(parameter_value)
        return function_value

    def to_string(self, *parameters: Union[Parameter, str, Mapping[int, Union[Parameter, str]]],
                  format: FunctionFormats = None):
        if len(parameters) == 0:
            parameters = DEFAULT_PARAM_NAMES
        elif len(parameters) == 1 and not isinstance(parameters[0], str):
            parameters = parameters[0]

        term_list = (term.to_string(parameters[param], format=format) for param, term in self.parameter_term_pairs)
        if self.coefficient != 1 or not self.parameter_term_pairs:
            term_list = chain([str(self.coefficient)], term_list)

        joiner = ' * '
        if format == FunctionFormats.PYTHON:
            joiner = '*'
        if format == FunctionFormats.LATEX:
            joiner = '\\cdot '
        function_string = joiner.join(term_list)
        return function_string

    def __imul__(self, parameter_term_pair: Tuple[int, SingleParameterTerm]):
        self.parameter_term_pairs.append(parameter_term_pair)
        return self

    def __repr__(self):
        return f"MPTerm({self.to_string()})"

    def __eq__(self, other):
        if not isinstance(other, MultiParameterTerm):
            return False
        elif self is other:
            return True
        else:
            return (self.parameter_term_pairs == other.parameter_term_pairs and
                    self.coefficient == other.coefficient)


class TermSchema(Schema):
    coefficient = NumberField()


class SimpleTermSchema(TermSchema):
    coefficient = None
    term_type = fields.String(validate=validate.OneOf(['polynomial', 'logarithm']))
    exponent = NumberField()

    def create_object(self):
        return SimpleTerm(None, 0)


class CompoundTermSchema(TermSchema):
    simple_terms = fields.List(fields.Nested(SimpleTermSchema))

    def create_object(self):
        return CompoundTerm()


class SegmentedTermSchema(TermSchema):
    simple_terms = fields.Constant([], load_only=True)
    segments = fields.List(fields.Nested('SingleParameterFunctionSchema'))
    intervals = NumpyField()

    def create_object(self):
        return SegmentedTerm([], [])


class ListOfPairs(fields.Dict):
    def _serialize(self, value, attr, obj, **kwargs):
        return super(ListOfPairs, self)._serialize(dict(value), attr, obj, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        dict_ = super(ListOfPairs, self)._deserialize(value, attr, data, **kwargs)
        return list(dict_.items())


class MultiParameterTermSchema(TermSchema):
    parameter_term_pairs = ListOfPairs(keys=fields.Int(),
                                       values=VariantSchemaField(CompoundTermSchema, SegmentedTermSchema))

    def create_object(self):
        return MultiParameterTerm()
