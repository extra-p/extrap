# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from abc import ABC, abstractmethod
from numbers import Real
from typing import Tuple, List, Union, Mapping

import numpy as np
from marshmallow import fields, validate

from extrap.entities.coordinate import Coordinate
from extrap.entities.fraction import Fraction
from extrap.entities.parameter import Parameter
from extrap.util.serialization_schema import Schema, NumberField

DEFAULT_PARAM_NAMES = (
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o')


class Term(ABC):

    def __init__(self):
        self.coefficient = 1

    @abstractmethod
    def to_string(self):
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
    def to_string(self, parameter: Union[Parameter, str] = 'p'):
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

    def to_string(self, parameter='p'):
        if self._term_type == "polynomial":
            return f"{parameter}**({self.exponent})"
        elif self._term_type == "logarithm":
            return f"log2({parameter})**({self.exponent})"

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
            return self.exponent == other.exponent and \
                   self._term_type == other._term_type


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

    def to_string(self, parameter='p'):
        function_string = '*'.join(t.to_string(parameter) for t in self.simple_terms)
        if self.coefficient != 1:
            function_string = str(self.coefficient) + '*' + function_string
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
            return self.coefficient == other.coefficient and \
                   self.simple_terms == other.simple_terms


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

    def to_string(self, *parameters: Union[Parameter, str, Mapping[int, Union[Parameter, str]]]):
        if len(parameters) == 0:
            parameters = DEFAULT_PARAM_NAMES
        elif len(parameters) == 1 and not isinstance(parameters[0], str):
            parameters = parameters[0]
        function_string = str(self.coefficient)
        for param, term in self.parameter_term_pairs:
            function_string += '*'
            function_string += term.to_string(parameters[param])
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
            return self.parameter_term_pairs == other.parameter_term_pairs and \
                   self.coefficient == other.coefficient


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


class ListOfPairs(fields.Dict):
    def _serialize(self, value, attr, obj, **kwargs):
        return super(ListOfPairs, self)._serialize(dict(value), attr, obj, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        dict_ = super(ListOfPairs, self)._deserialize(value, attr, data, **kwargs)
        return list(dict_.items())


class MultiParameterTermSchema(TermSchema):
    parameter_term_pairs = ListOfPairs(keys=fields.Int(), values=fields.Nested(CompoundTermSchema))

    def create_object(self):
        return MultiParameterTerm()
