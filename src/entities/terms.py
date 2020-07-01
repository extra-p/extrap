from util.deprecation import deprecated
from abc import ABC, abstractmethod
from entities.fraction import Fraction
from typing import Tuple, List, Dict, Union, Mapping

import numpy as np

from entities.coordinate import Coordinate
from entities.parameter import Parameter


class Term(ABC):

    def __init__(self):
        self.coefficient = 1

    @deprecated("Use property directly.")
    def set_coefficient(self, coefficient):
        self.coefficient = coefficient

    @deprecated("Use property directly.")
    def get_coefficient(self):
        return self.coefficient

    @abstractmethod
    def to_string(self):
        raise NotImplementedError

    def __repr__(self):
        return f"Term({self.to_string()})"


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

    def __init__(self, term_type, exponent):
        self.term_type = term_type
        self.exponent = exponent

    @deprecated("Use property directly.")
    def set_exponent(self, exponent):
        self.exponent = exponent

    @deprecated("Use property directly.")
    def get_exponent(self):
        return self.exponent

    def to_string(self, parameter='p'):
        if self.term_type == "polynomial":
            return f"({parameter}^{self.exponent})"
        elif self.term_type == "logarithm":
            return f"log2({parameter})^({self.exponent})"

    def evaluate(self, parameter_value):
        if self.term_type == "polynomial":
            return np.power(parameter_value, float(self.exponent))
        elif self.term_type == "logarithm":
            return np.power(np.log2(parameter_value), float(self.exponent))


class CompoundTerm(SingleParameterTerm):

    def __init__(self, *terms):
        super().__init__()
        self.simple_terms: List[SimpleTerm] = list(terms)

    @deprecated("Use property directly.")
    def get_simple_terms(self):
        return self.simple_terms

    def add_simple_term(self, simple_term):
        self.simple_terms.append(simple_term)

    def evaluate(self, parameter_value: float):
        function_value = self.coefficient
        for t in self.simple_terms:
            function_value *= t.evaluate(parameter_value)
        return function_value

    def to_string(self, parameter='p'):
        function_string = str(self.coefficient)
        for t in self.simple_terms:
            function_string += ' * '
            function_string += t.to_string(parameter)
        return function_string

    def __imul__(self, term: SingleParameterTerm):
        self.simple_terms.append(term)
        return self

    @staticmethod
    def create(a, b, c):
        compound_term = CompoundTerm()
        if a != 0:
            compound_term *= SimpleTerm("polynomial", Fraction(a, b))
        if c != 0:
            compound_term *= SimpleTerm("logarithm", c)
        return compound_term


class MultiParameterTerm(Term):

    def __init__(self, *terms: List[Tuple[int, SingleParameterTerm]]):
        super().__init__()
        self.parameter_term_pairs = list(terms)

    def add_parameter_term_pair(self, parameter_term_pair: Tuple[int, SingleParameterTerm]):
        self.parameter_term_pairs.append(parameter_term_pair)

    @deprecated("Use parameter_term_pairs property directly.")
    def get_compound_term_parameter_pairs(self):
        return self.parameter_term_pairs

    @deprecated("Use add_parameter_term_pair instead.")
    def add_compound_term_parameter_pair(self, parameter_term_pair: Tuple[int, SingleParameterTerm]):
        self.parameter_term_pairs.append(parameter_term_pair)

    def evaluate(self, parameter_values: Union[Tuple[float], Coordinate]):
        function_value = self.coefficient
        for param, term in self.parameter_term_pairs:
            parameter_value = parameter_values[param]
            function_value *= term.evaluate(parameter_value)
        return function_value

    def to_string(self, *parameters: Union[Parameter, str, Mapping[int, Union[Parameter, str]]]):
        if len(parameters) == 0:
            parameters = ('p', 'q', 'r', 's', 't')
        elif len(parameters) == 1 and not isinstance(parameters[0], str):
            parameters = parameters[0]
        function_string = str(self.coefficient)
        for param, term in self.parameter_term_pairs:
            function_string += ' * '
            function_string += term.to_string(parameters[param])
        return function_string

    def __imul__(self, parameter_term_pair: Tuple[Parameter, SingleParameterTerm]):
        self.parameter_term_pairs.append(parameter_term_pair)
        return self

    def __repr__(self):
        return f"MPTerm({self.to_string()})"
