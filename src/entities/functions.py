"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""
from typing import List, Mapping

from entities.parameter import Parameter
from entities.terms import CompoundTerm, MultiParameterTerm
from util.deprecation import deprecated
import collections
import numpy


class Function:
    def __init__(self, *compound_terms: CompoundTerm):
        """
        Initialize a Function object.
        """
        self.constant_coefficient = 1
        self.compound_terms: List[CompoundTerm] = list(compound_terms)

    def add_compound_term(self, compound_term):
        """
        Add a compound term to the function.
        """
        self.compound_terms.append(compound_term)

    def __iadd__(self, compound_term):
        self.add_compound_term(compound_term)
        return self

    @deprecated("Use property directly.")
    def get_compound_terms(self):
        """
        Return all the compound terms of the function.
        """
        return self.compound_terms

    @deprecated("Use property directly.")
    def get_compound_term(self, compound_term_id):
        """
        Return the compound term of the given id of the function.
        """
        return self.compound_terms[compound_term_id]

    @deprecated("Use property directly.")
    def set_constant_coefficient(self, constant_coefficient):
        """
        Set the constant coefficient of the function to the given value.
        """
        self.constant_coefficient = constant_coefficient

    @deprecated("Use property directly.")
    def get_constant_coefficient(self):
        """
        Return the constant coefficient of the function.
        """
        return self.constant_coefficient

    def evaluate(self, parameter_value):
        """
        Evalute the function according to the given value and return the result.
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

    def to_string(self, *parameters):
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


class ConstantFunction(Function):
    """
    This class represents a constant function.
    """

    def __init__(self, constant_coefficient=1):
        super().__init__()
        self.constant_coefficient = constant_coefficient
        self.add_compound_term = None
        self.__iadd__ = None

    def to_string(self, *_):
        """
        Returns a string representation of the constant function.
        """
        return str(self.constant_coefficient)


class SingleParameterFunction(Function):
    """
    This class represents a single parameter function
    """

    def __init__(self, *compound_terms):
        super().__init__(*compound_terms)

    def evaluate(self, parameter_value):
        if hasattr(parameter_value, '__len__') and (len(parameter_value) == 1 or isinstance(parameter_value, Mapping)):
            parameter_value = parameter_value[0]
        return super().evaluate(parameter_value)


class MultiParameterFunction(Function):

    def __init__(self, *compound_terms: MultiParameterTerm):
        super().__init__(*compound_terms)

    @deprecated("Use add_compound_term(Term) instead.")
    def add_multi_parameter_term(self, multi_parameter_term):
        self.add_compound_term(multi_parameter_term)

    @deprecated("Use indexer instead.")
    def get_multi_parameter_terms(self):
        return self.compound_terms

    def __repr__(self):
        return f"MultiParameterFunction({self.to_string()})"
