# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import List, Mapping, Union

import numpy
from marshmallow import fields
import matplotlib.ticker as mticker

from extrap.entities.parameter import Parameter
from extrap.entities.terms import CompoundTerm, MultiParameterTerm, CompoundTermSchema, MultiParameterTermSchema, DEFAULT_PARAM_NAMES
from extrap.util.serialization_schema import BaseSchema, NumberField


class Function:
    def __init__(self, *compound_terms: CompoundTerm):
        """
        Initialize a Function object.
        """
        self.constant_coefficient = 0
        self.compound_terms: List[CompoundTerm] = list(compound_terms)

    def add_compound_term(self, compound_term):
        """
        Add a compound term to the function.
        """
        self.compound_terms.append(compound_term)

    def reset_coefficients(self):
        self.constant_coefficient = 0
        for t in self.compound_terms:
            t.reset_coefficients()

    def __iadd__(self, compound_term):
        self.add_compound_term(compound_term)
        return self

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

    def to_string(self, *parameters: Union[str, Parameter]):
        """
        Return a string representation of the function.
        """
        function_string = str(self.constant_coefficient)
        for t in self.compound_terms:
            if t.to_string(*parameters)[0] != "-":
                function_string += '+'
            function_string += t.to_string(*parameters)
        return function_string
    
    def to_scientific_coefficient(self, coefficient):
        """
        This method takes a coefficient and formats it into a string using scientific notation.
        """
        formater = mticker.ScalarFormatter(useMathText=True)
        formater.set_powerlimits((-3, 3))
        formatted_coefficients = "{}".format(formater.format_data(float(coefficient)))
        coreff_terms = formatted_coefficients.split(" ")
        new_coeff = ""
        if not coreff_terms[0][:1].isnumeric():
            coeff = coreff_terms[0][1:]
            try:
                coeff = "{:.3f}".format(float(coeff))
            except ValueError:
                pass
            new_coeff += "-"
            new_coeff += coeff
            for i in range(len(coreff_terms)):
                if i != 0:
                    new_coeff += coreff_terms[i]
            return new_coeff
        else:
            coeff = coreff_terms[0]
            try:
                coeff = "{:.3f}".format(float(coeff))
            except ValueError:
                pass
            new_coeff += coeff
            for i in range(len(coreff_terms)):
                if i != 0:
                    new_coeff += coreff_terms[i]
            return new_coeff
    
    def to_latex_string(self, *parameters: Union[str, Parameter]):
        """
        Return a math string (using latex encoding) representation of the function.
        """
        new_coefficients = []
        new_coefficients.append(self.to_scientific_coefficient(self.constant_coefficient))
        for t in self.compound_terms:
            new_coefficients.append(self.to_scientific_coefficient(t.coefficient))
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
                    new_term = new_term + "*" + sub_term[1].to_string(parameter=parameters[sub_term[0]])
                    new_term = new_term.replace("log2{"+str(parameters[sub_term[0]])+"}", "\\log_2("+str(parameters[sub_term[0]])+")")
                else:
                    new_term = new_term + "*" + sub_term.to_string(parameter=parameters[0])
                    new_term = new_term.replace("log2{"+str(parameters[0])+"}", "\\log_2("+str(parameters[0])+")")
            new_term = new_term.replace("**", "^")
            new_term = new_term.replace("*", "\\cdot")
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
            return self.__dict__ == other.__dict__


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
    compound_terms: List[MultiParameterTerm]
    
    def __init__(self, *compound_terms: MultiParameterTerm):
        super().__init__(*compound_terms)

    def __repr__(self):
        return f"MultiParameterFunction({self.to_string()})"


class FunctionSchema(BaseSchema):
    constant_coefficient = NumberField()
    compound_terms: List[CompoundTerm] = fields.List(fields.Nested(CompoundTermSchema))


class ConstantFunctionSchema(FunctionSchema):
    compound_terms = fields.Constant([], load_only=True)

    def create_object(self):
        return ConstantFunction()


class SingleParameterFunctionSchema(FunctionSchema):
    def create_object(self):
        return SingleParameterFunction()


class MultiParameterFunctionSchema(FunctionSchema):
    compound_terms: List[CompoundTerm] = fields.List(fields.Nested(MultiParameterTermSchema))

    def create_object(self):
        return MultiParameterFunction()
