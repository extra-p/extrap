"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


class SingleParameterFunction:
    """
    This class represents a single parameter function
    """
    
    
    def __init__(self):
        """
        Initialize a SingleParameterFunction object.
        """
        self.constant_coefficient = 1
        self.compound_terms = []


    def add_compound_term(self, compound_term):
        """
        Add a compound term to the single parameter function.
        """
        self.compound_terms.append(compound_term)


    def get_compound_terms(self):
        """
        Return all the compound terms of the function.
        """
        return self.compound_terms
    
    
    def get_compound_term(self, compound_term_id):
        """
        Return the compound term of the given id of the function.
        """
        return self.compound_terms[compound_term_id]
    
    
    def set_constant_coefficient(self, constant_coefficient):
        """
        Set the constant coefficient of the function to the given value.
        """
        self.constant_coefficient = constant_coefficient


    def get_constant_coefficient(self):
        """
        Return the constant coefficient of the function.
        """
        return self.constant_coefficient


    def evaluate(self, parameter_value):
        """
        Evalute the function according to the given value and return the result.
        """
        function_value = self.constant_coefficient
        for compound_term_id in range(len(self.compound_terms)):
            function_value += self.compound_terms[compound_term_id].evaluate(parameter_value)
        return function_value


    def to_string(self, parameter, exact=False):
        """
        Return a string representation of the function.
        """
        if exact == True:
            function_string = str(self.constant_coefficient)
            for compound_term_id in range(len(self.compound_terms)):
                function_string += self.compound_terms[compound_term_id].to_string(parameter, exact)
            return function_string
        else:
            function_string = "{:.2E}".format(self.constant_coefficient)
            for compound_term_id in range(len(self.compound_terms)):
                function_string += self.compound_terms[compound_term_id].to_string(parameter)
            return function_string
        
    
    