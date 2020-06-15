"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


class MultiParameterFunction:
    
    def __init__(self):
        self.constant_coefficient = 1
        self.multi_parameter_terms = []
    
    def add_multi_parameter_term(self, multi_parameter_term):
        self.multi_parameter_terms.append(multi_parameter_term)

    def get_multi_parameter_terms(self):
        return self.multi_parameter_terms
    
    def get_multi_parameter_term(self, multi_parameter_term_id):
        """
        Return the multi parameter term of the given id of the function.
        """
        return self.multi_parameter_terms[multi_parameter_term_id]

    def set_constant_coefficient(self, constant_coefficient):
        self.constant_coefficient = constant_coefficient

    def get_constant_coefficient(self):
        return self.constant_coefficient

    def evaluate(self, parameter_value_pairs):
        function_value = self.constant_coefficient
        for i in range(len(self.multi_parameter_terms)):
            function_value += self.multi_parameter_terms[i].evaluate(parameter_value_pairs)
        return function_value

    def to_string(self, exact=False):
        """
        Return a string representation of the function.
        """
        if exact == True:
            function_string = str(self.constant_coefficient)
            for i in range(len(self.multi_parameter_terms)):
                function_string += self.multi_parameter_terms[i].to_string(exact)
            return function_string
        else:
            function_string = "{:.2E}".format(self.constant_coefficient)
            for i in range(len(self.multi_parameter_terms)):
                function_string += self.multi_parameter_terms[i].to_string()
            return function_string
        
        #function_string = str(self.constant_coefficient)
        #for i in range(len(self.multi_parameter_terms)):
        #    function_string += self.multi_parameter_terms[i].to_string()
        #return function_string
    
    