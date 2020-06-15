from entities.simple_term import SimpleTerm

class CompoundTerm:
    
    def __init__(self):
        self.coefficient = 1
        self.simple_terms = []

    def set_coefficient(self, coefficient):
        self.coefficient = coefficient

    def get_coefficient(self):
        return self.coefficient

    def get_simple_terms(self):
        return self.simple_terms

    def add_simple_term(self, simple_term):
        self.simple_terms.append(simple_term)
    
    def evaluate(self, paramter_value):
        function_value = self.coefficient
        for i in range(len(self.simple_terms)):
            function_value *= self.simple_terms[i].evaluate(paramter_value)
        return function_value

    def to_string(self, parameter, exact=False):
        if exact == True:
            function_string = "+" + str(self.coefficient)
            for i in range(len(self.simple_terms)):
                function_string += self.simple_terms[i].to_string(parameter, exact)
            return function_string
        else:
            function_string = "+{:.2E}".format(self.coefficient)
            for i in range(len(self.simple_terms)):
                function_string += self.simple_terms[i].to_string(parameter)
            return function_string

    def create_compound_term(self, a, b, c):
        compound_term = CompoundTerm()
        compound_term.set_coefficient(1)
        if a != 0:
            simple_term = SimpleTerm("polynomial", a/b)
            compound_term.add_simple_term(simple_term)
        if c != 0:
            simple_term = SimpleTerm("logarithm", c)
            compound_term.add_simple_term(simple_term)
        return compound_term

class CompoundTermParameterPair:
    
    def __init__(self, parameter, compound_term):
        self.parameter = parameter
        self.compound_term = compound_term

    def set_parameter(self, parameter):
        self.parameter = parameter
    
    def get_parameter(self):
        return self.parameter

    def set_compound_term(self, compound_term):
        self.compound_term = compound_term

    def get_compound_term(self):
        return self.compound_term