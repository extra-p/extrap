class MultiParameterTerm:
    
    def __init__(self):
        self.coefficient = 1
        self.compound_term_parameter_pairs = []

    def set_coefficient(self, coefficient):
        self.coefficient = coefficient
    
    def get_coefficient(self):
        return self.coefficient

    def add_compound_term_parameter_pair(self, compound_term_parameter_pair):
        self.compound_term_parameter_pairs.append(compound_term_parameter_pair)

    def get_compound_term_parameter_pairs(self):
        return self.compound_term_parameter_pairs

    def evaluate(self, parameter_value_pairs):
        function_value = self.coefficient
        for i in range(len(self.compound_term_parameter_pairs)):
            paramter_value = parameter_value_pairs[self.compound_term_parameter_pairs[i].get_parameter().get_name()]
            function_value *= self.compound_term_parameter_pairs[i].get_compound_term().evaluate(paramter_value)
        return function_value

    def to_string(self, exact=False):
        if exact == True:
            function_string = "+" + str(self.coefficient)
            for i in range(len(self.compound_term_parameter_pairs)):
                function_string += self.compound_term_parameter_pairs[i].get_compound_term().to_string(self.compound_term_parameter_pairs[i].get_parameter(), exact)
            return function_string
        else:
            function_string = "+{:.2E}".format(self.coefficient)
            for i in range(len(self.compound_term_parameter_pairs)):
                function_string += self.compound_term_parameter_pairs[i].get_compound_term().to_string(self.compound_term_parameter_pairs[i].get_parameter())
            return function_string
                
            
        #function_string = "+" + str(self.coefficient)
        #for i in range(len(self.compound_term_parameter_pairs)):
        #    function_string += self.compound_term_parameter_pairs[i].get_compound_term().to_string(self.compound_term_parameter_pairs[i].get_parameter())
        #return function_string
    
    