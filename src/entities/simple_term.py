from math import log

class SimpleTerm:
    
    def __init__(self, term_type, exponent):
        self.term_type = term_type
        self.exponent = exponent

    def set_term_type(self, term_type):
        self.term_type = term_type

    def get_term_type(self):
        return self.term_type
    
    def set_exponent(self, exponent):
        self.exponent = exponent

    def get_exponent(self):
        return self.exponent

    def to_string(self, parameter, exact=False):
        if exact == True:
            if self.term_type == "polynomial":
                return "*(" + parameter.get_name() + "**" + str(self.exponent) +")"
            elif self.term_type == "logarithm":
                return "*log(" + parameter.get_name() + ",2)**" + str(self.exponent)
        else:
            if self.term_type == "polynomial":
                return "*(" + parameter.get_name() + "**{:.2E}".format(self.exponent) +")"
            elif self.term_type == "logarithm":
                return "*log(" + parameter.get_name() + ",2)**{:.2E}".format(self.exponent)

    def evaluate(self, parameter_value):
        if self.term_type == "polynomial":
            return pow(parameter_value, self.exponent)
        elif self.term_type == "logarithm":
            return pow(log(parameter_value, 2), self.exponent)