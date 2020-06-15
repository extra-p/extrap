"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


import numpy

    
class MultiParameterHypothesis:
    """
    This class represents a multi parameter hypothesis, it is used to represent
    a performance function with several parameters. However, it can have also
    only one parameter. The modeler calls many of these objects to find the best
    model that fits the data.
    """
    
    
    def __init__(self, function, median):
        """
        Initialize MultiParameterHypothesis object.
        """
        self.function = function
        self.RSS = 0
        self.rRSS = 0
        self.SMAPE = 0
        self.AR2 = 0
        self.RE = 0
        self.median = median


    def get_function(self):
        """
        Return the function.
        """
        return self.function
    
    
    def get_RSS(self):
        """
        Return the RSS.
        """
        return self.RSS


    def get_rRSS(self):
        """
        Return the rRSS.
        """
        return self.rRSS


    def get_AR2(self):
        """
        Return the AR2.
        """
        return self.AR2


    def get_SMAPE(self):
        """
        Return the SMAPE.
        """
        return self.SMAPE
    
    
    def get_RE(self):
        """
        Return the RE.
        """
        return self.RE
    
    
    def set_RSS(self, RSS):
        """
        Set the RSS.
        """
        self.RSS = RSS


    def set_rRSS(self, rRSS):
        """
        Set the rRSS.
        """
        self.rRSS = rRSS


    def set_AR2(self, AR2):
        """
        Set the AR2.
        """
        self.AR2 = AR2


    def set_SMAPE(self, SMAPE):
        """
        Set the SMAPE.
        """
        self.SMAPE = SMAPE
    
    
    def set_RE(self, RE):
        """
        Set the RE.
        """
        self.RE = RE
    
    
    def clean_constant_coefficient(self, phi, measurements):
        """
        This function is used to correct numerical imprecision in the caculations,
        when the constant coefficient should be zero but is instead very small.
        We take into account the minimum data value to make sure that we don't "nullify"
        actually relevant numbers.
        """
        minimum = 0
        for measurements_id in range(len(measurements)):
            if measurements_id == 0:
                if self.median == True:
                    minimum = measurements[measurements_id].get_value_median()
                else:
                    minimum = measurements[measurements_id].get_value_mean()
            else:
                if self.median == True:
                    if measurements[measurements_id].get_value_median() < minimum:
                        minimum = measurements[measurements_id].get_value_median()
                else:
                    if measurements[measurements_id].get_value_mean() < minimum:
                        minimum = measurements[measurements_id].get_value_mean()

        if abs(self.function.get_constant_coefficient() / minimum) < phi:
            self.function.set_constant_coefficient(0)
    

    def compute_cost(self, measurements, coordinates):
        """
        Compute the cost for a multi parameter hypothesis.
        """
        self.RSS = 0
        self.rRSS = 0
        smape = 0
        re_sum = 0
        
        for i in range(len(measurements)):
            measurement = measurements[i]
            coordinate_id = measurement.get_coordinate_id()
            coordinate = coordinates[coordinate_id]
            dimensions = coordinate.get_dimensions()
            parameter_value_pairs = {}
            for i in range(dimensions):
                parameter, value = coordinate.get_parameter_value(i)
                parameter_value_pairs[parameter.get_name()] = float(value)
            
            predicted = self.function.evaluate(parameter_value_pairs)
            #print(predicted)
            if self.median == True:
                actual = measurement.get_value_median()
            else:
                actual = measurement.get_value_mean()
            #print(actual)
            
            difference = predicted - actual
            #absolute_difference = abs(difference)
            abssum = abs(actual) + abs(predicted)

            # calculate relative error
            absolute_error = abs(predicted - actual)
            relative_error = absolute_error / actual
            re_sum = re_sum + relative_error

            self.RSS += difference * difference
            if self.median == True:
                relativeDifference = difference / measurement.get_value_median()
            else:
                relativeDifference = difference / measurement.get_value_mean()
            self.rRSS += relativeDifference * relativeDifference
    
            if abssum != 0.0:
                # This `if` condition prevents a division by zero, but it is correct: if sum is 0, both `actual` and `predicted`
                # must have been 0, and in that case the error at this point is 0, so we don't need to add anything.
                smape += abs(difference) / abssum * 2
      
        # times 100 for percentage error
        self.RE    = re_sum / len(measurements)
        self.SMAPE = smape / len(measurements) * 100
        

    def compute_adjusted_rsquared(self, TSS, measurements):
        """
        Compute the adjusted R^2 for the hypothesis.
        """
        self.AR2 = 0.0
        adjR = 1.0 - (self.RSS / TSS)
        counter = 0
        
        for i in range(len(self.function.get_multi_parameter_terms())):
            counter += len(self.function.get_multi_parameter_terms()[i].get_compound_term_parameter_pairs())
        
        degrees_freedom = len(measurements) - counter - 1
        self.AR2 = ( 1.0 - (1.0 - adjR) * (len(measurements) - 1.0) / degrees_freedom )
  
  
    def compute_coefficients(self, measurements, coordinates):
        """
        Computes the coefficients of the function using the least squares solution.
        """
        hypothesis_total_terms = len(self.function.get_multi_parameter_terms()) + 1
        
        # creating a numpy matrix representation of the lgs
        a_list = []
        b_list = []
        for element_id in range(len(measurements)):
            if self.median == True:
                value = measurements[element_id].get_value_median()
            else:
                value = measurements[element_id].get_value_mean()
            list_element = []
            for multi_parameter_term_id in range(hypothesis_total_terms):
                if multi_parameter_term_id == 0:
                    list_element.append(1)
                else:
                    multi_parameter_term = self.function.get_multi_parameter_term(multi_parameter_term_id-1)
                    #_, parameter_value = coordinates[element_id].get_parameter_value(0)
                    coordinate_id = measurements[element_id].get_coordinate_id()
                    coordinate = coordinates[coordinate_id]
                    dimensions = coordinate.get_dimensions()
                    parameter_value_pairs = {}
                    for i in range(dimensions):
                        parameter, value = coordinate.get_parameter_value(i)
                        parameter_value_pairs[parameter.get_name()] = float(value)
                    multi_parameter_term_value = multi_parameter_term.evaluate(parameter_value_pairs)
                    list_element.append(multi_parameter_term_value)
            a_list.append(list_element)
            b_list.append(value)
            #print(str(list_element)+"[x]=["+str(value)+"]")
            #logging.debug(str(list_element)+"[x]=["+str(value)+"]")
            
        # solving the lgs for X to get the coefficients
        A = numpy.array(a_list)
        B = numpy.array(b_list)
        X = numpy.linalg.lstsq(A,B,None)
        #print("Coefficients:"+str(X[0]))
        #logging.debug("Coefficients:"+str(X[0]))
        
        # setting the coefficients for the hypothesis
        self.function.set_constant_coefficient(X[0][0])
        for multi_parameter_term_id in range(hypothesis_total_terms-1):
            self.function.get_multi_parameter_term(multi_parameter_term_id).set_coefficient(X[0][multi_parameter_term_id+1])
            
            
    def is_valid(self):
        """
        Checks if there is a numeric imprecision. If this is the case the hypothesis will be ignored.
        """
        valid = not (self.RSS != self.RSS or abs(self.RSS) == float('inf'))
        return valid
        
        
    def calc_term_contribution(self, term_index, measurements, coordinates):
        """
        Calculates the term contribution of the term with the given term id to see if it is smaller than epsilon.
        """
        multi_parameter_terms = self.function.get_multi_parameter_terms()
        #compound_terms = self.function.get_compound_terms()
        maximum_term_contribution = 0
        for element_id in range(len(measurements)):
            #_, parameter_value = coordinates[element_id].get_parameter_value(0)
            dimensions = coordinates[element_id].get_dimensions()
            parameter_value_pairs = {}
            for i in range(dimensions):
                parameter, value = coordinates[element_id].get_parameter_value(i)
                parameter_value_pairs[parameter.get_name()] = float(value)
            if self.median == True:
                contribution = abs( multi_parameter_terms[term_index].evaluate(parameter_value_pairs) / measurements[element_id].get_value_median())
            else:
                contribution = abs( multi_parameter_terms[term_index].evaluate(parameter_value_pairs) / measurements[element_id].get_value_mean())
            if contribution > maximum_term_contribution:
                maximum_term_contribution = contribution
        return maximum_term_contribution
    
    