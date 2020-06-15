"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


import numpy
import logging

    
class SingleParameterHypothesis:
    """
    This class represents a single parameter hypothesis, it is used to represent
    a performance function for one parameter. The modeler calls many of these objects
    to find the best model that fits the data.
    """
    
    
    def __init__(self, function, median):
        """
        Initialize SingleParameterHypothesis object.
        """
        self.function = function
        self.RSS = 0
        self.rRSS = 0
        self.SMAPE = 0
        self.AR2 = 0
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
    
    
    def clean_constant_coefficient(self, phi, training_measurements):
        """
        This function is used to correct numerical imprecision in the caculations,
        when the constant coefficient should be zero but is instead very small.
        We take into account the minimum data value to make sure that we don't "nullify"
        actually relevant numbers.
        """
        minimum = 0
        for training_measurements_id in range(len(training_measurements)):
            if training_measurements_id == 0:
                if self.median == True:
                    minimum = training_measurements[training_measurements_id].get_value_median()
                else:
                    minimum = training_measurements[training_measurements_id].get_value_mean()
            else:
                if self.median == True:
                    if training_measurements[training_measurements_id].get_value_median() < minimum:
                        minimum = training_measurements[training_measurements_id].get_value_median()
                else:
                    if training_measurements[training_measurements_id].get_value_mean() < minimum:
                        minimum = training_measurements[training_measurements_id].get_value_mean()

        if abs(self.function.get_constant_coefficient() / minimum) < phi:
            self.function.set_constant_coefficient(0)
    

    def compute_cost(self, training_measurements, validation_measurement, validation_coordinate):
        """
        Compute the cost for the single parameter model using leave one out crossvalidation.
        """
        _, value = validation_coordinate.get_parameter_value(0)
        predicted = self.function.evaluate(value)
        if self.median == True:
            actual = validation_measurement.get_value_median()
        else:
            actual = validation_measurement.get_value_mean()
        difference = predicted - actual
        self.RSS += difference * difference
        if self.median == True:
            relative_difference = difference / validation_measurement.get_value_median()
        else:
            relative_difference = difference / validation_measurement.get_value_mean()
        self.rRSS += relative_difference * relative_difference
        abssum = abs(actual) + abs(predicted) 
        if abssum != 0:
            self.SMAPE += (abs(difference) / abssum * 2) / len(training_measurements) * 100
        
        
    def compute_adjusted_rsquared(self, TSS, measurements):
        """
        Compute the adjusted R^2 for the hypothesis.
        """
        adjR = 1.0 - (self.RSS / TSS)
        degrees_freedom = len(measurements) - len(self.function.get_compound_terms()) - 1
        self.AR2 = ( 1.0 - (1.0 - adjR) * (len(measurements) - 1.0) / degrees_freedom )
  
  
    def compute_coefficients(self, measurements, coordinates):
        """
        Computes the coefficients of the function using the least squares solution.
        """
        hypothesis_total_terms = len(self.function.get_compound_terms()) + 1
        
        # creating a numpy matrix representation of the lgs
        a_list = []
        b_list = []
        for element_id in range(len(measurements)):
            if self.median == True:
                value = measurements[element_id].get_value_median()
            else:
                value = measurements[element_id].get_value_mean()
            list_element = []
            for compound_term_id in range(hypothesis_total_terms):
                if compound_term_id == 0:
                    list_element.append(1)
                else:
                    compound_term = self.function.get_compound_term(compound_term_id-1)
                    _, parameter_value = coordinates[element_id].get_parameter_value(0)
                    compound_term_value = compound_term.evaluate(parameter_value)
                    list_element.append(compound_term_value)
            a_list.append(list_element)
            b_list.append(value)
            #logging.debug(str(list_element)+"[x]=["+str(value)+"]")
            
        # solving the lgs for X to get the coefficients
        A = numpy.array(a_list)
        B = numpy.array(b_list)
        X = numpy.linalg.lstsq(A,B,None)
        #logging.debug("Coefficients:"+str(X[0]))
        
        # setting the coefficients for the hypothesis
        self.function.set_constant_coefficient(X[0][0])
        for compound_term_id in range(hypothesis_total_terms-1):
            self.function.get_compound_term(compound_term_id).set_coefficient(X[0][compound_term_id+1])
            
            
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
        compound_terms = self.function.get_compound_terms()
        maximum_term_contribution = 0
        for element_id in range(len(measurements)):
            _, parameter_value = coordinates[element_id].get_parameter_value(0)
            if self.median == True:
                contribution = abs( compound_terms[term_index].evaluate(parameter_value) / measurements[element_id].get_value_median())
            else:
                contribution = abs( compound_terms[term_index].evaluate(parameter_value) / measurements[element_id].get_value_mean())
            if contribution > maximum_term_contribution:
                maximum_term_contribution = contribution
        return maximum_term_contribution
    
    