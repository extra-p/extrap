"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


class ConstantHypothesis:
    """
    This class represents a constant hypothesis, it is used to represent a performance
    function that is not affected by the input value of a parameter. The modeler calls this
    class first to see if there is a constant model that describes the data best.
    """
    
    
    def __init__(self, function, median):
        """
        Initialize the ConstantHypothesis.
        """
        self.function = function
        self.RSS = 0
        self.SMAPE = 0
        self.AR2 = 0
        self.rRSS = 0
        self.median = median
        
    
    def set_RSS(self, rss):
        self.RSS = rss

    def set_SMAPE(self, smape):
        self.SMAPE = smape
        
    def set_AR2(self, ar2):
        self.AR2 = ar2
        
    def set_rRSS(self, rrss):
        self.rRSS = rrss
        
    
    #TODO: should this be calculated?
    def get_AR2(self):
        return 1
        
        
    def get_function(self):
        """
        Returns the function string.
        """
        return self.function
        
    
    def get_RSS(self):
        """
        Return the RSS of the hypothesis.
        """
        return self.RSS
    
    
    def get_SMAPE(self):
        """
        Return the SMAPE of the hypothesis.
        """
        return self.SMAPE
    
    
    def compute_cost(self, measurements):
        """
        Computes the cost of the constant hypothesis using all data points.
        """
        smape = 0
        for element_id in range(len(measurements)):
            #TODO: remove old code in comments
            #_, value = coordinates[element_id].get_parameter_value(0)
            #predicted = self.function.evaluate(value)
            predicted = self.function.get_constant_coefficient()
            if self.median == True:
                actual = measurements[element_id].get_value_median()
            else:
                actual = measurements[element_id].get_value_mean()
            #actual = measurements[element_id].get_value()
            difference = predicted - actual
            self.RSS += difference * difference
            abssum = abs(actual) + abs(predicted) 
            if abssum != 0:
                smape += abs(difference) / abssum * 2
        self.SMAPE = smape / len(measurements) * 100;
         
