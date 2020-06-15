"""
This file is part of the Extra-P software (https://github.com/MeaParvitas/Extra-P)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""


class Measurement:
    """
    This class represents a measurement, i.e. the value measured for a specific metric and callpath at a coordinate.
    """
    
    
    def __init__(self, coordinate_id, callpath_id, metric_id, value_mean, value_median):
        """
        Initialize the Measurement object.
        """
        self.coordinate_id = coordinate_id
        self.callpath_id = callpath_id
        self.metric_id = metric_id
        self.value_mean = value_mean
        self.value_median = value_median
        
    
    def get_callpath_id(self):
        """
        Return the callpath id of the measurement.
        """
        return self.callpath_id


    def get_metric_id(self):
        """
        Return the metric id of the measurement.
        """
        return self.metric_id
    
    
    def get_coordinate_id(self):
        """
        Return the coordinate id of the measurement.
        """
        return self.coordinate_id
    
    
    def get_value_mean(self):
        """
        Return the mean measured value.
        """
        return self.value_mean
    
    
    def get_value_median(self):
        """
        Return the median measured value.
        """
        return self.value_median
    
    
    def set_value_mean(self, value_mean):
        """
        Set the mean measured value to the given value.
        """
        self.value_mean = value_mean
    
    
    def set_value_median(self, value_median):
        """
        Set the median measured value to the given value.
        """
        self.value_median = value_median
    
        