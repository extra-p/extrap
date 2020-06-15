"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


class Metric:
    """
    This class represents a metric such as time or FLOPS.
    """
    
    
    def __init__(self, name):
        """
        Initializes the metric object.
        """
        self.name = name
    
    
    def get_name(self):
        """
        Returns the name of the metric.
        """
        return self.name
    
    