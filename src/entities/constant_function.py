"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


class ConstantFunction:
    """
    This class represents a constant function.
    """
    
    
    def __init__(self):
        """
        Initialize ConstantFunction object.
        """
        self.constant_coefficient = 1
        
        
    def set_constant_coefficient(self, constant_coefficient):
        """
        Set the value of the constant coefficient.
        """
        self.constant_coefficient = constant_coefficient


    def get_constant_coefficient(self):
        """
        Return the value of the constant coefficient.
        """
        return self.constant_coefficient
    
     
    def to_string(self, _=None, exact=False):
        """
        Returns a string representation of the constant function.
        """
        if exact == True:
            return str(self.constant_coefficient)
        else:
            return "{:.2E}".format(self.constant_coefficient)
    
