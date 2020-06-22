"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


class Coordinate:
    """
    This class represents a coordinate, i.e. a point where a measurement is taken.
    """

    def __init__(self, parampair=[]):
        """
        Initialize the coordinate object.
        """
        self.parameters = [p for p, v in parampair]
        self.values = [v for p, v in parampair]
        self.dimensions = 0

    def add_parameter_value(self, parameter, value):
        """
        Add a parameter-value combination to the coordinate.
        Increasing the number of dimensions by 1.
        """
        self.parameters.append(parameter)
        self.values.append(value)
        self.dimensions += 1

    def get_dimensions(self):
        """
        Returns the number of dimensions of the coordinate.
        """
        return self.dimensions

    def get_parameter_value(self, dimension):
        """
        Returns the parameter-value combination of the given dimension.
        """
        parameter = self.parameters[dimension]
        value = self.values[dimension]
        return parameter, value

    def get_as_string(self):
        """
        Returns a string representation of the coordinate.
        """
        return str(self)

    def __str__(self):
        """
        Returns a string representation of the coordinate.
        """
        coordinate_string = ""
        for element_id in range(self.dimensions):
            parameter = self.parameters[element_id]
            parameter_name = parameter.get_name()
            value = self.values[element_id]
            coordinate_string += "("+parameter_name+","+str(value)+")"
        return coordinate_string

    def __repr__(self):
        """
        Returns a string representation of the coordinate.
        """
        return f'Coordinate({str(self)})'

    def __hash__(self):
        items = (tuple(self.parameters), tuple(self.values))
        return hash(items)

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        return self is other or (
            self.parameters == other.parameters and
            self.values == self.values)
