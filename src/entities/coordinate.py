"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""
import itertools
from typing import Union, List, Tuple, Iterable

from entities.parameter import Parameter
from util.deprecation import deprecated
from util.serialization_schema import make_value_schema


class Coordinate:
    """
    This class represents a coordinate, i.e. a point where a measurement is taken.
    """
    """
    Counter for global coordinate ids
    """
    ID_COUNTER = itertools.count()

    def __init__(self, *parts: Union[List[Tuple[Parameter, float]], Iterable[float], float]):
        """
        Initialize the coordinate object.
        """
        self._parameters = ()
        if len(parts) == 1 and isinstance(parts[0], Iterable):
            parts = parts[0]
            if isinstance(parts, list) and len(parts) > 0 and isinstance(parts[0], tuple):
                deprecated.code()
                self._values = tuple(v for _, v in parts)
                self._parameters = tuple(p for p, _ in parts)
            else:
                self._values = tuple(parts)
        else:
            self._values = parts

        self.id = next(Coordinate.ID_COUNTER)

    @property
    def dimensions(self):
        """
        Returns the number of dimensions of the coordinate.
        """
        return len(self._values)

    @deprecated("No replacement. Coordinates are immutable.")
    def add_parameter_value(self, parameter, value):
        """
        Add a parameter-value combination to the coordinate.
        Increasing the number of dimensions by 1.
        """
        self._parameters = tuple(itertools.chain(self._parameters, (parameter,)))
        self._values = tuple(itertools.chain(self._values, (value,)))

    @deprecated("Use property directly.")
    def get_dimensions(self):
        """
        Returns the number of dimensions of the coordinate.
        """
        return self.dimensions

    @deprecated("Parameters are no longer in use.")
    def get_parameter_value(self, dimension):
        """
        Returns the parameter-value combination of the given dimension.
        """
        value = self._values[dimension]
        return dimension, value

    @deprecated("Use str(Coordinate).")
    def get_as_string(self):
        """
        Returns a string representation of the coordinate.
        """
        return str(self)

    def __str__(self):
        """
        Returns a string representation of the coordinate.
        """
        return str(self._values)

    def __getitem__(self, param):
        """
        Returns the value for a parameter.
        """
        if isinstance(param, int):
            return self._values[param]
        return self._values[self._parameters.index(param)]

    def __repr__(self):
        """
        Returns a string representation of the coordinate.
        """
        return f'Coordinate{str(self._values)}'

    def __hash__(self):
        return hash(self._values)

    def __eq__(self, other):
        if not isinstance(other, Coordinate):
            return NotImplemented
        return self is other or self._values == other._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __lt__(self, other):
        if not isinstance(other, Coordinate):
            return NotImplemented
        return self._values < other._values

    def as_tuple(self):
        return self._values


CoordinateSchema = make_value_schema(Coordinate, '_values')
