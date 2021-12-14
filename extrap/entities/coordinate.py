# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import itertools
from typing import Union, Iterable

from extrap.util.serialization_schema import make_value_schema


class Coordinate:
    """
    This class represents a coordinate, i.e. a point where a measurement is taken.
    """
    """
    Counter for global coordinate ids
    """
    ID_COUNTER = itertools.count()

    def __init__(self, *parts: Union[Iterable[float], float]):
        """
        Initialize the coordinate object.
        """
        if len(parts) == 1 and isinstance(parts[0], Iterable):
            self._values = tuple(parts[0])
        else:
            self._values = parts

        self.id = next(Coordinate.ID_COUNTER)

    @property
    def dimensions(self):
        """
        Returns the number of dimensions of the coordinate.
        """
        return len(self._values)

    def __str__(self):
        """
        Returns a string representation of the coordinate.
        """
        return str(self._values)

    def __getitem__(self, param: int):
        """
        Returns the value for a parameter.
        """
        assert param >= 0
        return self._values[param]

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

    def as_partial_tuple(self, except_param):
        return tuple(c for p, c in enumerate(self._values) if p != except_param)

    def is_mostly_lower(self, other: 'Coordinate', except_param):
        return all(a <= b
                   for i, (a, b) in enumerate(zip(self._values, other._values))
                   if i != except_param)

    def is_mostly_equal(self, other: 'Coordinate', except_param):
        return all(a == b
                   for i, (a, b) in enumerate(zip(self._values, other._values))
                   if i != except_param)


CoordinateSchema = make_value_schema(Coordinate, '_values')
