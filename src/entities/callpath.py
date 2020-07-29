"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""
import itertools
from util.deprecation import deprecated


class Callpath:
    """
    This class represents a callpath of an application.
    """
    """
    Counter for global callpath ids
    """
    ID_COUNTER = itertools.count()

    def __init__(self, name):
        """
        Initialize callpath object.
        """
        self.name = name
        self.id = next(Callpath.ID_COUNTER)

    @deprecated("Use property directly.")
    def get_name(self):
        """
        Return the name of a callpath object.
        """
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Callpath):
            return NotImplemented
        return self is other or self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Callpath({self.name})"
