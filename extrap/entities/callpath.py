# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import itertools

from extrap.util.serialization_schema import make_value_schema


class Callpath:
    """
    This class represents a callpath of an application.
    """
    """
    Counter for global callpath ids
    """
    ID_COUNTER = itertools.count()
    EMPTY: 'Callpath'

    def __init__(self, name):
        """
        Initialize callpath object.
        """
        self.name = name
        self.id = next(Callpath.ID_COUNTER)

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


CallpathSchema = make_value_schema(Callpath, 'name')
Callpath.EMPTY = Callpath(None)
