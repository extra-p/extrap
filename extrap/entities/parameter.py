# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import itertools

from extrap.util.serialization_schema import make_value_schema


class Parameter:
    """
    Counter for global parameter ids
    """
    ID_COUNTER = itertools.count()

    def __init__(self, name):
        self.name = name
        self.id = next(Parameter.ID_COUNTER)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            return NotImplemented
        return self is other or self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Parameter({self.name})"


ParameterSchema = make_value_schema(Parameter, 'name')
