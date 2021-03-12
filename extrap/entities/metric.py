# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import itertools
import json

from extrap.entities.named_entity import NamedEntityWithTags
from extrap.util.serialization_schema import make_value_schema


class Metric(NamedEntityWithTags):
    """
    This class represents a metric such as time or FLOPS.
    """
    TYPENAME = "Metric"

    ID_COUNTER = itertools.count()
    """
    Counter for global metric ids
    """


MetricSchema = make_value_schema(Metric, '_data')
