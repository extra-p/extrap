# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
import numpy as np

from extrap.modelers.aggregation.abstract_binary_aggregation import BinaryAggregation


class SumAggregation(BinaryAggregation):
    NAME = 'Sum'

    def binary_operator(self, a, b):
        return a + b



    TAG_DISABLED = 'agg__disabled__sum'
    TAG_USAGE_DISABLED = 'agg__usage_disabled__sum'
    _OPERATION_NAME = 'sum'


class MaxAggregation(BinaryAggregation):
    NAME = 'Max'

    def binary_operator(self, a, b):
        return np.maximum(a, b)

    TAG_DISABLED = 'agg__disabled__max'
    TAG_USAGE_DISABLED = 'agg__usage_disabled__max'
    _OPERATION_NAME = 'max'
