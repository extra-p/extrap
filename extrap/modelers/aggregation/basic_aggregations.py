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

    NOT_CALCULABLE_TAG = 'agg_sum__not_calculable'
    _OPERATION_NAME = 'sum'


class MaxAggregation(BinaryAggregation):
    NAME = 'Max'

    def binary_operator(self, a, b):
        return np.maximum(a, b)

    NOT_CALCULABLE_TAG = 'agg_max__not_calculable'
    _OPERATION_NAME = 'max'
