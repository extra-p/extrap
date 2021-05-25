# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import numpy as np

from extrap.entities.callpath import Callpath
from extrap.modelers.aggregation.abstract_binary_aggregation import BinaryAggregation, numeric_array_t, \
    SumAggregationFunction

from typing import Callable, Union, List, Tuple, Dict, Sequence, Mapping
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.model import Model


class SumAggregation(BinaryAggregation):
    NAME = 'Sum'

    def binary_operator(self, a, b):
        return a + b

    def aggregate_model(self, agg_models, callpath: Callpath, measurements: Sequence[Measurement], metric: Metric):
        function = SumAggregationFunction([m.hypothesis.function for m in agg_models])
        hypothesis_type = type(agg_models[0].hypothesis)
        hypothesis = hypothesis_type(function, agg_models[0].hypothesis._use_median)
        hypothesis.compute_cost(measurements)
        model = Model(hypothesis, callpath, metric)
        return model

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
