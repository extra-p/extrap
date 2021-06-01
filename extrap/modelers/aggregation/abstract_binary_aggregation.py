# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from abc import ABC, abstractmethod
from numbers import Number
from typing import Callable, Union, List, Tuple, Dict, Sequence, Mapping

import numpy

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import Node
from extrap.entities.functions import Function, SingleParameterFunction, MultiParameterFunction, ConstantFunction
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.entities.parameter import Parameter
from extrap.entities.terms import CompoundTerm
from extrap.modelers.aggregation import Aggregation
from extrap.util.classproperty import classproperty
from extrap.util.progress_bar import DUMMY_PROGRESS

numeric_array_t = Union[Number, numpy.ndarray]


class BinaryAggregationFunction(Function, ABC):
    def __init__(self, function_terms):
        """
        Initialize a Function object.
        """
        super().__init__()
        self.compound_terms = function_terms
        self.raw_terms = function_terms
        self.aggregate()

    @abstractmethod
    def evaluate(self, parameter_value):
        raise NotImplementedError

    @abstractmethod
    def aggregate(self):
        raise NotImplementedError

    @abstractmethod
    def to_string(self, *parameters: Union[str, Parameter]):
        """
        Return a string representation of the function.
        """
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, BinaryAggregationFunction):
            return NotImplemented
        elif self is other:
            return True
        else:
            return self.raw_terms == other.raw_terms and \
                   self.compound_terms == other.compound_terms and \
                   self.constant_coefficient == other.constant_coefficient


class BinaryAggregation(Aggregation, ABC):

    @classproperty
    @abstractmethod
    def TAG_DISABLED(cls) -> str:  # noqa
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def TAG_USAGE_DISABLED(cls) -> str:  # noqa
        raise NotImplementedError

    @abstractmethod
    def binary_operator(self, a: numeric_array_t, b: numeric_array_t) -> numeric_array_t:
        raise NotImplementedError

    @property
    @abstractmethod
    def _OPERATION_NAME(self) -> str:  # noqa
        raise NotImplementedError

    def aggregate(self, models, calltree, metrics, progress_bar=DUMMY_PROGRESS):
        progress_bar.total += len(models)
        result = {}
        for metric in metrics:
            if metric.lookup_tag(self.TAG_DISABLED, False):
                continue
            self.walk_nodes(result, calltree, models, metric, progress_bar=progress_bar)

        return result

    def walk_nodes(self, result: Dict[Tuple[Callpath, Metric], Model], node: Node,
                   models: Dict[Tuple[Callpath, Metric], Model], metric: Metric, path='', progress_bar=DUMMY_PROGRESS):
        agg_models: List[Model] = []
        if node.name:
            if path == "":
                path = node.name
            else:
                path = path + '->' + node.name

        callpath = node.path if node.path else Callpath(path)
        key = (callpath, metric)
        if key in models:
            own_model = models[key]
            agg_models.append(own_model)
        else:
            own_model = None
            progress_bar.total += 1

        for c in node:
            model = self.walk_nodes(result, c, models, metric, path, progress_bar)
            if model is not None:
                agg_models.append(model)

        if not agg_models:
            model = None
        elif callpath.lookup_tag(self.TAG_DISABLED, False):
            model = own_model
        else:
            if len(agg_models) == 1:
                model = agg_models[0]
            else:
                measurements = self.aggregate_measurements(agg_models)
                model = self.aggregate_model(agg_models, callpath, measurements, metric)
                model.measurements = measurements

        if model is not None:
            if node.path == Callpath.EMPTY:
                node.path = callpath
            result[(node.path, metric)] = model

        progress_bar.update(1)

        # check how model may be used in aggregated model of parent
        usage_disabled = callpath.lookup_tag(self.TAG_USAGE_DISABLED, False)
        if usage_disabled:
            if usage_disabled == self.TAG_USAGE_DISABLED_agg_model:
                return own_model
            return None

        return model

    @abstractmethod
    def aggregate_model(self, agg_models, callpath: Callpath, measurements: Sequence[Measurement], metric: Metric):
        raise NotImplementedError

    def aggregate_measurements(self, agg_models: List[Model]):
        rest = iter(agg_models)
        first = next(rest)
        data = {}
        for m in first.measurements:
            agg = Measurement(m.coordinate, m.callpath, m.metric, None)
            agg.mean = m.mean
            agg.median = m.median
            agg.maximum = m.maximum
            agg.minimum = m.minimum
            agg.std = m.std
            data[m.coordinate] = agg
        for model in rest:
            for m in model.measurements:
                agg = data[m.coordinate]
                agg.mean = self.binary_operator(agg.mean, m.mean)
                agg.median = self.binary_operator(agg.median, m.median)
                agg.maximum = self.binary_operator(agg.maximum, m.maximum)
                agg.minimum = self.binary_operator(agg.minimum, m.minimum)
                agg.std = self.binary_operator(agg.std, m.std)
        return list(data.values())
