# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from abc import ABC, abstractmethod
from collections import defaultdict
from numbers import Number
from typing import Union, List, Tuple, Dict, Sequence, Optional

import numpy
from marshmallow import fields

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import Node, CallTree
from extrap.entities.function_computation import ComputationFunction, ComputationFunctionSchema
from extrap.entities.functions import FunctionSchema
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.entities.named_entity import TAG_SEPARATOR
from extrap.modelers.aggregation import Aggregation, AggregatedModel
from extrap.util.classproperty import classproperty
from extrap.util.progress_bar import DUMMY_PROGRESS

numeric_array_t = Union[Number, numpy.ndarray]


class BinaryAggregationFunction(ComputationFunction, ABC):
    def __init__(self, function_terms):
        """
        Initialize a Function object.
        """
        super().__init__(None)
        self.raw_terms = function_terms
        self.aggregate()

    @abstractmethod
    def aggregate(self):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, BinaryAggregationFunction):
            return NotImplemented
        elif self is other:
            return True
        else:
            result = self.raw_terms == other.raw_terms
            result = result and super().__eq__(other)
            return result


class BinaryAggregationFunctionSchema(ComputationFunctionSchema):
    raw_terms = fields.List(fields.Nested(FunctionSchema))

    def create_object(self):
        return NotImplemented, BinaryAggregationFunction

    def postprocess_object(self, obj: BinaryAggregationFunction) -> BinaryAggregationFunction:
        if not obj.sympy_function:
            obj.aggregate()
        return obj


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
        calltree.ensure_callpaths_exist()
        progress_bar.total += len(models)
        result = {}
        for metric in metrics:
            if metric.lookup_tag(self.TAG_DISABLED, False, suffix=self.tag_suffix):
                continue
            self.walk_nodes(result, calltree, models, metric, progress_bar=progress_bar)
            if (None, metric) in result:
                del result[(None, metric)]

        return result

    def walk_nodes(self, result: Dict[Tuple[Callpath, Metric], Model], node: Node,
                   models: Dict[Tuple[Callpath, Metric], Model], metric: Metric, path='', progress_bar=DUMMY_PROGRESS):
        agg_models: Dict[Optional[str], List[Model]] = defaultdict(list)
        callpath = node.path if node.path else Callpath.EMPTY
        own_category = callpath.lookup_tag(self.TAG_CATEGORY, suffix=self.tag_suffix)

        key = (callpath, metric)
        if key in models:
            own_model = models[key]
            agg_models[own_category].append(own_model)
        else:
            own_model = None
            progress_bar.total += 1

        for c in node:
            res_models = self.walk_nodes(result, c, models, metric, path, progress_bar)
            for category, model in res_models.items():
                if model is not None:
                    agg_models[category].append(model)

        res_models: Dict[Optional[str], Optional[Model]] = {}
        for category in agg_models.keys():
            if not agg_models[category]:
                res_models[category] = None
            elif callpath.lookup_tag(self.TAG_DISABLED, False, suffix=self.tag_suffix):
                res_models[category] = own_model
            else:
                if len(agg_models[category]) == 1:
                    res_models[category] = agg_models[category][0]
                else:
                    measurements = self.aggregate_measurements(agg_models[category])
                    res_models[category] = self.aggregate_model(agg_models[category], callpath, measurements, metric)
                    res_models[category].measurements = measurements

        if res_models.get(own_category) is not None:
            result[(node.path, metric)] = res_models[own_category]

        for category, model in res_models.items():
            if category is None:
                continue
            elif category == own_category:
                continue
            elif res_models[category] is None:
                continue

            category_node = node.find_child(category)
            if not category_node:
                if isinstance(node, CallTree):
                    category_path = Callpath(category)
                    category_path.tags[self.TAG_CATEGORY] = category
                    if self.tag_suffix:
                        category_path.tags[self.TAG_CATEGORY + TAG_SEPARATOR + self.tag_suffix] = category
                else:
                    category_path = node.path.concat(category)
                    category_path.tags[self.TAG_CATEGORY] = category
                    if self.tag_suffix:
                        category_path.tags[self.TAG_CATEGORY + TAG_SEPARATOR + self.tag_suffix] = category
                category_node = Node(category, category_path)
                node.add_child_node(category_node)

            result[(category_node.path, metric)] = res_models[category]

        progress_bar.update(1)

        # check how model may be used in aggregated model of parent
        usage_disabled = callpath.lookup_tag(self.TAG_USAGE_DISABLED, False, suffix=self.tag_suffix)
        if usage_disabled:
            if usage_disabled == self.TAG_USAGE_DISABLED_agg_model:
                return {own_category: own_model}
            return {}

        return res_models

    @abstractmethod
    def aggregate_model(self, agg_models, callpath: Callpath, measurements: Sequence[Measurement],
                        metric: Metric) -> AggregatedModel:
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
            if not model.measurements:
                continue
            for m in model.measurements:
                agg = data[m.coordinate]
                agg.mean = self.binary_operator(agg.mean, m.mean)
                agg.median = self.binary_operator(agg.median, m.median)
                agg.maximum = self.binary_operator(agg.maximum, m.maximum)
                agg.minimum = self.binary_operator(agg.minimum, m.minimum)
                agg.std = self.binary_operator(agg.std, m.std)
        return list(data.values())
