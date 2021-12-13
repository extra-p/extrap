# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import abc
from abc import ABC
from collections import defaultdict
from inspect import signature
from itertools import permutations
from typing import Union, Sequence, Callable, Iterable

from extrap.entities.calculation_element import CalculationElement
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.hypotheses import Hypothesis
from extrap.entities.model import Model
from extrap.util.extension_loader import load_extensions

REQUIRED_METRICS_NAME = '_required_metrics'

from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.util.classproperty import classproperty


class ConversionMetrics:
    def __init__(self, *required_metric: str):
        self.required_metrics = [Metric(name) for name in required_metric]

    def __call__(self, f):
        setattr(f, REQUIRED_METRICS_NAME, self.required_metrics)
        return f


class AbstractMetricConverter(ABC):
    Value = Union[Measurement, ComputationFunction]

    _conversions = []

    @abc.abstractmethod
    @classproperty
    def NAME(cls) -> str:
        pass

    new_metric: Metric

    def __init__(self):
        self._order_mapping = []

    def __init_subclass__(cls, **kwargs):
        cls.new_metric = Metric(cls.NAME)
        cls._conversions = cls._conversions.copy()
        for method_name in dir(cls):
            method = getattr(cls, method_name, None)
            if method:
                metrics = getattr(method, REQUIRED_METRICS_NAME, None)
                if metrics:
                    sig = signature(method)
                    if len(metrics) != len(sig.parameters) - 1:
                        raise SyntaxError("For each metric there must exist exactly one parameter, besides self.")
                    cls._conversions.append(method)
        if len(cls._conversions) < 2:
            raise TypeError("At least two conversion functions must be defined.")
        super().__init_subclass__(**kwargs)

    def check_and_initialise(self, *metric_set):
        conversions = self._determine_permutation(metric_set)
        if conversions:
            self._order_mapping = [p for p, _ in conversions]
            return True
        return False

    @classmethod
    def _determine_permutation(cls, metric_set):
        if len(metric_set) != len(cls._conversions):
            return False
        for conversions in permutations(enumerate(cls._conversions)):
            if all(
                    all(m in metric_set[i] for m in getattr(conversion_function, REQUIRED_METRICS_NAME))
                    for i, (_, conversion_function) in
                    enumerate(conversions)
            ):
                return conversions
        return None

    @classmethod
    def try_create(cls, *metric_set):
        conversions = cls._determine_permutation(metric_set)
        if conversions:
            result = cls()
            result._order_mapping = [p for p, _ in conversions]
            return result
        return None

    def get_required_metrics(self, i):
        conversion = self._conversions[self._order_mapping[i]]
        return getattr(conversion, REQUIRED_METRICS_NAME)

    def get_conversion(self, i) -> Callable[[AbstractMetricConverter, *CalculationElement], Value]:
        conversion = self._conversions[self._order_mapping[i]]
        return conversion

    def convert_measurements(self, i, measurement_sets: Iterable[Sequence[Measurement]]):
        conversion_function = self.get_conversion(i)
        if len(self.get_required_metrics(i)) == 1:
            result = [conversion_function(self, measurement) for measurement in next(iter(measurement_sets))]
        else:
            measurements_ordered = defaultdict(list)
            for measurements in measurement_sets:
                for m in measurements:
                    measurements_ordered[m.coordinate].append(m)

            result = [conversion_function(self, *measurements) for measurements in measurements_ordered.values()]
        for m in result:
            m.metric = self.new_metric
        return result

    def convert_models(self, i, model_sets: Iterable[Model], measurements):
        conversion_function = self.get_conversion(i)
        functions = [ComputationFunction(m.hypothesis.function) for m in model_sets]
        function = conversion_function(self, *functions)
        if type(function) == ComputationFunction and function.original_function:
            function = function.original_function
        model = next(iter(model_sets))
        hypothesis_class = Hypothesis.infer_best_type([m.hypothesis for m in model_sets])
        hypothesis = hypothesis_class(function, model.hypothesis.use_median)
        hypothesis.compute_cost(measurements)
        if hasattr(hypothesis, 'compute_adjusted_rsquared'):
            _, constant_cost = Hypothesis.calculate_constant_indicators(measurements, model.hypothesis.use_median)
            hypothesis.compute_adjusted_rsquared(constant_cost, measurements)
        result = Model(hypothesis, model.callpath, self.new_metric)
        result.measurements = measurements
        return result


all_conversions = load_extensions(__path__, __name__, AbstractMetricConverter)
