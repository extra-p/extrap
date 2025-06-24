# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
import importlib.util
from bisect import bisect_left
from collections import namedtuple
from itertools import groupby
from typing import Sequence

import numpy as np

from extrap.entities.functions import SingleParameterFunction, ConstantFunction
from extrap.entities.hypotheses import SingleParameterHypothesis, ConstantHypothesis
from extrap.entities.measurement import Measurement, Measure
from extrap.entities.model import Model
from extrap.entities.terms import CompoundTerm
from extrap.modelers.modeler_options import modeler_options
from extrap.modelers.single_parameter.abstract_base import AbstractSingleParameterModeler
from extrap.util.exceptions import RecoverableError
from extrap.util.progress_bar import DUMMY_PROGRESS

_adaptive_modeler_package = importlib.util.find_spec('extrap_adaptive_modeler')
if _adaptive_modeler_package is not None:
    from extrap_adaptive_modeler.load_model import get_model
    from extrap_adaptive_modeler.lazy_tensorflow import load_tensorflow

from ..basic import SingleParameterModeler

_NOISE_CATEGORIES = [0.0125, 0.025, 0.05, .1, .2, .4, .8]

# The following constants are fixed for the provided ML model
_BUCKET_COORDINATES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
_TERMS = [CompoundTerm.create(*expo) for expo in
          [(0, 1, 0), (0, 1, 1), (0, 1, 2), (1, 4, 0), (1, 3, 0),
           (1, 4, 1), (1, 3, 1), (1, 4, 2), (1, 3, 2), (1, 2, 0),
           (1, 2, 1), (1, 2, 2), (2, 3, 0), (3, 4, 0), (2, 3, 1),
           (3, 4, 1), (4, 5, 0), (2, 3, 2), (3, 4, 2), (1, 1, 0),
           (1, 1, 1), (1, 1, 2), (5, 4, 0), (5, 4, 1), (4, 3, 0),
           (4, 3, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2), (5, 3, 0),
           (7, 4, 0), (2, 1, 0), (2, 1, 1), (2, 1, 2), (9, 4, 0),
           (7, 3, 0), (5, 2, 0), (5, 2, 1), (5, 2, 2), (8, 3, 0),
           (11, 4, 0), (3, 1, 0), (3, 1, 1)]]

_PresetOptions = namedtuple('PresetOptions',
                            ['threshold', 'retrain_epochs', 'retrain_examples_per_class', 'noise_aware'])


@modeler_options
class AdaptiveModeler(AbstractSingleParameterModeler):
    NAME = "Adaptive"

    PRESETS = {
        -2: _PresetOptions(0.050301028, 1, 20, False),
        -1: _PresetOptions(0.035916109, 1, 100, False),
        0: _PresetOptions(0.031965664, 1, 200, False),
        1: _PresetOptions(0.0255837, 1, 2000, False),
        2: _PresetOptions(0.0255837, 1, 2000, True)
    }

    NOISE_CI = 2

    _cached_mlmodels = {}

    def _set_preset(self, value: int):
        self._preset = value
        p = self.PRESETS[value]
        self._threshold, self.retrain_epochs, self.retrain_examples_per_class, self.noise_aware = p

    preset = modeler_options.add(0, int, range=range(-2, 2), on_change=_set_preset)

    def __init__(self, use_median=False):
        super().__init__(use_median)
        self.preset = 0
        self.no_constants = False
        self._basic_modeler: AbstractSingleParameterModeler = SingleParameterModeler()

    def model(self, measurement_list_: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:
        if _adaptive_modeler_package is None:
            raise RecoverableError(
                "To use the adaptive modeler, please install Extra-P with the adaptive modeler extension.\n"
                "You can do that using 'pip install extrap[adaptive_modeling]'.")

        all_models = []
        for _, measurement_group in groupby(measurement_list_,
                                            key=lambda ms: ({m.coordinate[0] for m in ms}, ms[0].metric)):
            measurement_list = list(measurement_group)

            positions = np.array([m.coordinate[0] for m in measurement_list[0]])

            mean_values = np.array([np.fromiter((m.mean for m in ml), float, len(ml)) for ml in measurement_list])
            min_values = np.array([np.array([m.minimum for m in ml]) for ml in measurement_list])
            max_values = np.array([np.array([m.maximum for m in ml]) for ml in measurement_list])
            if self.use_measure == Measure.MEAN:
                values = mean_values
            elif self.use_measure == Measure.MEDIAN:
                values = np.array([np.array([m.median for m in ml]) for ml in measurement_list])
            elif self.use_measure == Measure.MINIMUM:
                values = min_values
            elif self.use_measure == Measure.MAXIMUM:
                values = max_values
            else:
                raise ValueError(f"Unsupported measure: {self.use_measure}")

            noise = self.get_noise(mean_values, min_values, max_values)

            functions = self._predict_functions(positions, values, noise)
            if functions is not None:
                models = self._create_ml_models(functions, measurement_list)

                self._update_with_additional_basic_models(measurement_list, models, noise)
                all_models.extend(models)
        return all_models

    def _update_with_additional_basic_models(self, measurement_list, models, noise):
        for i, measurements in enumerate(measurement_list):
            if noise[i] >= self._threshold * self.NOISE_CI:
                continue
            basic_model = self._basic_modeler.model([measurements])[0]
            if models[i].hypothesis.SMAPE < basic_model.hypothesis.SMAPE:
                models[i] = basic_model

    def _create_ml_models(self, functions, measurement_list):
        models = []
        for i, function in enumerate(functions):
            if not function.compound_terms[0].simple_terms:
                hypothesis = ConstantHypothesis(ConstantFunction(), self.use_measure)
            else:
                hypothesis = SingleParameterHypothesis(function, self.use_measure)
            hypothesis.compute_coefficients(measurement_list[i])
            hypothesis.compute_cost(measurement_list[i])
            models.append(Model(hypothesis))
        return models

    @staticmethod
    def _preprocess(positions, values):
        max_p = np.max(positions)
        max_sp = _BUCKET_COORDINATES[-1]
        return (positions * max_sp) / max_p, values / positions

    def _predict_functions(self, positions, values, noise_orig):
        tf = load_tensorflow()

        transformed_points, transformed_values_list = self._preprocess(positions, values)
        bucket_indices = self._bucketize([(p, i + 1) for i, p in enumerate(transformed_points)])
        noise_category_list = self._noise_category(noise_orig)

        top_k_classes = []
        for noise_category, groups in groupby(zip(noise_category_list, transformed_values_list),
                                              key=lambda ms: ms[0]):
            transformed_values = np.array([v for _, v in groups])
            ml_model, self._cached_mlmodels = get_model(bucket_indices, noise_category, positions, tf,
                                                        self.retrain_epochs, self.retrain_examples_per_class,
                                                        self._cached_mlmodels)

            bucket_values = np.zeros((len(transformed_values), len(bucket_indices)))
            for bi, pi in enumerate(bucket_indices):
                if pi > 0:
                    bucket_values[:, bi] = transformed_values[:, pi - 1]

            # do not use model.predict() to prevent memory leak
            prediction = ml_model(bucket_values)
            top_k_classes.extend([int(c) for c in tf.math.top_k(prediction, 1).indices])
        terms = (_TERMS[c] for c in top_k_classes)
        functions = (SingleParameterFunction(copy.copy(t)) for t in terms)
        return functions

    def get_noise(self, mean_values, min_values, max_values):
        mean_values = mean_values.copy()
        mean_values[mean_values == 0] = 1
        rel_max = np.nanmax(max_values / mean_values, axis=1)
        rel_min = np.nanmin(min_values / mean_values, axis=1)
        return rel_max - rel_min

    def _noise_category(self, noise):
        if not self.noise_aware:
            return np.full_like(noise, 0.2)
        categories = _NOISE_CATEGORIES
        thresholds = [(categories[i] + categories[i + 1]) / 2 for i in range(len(categories) - 1)]
        idx = np.searchsorted(thresholds, noise)
        return np.array(categories)[idx]

    # def create_model_simple(self, legacy_info, is_exponential=False):
    #     if isinstance(self._modeler, EXTRAP.SingleParameterSimpleModelGenerator):
    #         self._modeler.generateHypothesisBuildingBlockSet([], [])
    #         terms = self.exponential_terms if is_exponential else self.terms
    #         for exp in terms:
    #             term = EXTRAP.CompoundTerm.fromLegacy(*exp)
    #             self._modeler.addHypothesisBuildingBlock(term)
    #         return self._modeler.createModel(*legacy_info)
    #     else:
    #         raise TypeError()

    @staticmethod
    def _bucketize(points):
        bucket_count = len(_BUCKET_COORDINATES)
        point_buckets = [0] * bucket_count
        point_buckets_delta = [float('inf')] * bucket_count
        for p, v in points:
            idx = bisect_left(_BUCKET_COORDINATES, p)
            if idx >= bucket_count:
                idx = bucket_count - 1
            sp1 = _BUCKET_COORDINATES[idx]
            if idx - 1 >= 0:
                sp2 = _BUCKET_COORDINATES[idx - 1]
            else:
                sp2 = float('-inf')
            if (sp1 - p <= p - sp2) and point_buckets_delta[idx] >= sp1 - p:
                point_buckets_delta[idx] = sp1 - p
                point_buckets[idx] = v
            if (sp1 - p >= p - sp2) and point_buckets_delta[idx - 1] >= p - sp2:
                point_buckets_delta[idx - 1] = p - sp2
                point_buckets[idx - 1] = v
        return point_buckets
