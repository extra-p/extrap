# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations
import copy
from typing import Sequence, Collection
import warnings
import numpy

from extrap.entities.callpath import Callpath
from extrap.entities.functions import SingleParameterFunction, Function
from extrap.entities.hypotheses import SingleParameterHypothesis, ConstantHypothesis
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.modelers.single_parameter.abstract_base import AbstractSingleParameterModeler
from extrap.modelers.single_parameter.basic import SingleParameterModeler
from extrap.util.exceptions import RecoverableError
from extrap.util.progress_bar import DUMMY_PROGRESS
from extrap.modelers.modeler_options import modeler_options


@modeler_options
class HardwareCounterModeler(AbstractSingleParameterModeler):
    NAME = 'Hardware-Counter'
    DESCRIPTION = 'Creates a runtime model based on data of hardware performance counters'

    number_used_counters = modeler_options.add(3, int, range=range(1, 5))
    no_negative_coefficients = modeler_options.add(True, bool)
    weighted_terms = modeler_options.add(False, bool, 'Uses all counters.')

    def __init__(self):
        """
        Initialize HardwareCounterModeler object.
        """
        super().__init__(use_median=False)

        self.min_measurement_points = 5
        self.counter_list = [
            "PAPI_LST_INS",
            "PAPI_LD_INS",
            "PAPI_SR_INS",
            "PAPI_BR_INS",
            "PAPI_BR_UCN",
            "PAPI_BR_CN",
            "PAPI_DP_OPS",
            "PAPI_SP_OPS",
            "PAPI_FP_OPS",
            "PAPI_FP_INS",
            "PAPI_VEC_DP",
            "PAPI_VEC_SP",
            "PAPI_VEC_INS",
        ]
        self.metrics_to_improve = ['time']
        self.total_count_metric = Metric('PAPI_TOT_INS')
        self.single_param_modeler = SingleParameterModeler()

    def determine_relevant_counter_models(self, callpath_models) -> list[Model]:

        actual_counter_list = self.determine_relevant_counters(callpath_models)

        counter_weights = []
        for model in callpath_models:
            metric = model.metric
            if str(metric) in actual_counter_list:
                # TODO maybe add median calculation
                val_sum = sum(measurement.mean for measurement in model.measurements)
                counter_weights.append((val_sum, model))

        counter_weights.sort(
            key=lambda x: x[0],
            reverse=True
        )
        if len(counter_weights) < self.number_used_counters:
            warnings.warn('You have selected more counters than available for modeling. '
                          f'The modeling will use the maximal number of available counters ({len(counter_weights)}).')
        relevant_models = [tpl[1] for tpl in counter_weights[:self.number_used_counters]]
        return relevant_models

    def determine_relevant_counters(self, callpath_models):
        actual_counter_list = set(self.counter_list)
        counters_in_experiment = set(str(m.metric) for m in callpath_models)
        actual_counter_list.intersection_update(counters_in_experiment)

        def check_and_remove(check_for, remove, also_reverse=False):
            if all(counter in actual_counter_list for counter in check_for):
                for counter in remove:
                    if counter not in check_for:
                        try:
                            actual_counter_list.remove(counter)
                        except KeyError:
                            pass
            if also_reverse:
                check_and_remove(remove, check_for, also_reverse=False)

        check_and_remove(['PAPI_BR_CN', 'PAPI_BR_UCN'], ['PAPI_BR_INS'], True)
        check_and_remove(['PAPI_LD_INS', 'PAPI_SR_INS'], ['PAPI_LST_INS'], True)
        all_fp_counters = ["PAPI_DP_OPS", "PAPI_SP_OPS", 'PAPI_FP_OPS', 'PAPI_FP_INS', 'PAPI_VEC_INS', "PAPI_VEC_SP",
                           "PAPI_VEC_DP", ]
        check_and_remove(["PAPI_DP_OPS", "PAPI_SP_OPS"], all_fp_counters)
        check_and_remove(["PAPI_VEC_DP", "PAPI_VEC_SP"], all_fp_counters)
        check_and_remove(['PAPI_FP_OPS'], all_fp_counters)
        check_and_remove(['PAPI_FP_INS'], all_fp_counters)
        check_and_remove(['PAPI_VEC_INS'], all_fp_counters)
        return actual_counter_list

    def check_model(self, model, coordinate, threshold):
        model_function = model.hypothesis.function
        value = model_function.evaluate(coordinate)

        if model_function.constant_coefficient == 0:
            return True
        deviation = abs(value / model_function.constant_coefficient)

        return deviation >= threshold

    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:
        callpath_models = {}

        gamma = 1.05

        for measurement_sequence in progress_bar(measurements):
            callpath = measurement_sequence[0].callpath
            metric = measurement_sequence[0].metric
            coordinates = tuple(m.coordinate[0] for m in measurement_sequence)
            key = callpath, coordinates

            if key not in callpath_models:
                callpath_models[key] = []

            model = self.single_param_modeler.create_model(measurement_sequence)
            model.callpath = callpath
            model.metric = metric
            model.measurements = measurement_sequence
            callpath_models[key].append(model)

        if self.weighted_terms:
            return self._model_weighted(callpath_models, measurements, progress_bar)

        models = []
        for measurement_sequence in progress_bar(measurements):
            callpath = measurement_sequence[0].callpath
            metric = measurement_sequence[0].metric
            coordinates = tuple(m.coordinate[0] for m in measurement_sequence)
            key = callpath, coordinates
            if str(metric) in self.metrics_to_improve:
                desired_counters = self.determine_relevant_counter_models(callpath_models[key])
                if not desired_counters:
                    for model in callpath_models[key]:
                        if model.metric is metric:
                            models.append(model)
                    warnings.warn(
                        'Missing counter data completely for at least one callpath, falling back to default modeler.')
                    continue
                if len(desired_counters) + 1 > len(coordinates):
                    required_points = len(desired_counters) + 1 - len(coordinates)
                    warnings.warn('The number of measurements is too low for the number of counters you selected. '
                                  f'You should add {required_points} additional measurement '
                                  f'point{"s" if required_points > 1 else ""} or select only {len(coordinates) - 1} '
                                  f'counter{"s" if (len(coordinates) - 1) > 1 else ""} for modeling.')

                runtime_function = SingleParameterFunction()
                for model in desired_counters:
                    for term in model.hypothesis.function.compound_terms:
                        runtime_function.add_compound_term(copy.copy(term))

                hypothesis = SingleParameterHypothesis(runtime_function, self.use_median)
                hypothesis.function.reset_coefficients()
                hypothesis.compute_coefficients(
                    measurement_sequence, no_negative_coefficients=self.no_negative_coefficients)
                previous = numpy.seterr(divide='ignore', invalid='ignore')

                # for all compound terms check if they are smaller than minimum allowed contribution
                # ignore the terms that contribute less than epsilon to the function
                small_terms = [i for i, term in enumerate(hypothesis.function.compound_terms)
                               if term.coefficient == 0 or hypothesis.calc_term_contribution(term,
                                                                                             measurement_sequence) < self.epsilon]
                for i in reversed(small_terms):
                    del hypothesis.function.compound_terms[i]
                numpy.seterr(**previous)
                hypothesis.compute_cost(measurement_sequence)

                runtime_model = Model(hypothesis, callpath, measurement_sequence[0].metric)
                runtime_model.measurements = measurement_sequence
                models.append(runtime_model)
            else:
                for model in callpath_models[key]:
                    if model.metric is metric:
                        models.append(model)

        return models

    def _model_weighted(self, callpath_models, measurements, progress_bar):
        models = []
        for measurement_sequence in progress_bar(measurements):
            callpath = measurement_sequence[0].callpath
            metric = measurement_sequence[0].metric
            coordinates = tuple(m.coordinate[0] for m in measurement_sequence)
            key = callpath, coordinates
            if str(metric) in self.metrics_to_improve:
                callpath_data: list = callpath_models[key]
                desired_counters = self.determine_relevant_counters(callpath_data)
                counter_weights = {
                    models.metric: numpy.array(
                        [measurement.mean for measurement in sorted(models.measurements, key=lambda m: m.coordinate)])
                    for models in callpath_data
                    if models.metric == self.total_count_metric or str(models.metric) in desired_counters
                }

                if self.total_count_metric not in counter_weights:
                    raise RecoverableError(f"Missing {self.total_count_metric}, which is necessary for weighting.")

                rest_weights = (counter_weights[self.total_count_metric] * 2 -
                                numpy.sum(list(counter_weights.values()), axis=0)) / counter_weights[
                                   self.total_count_metric]
                for metric in counter_weights:
                    counter_weights[metric] /= counter_weights[self.total_count_metric]

                counter_weights[self.total_count_metric] = rest_weights

                runtime_function = SingleParameterFunction()
                measurements_time = list(sorted(measurement_sequence, key=lambda m: m.coordinate))

                for model in callpath_models[key]:
                    if model.metric in counter_weights:
                        if isinstance(model.hypothesis, ConstantHypothesis):
                            weighted_measurements = []
                            for m, w in zip(measurements_time, counter_weights[model.metric]):
                                weighted_measurements.append(m.value(self.use_median) * w)
                            runtime_function.constant_coefficient += numpy.mean(weighted_measurements)
                        else:
                            hypothesis: SingleParameterHypothesis = copy.deepcopy(model.hypothesis)
                            hypothesis.function.reset_coefficients()
                            weighted_measurements = []
                            for m, w in zip(measurements_time, counter_weights[model.metric]):
                                weighted_measurements.append(m * w)
                            hypothesis.compute_coefficients(weighted_measurements)
                            runtime_function += hypothesis.function.compound_terms[0]
                            runtime_function.constant_coefficient += hypothesis.function.constant_coefficient

                hypothesis = SingleParameterHypothesis(runtime_function, self.use_median)
                previous = numpy.seterr(divide='ignore', invalid='ignore')

                # for all compound terms check if they are smaller than minimum allowed contribution
                # ignore the terms that contribute less than epsilon to the function
                small_terms = [i for i, term in enumerate(hypothesis.function.compound_terms)
                               if term.coefficient == 0 or hypothesis.calc_term_contribution(term,
                                                                                             measurement_sequence) < self.epsilon]
                for i in reversed(small_terms):
                    del hypothesis.function.compound_terms[i]
                numpy.seterr(**previous)
                hypothesis.compute_cost(measurement_sequence)

                runtime_model = Model(hypothesis, callpath, measurement_sequence[0].metric)
                runtime_model.measurements = measurement_sequence
                models.append(runtime_model)
            else:
                for model in callpath_models[key]:
                    if model.metric is metric:
                        models.append(model)
        return models
