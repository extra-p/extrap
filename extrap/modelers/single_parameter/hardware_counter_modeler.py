# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
from typing import Sequence
import numpy

from extrap.entities.functions import SingleParameterFunction
from extrap.entities.hypotheses import SingleParameterHypothesis
from extrap.entities.measurement import Measurement
from extrap.entities.model import Model
from extrap.modelers.single_parameter.abstract_base import AbstractSingleParameterModeler
from extrap.modelers.single_parameter.basic import SingleParameterModeler
from extrap.util.progress_bar import DUMMY_PROGRESS
from extrap.modelers.modeler_options import modeler_options


@modeler_options
class HardwareCounterModeler(AbstractSingleParameterModeler):
    NAME = 'Hardware-Counter'
    DESCRIPTION = 'Creates a runtime model based on data of hardware performance counters'

    number_used_counters = modeler_options.add(3, int, range=range(1, 5))
    no_negative_coefficients = modeler_options.add(True, bool)

    counter_data = {}

    def __init__(self):
        """
        Initialize HardwareCounterModeler object.
        """
        super().__init__(use_median=False)

        self.min_measurement_points = 5
        self.counter_list = [
            "PAPI_LD_INS",
            "PAPI_SR_INS",
            "PAPI_BR_UCN",
            "PAPI_BR_CN",
            "PAPI_DP_OPS",
            "PAPI_SP_OPS",
        ]
        self.coordinates = {}
        self.global_counter_data = {}
        self.callpath_counter_data = {}

    def accumulate_counter_data(self, measurements: Sequence[Sequence[Measurement]]) -> dict:
        measurements_per_sequence = len(measurements[0])

        counter_data = {
            "time": [0] * measurements_per_sequence,
            "PAPI_TOT_INS": [0] * measurements_per_sequence,
            "PAPI_LST_INS": [0] * measurements_per_sequence,
            "PAPI_LD_INS": [0] * measurements_per_sequence,
            "PAPI_SR_INS": [0] * measurements_per_sequence,
            "PAPI_BR_INS": [0] * measurements_per_sequence,
            "PAPI_BR_UCN": [0] * measurements_per_sequence,
            "PAPI_BR_CN": [0] * measurements_per_sequence,
            "PAPI_DP_OPS": [0] * measurements_per_sequence,
            "PAPI_SP_OPS": [0] * measurements_per_sequence,
        }

        for measurement in measurements:
            current_metric = str(measurement[0].metric)
            if current_metric in counter_data:
                counter_data[current_metric] = list(
                    map(lambda x: x[0] + x[1].mean, zip(counter_data[current_metric], measurement)))

        return counter_data

    def determine_relevant_counters(self, callpath) -> list:
        callpath_data = self.callpath_counter_data[callpath]

        relevant_counters = []
        counter_weights = []
        for measurements in callpath_data:
            metric = measurements[0].metric
            if str(metric) in self.counter_list:
                val_sum = sum([measurement.mean for measurement in measurements])
                counter_weights.append((val_sum, metric))

        counter_weights.sort(
            key=lambda x: x[0],
            reverse=True
        )
        relevant_counters = [tpl[1] for tpl in counter_weights[:self.number_used_counters]]
        return relevant_counters

    def accumulate_callpath_data(self, measurements: Sequence[Sequence[Measurement]]) -> dict:
        callpath_data = {}
        for measurement_sequence in measurements:
            callpath = measurement_sequence[0].callpath
            if not callpath in callpath_data:
                callpath_data[callpath] = []
            callpath_data[callpath].append(measurement_sequence)
        return callpath_data

    def check_model(self, model, coordinate, threshold):
        model_function = model.hypothesis.function
        value = model_function.evaluate(coordinate)

        if model_function.constant_coefficient == 0:
            return True
        deviation = abs(value / model_function.constant_coefficient)

        return deviation >= threshold

    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:
        self.coordinates = [measurement.coordinate for measurement in measurements[0]]
        # self.global_counter_data = self.accumulate_counter_data(measurements)
        # self.counter_weights = self.determine_counter_weights(measurements)
        self.callpath_counter_data = self.accumulate_callpath_data(measurements)
        counter_weight_distribution = {}
        callpath_models = {}
        single_param_modeler = SingleParameterModeler()
        models = []
        epsilon = 1.05

        for measurement_sequence in progress_bar(measurements):
            callpath = measurement_sequence[0].callpath
            metric = measurement_sequence[0].metric
            if callpath not in callpath_models:
                callpath_models[callpath] = []
            model = single_param_modeler.create_model(measurement_sequence)
            model.metric = metric
            model.callpath = callpath
            callpath_models[callpath].append(model)

        for measurement_sequence in progress_bar(measurements):
            callpath = measurement_sequence[0].callpath
            metric = measurement_sequence[0].metric
            if str(metric) == "time":
                desired_metrics = self.determine_relevant_counters(callpath)
                runtime_function = SingleParameterFunction()
                for model in callpath_models[callpath]:
                    if model.metric in desired_metrics and self.check_model(model, self.coordinates[-1], epsilon):
                        for term in model.hypothesis.function.compound_terms:
                            runtime_function.add_compound_term(copy.copy(term))
                hypothesis = SingleParameterHypothesis(runtime_function, self.use_median)
                hypothesis.function.reset_coefficients()
                hypothesis.compute_coefficients(
                    measurement_sequence, no_negative_coefficients=self.no_negative_coefficients)
                previous = numpy.seterr(divide='ignore', invalid='ignore')
                # for all compound terms check if they are smaller than minimum allowed contribution

                # ignore this hypothesis, since one of the terms contributes less than epsilon to the function
                small_terms = [i for i, term in enumerate(hypothesis.function.compound_terms)
                               if term.coefficient == 0 or hypothesis.calc_term_contribution(term, measurement_sequence) < self.epsilon]
                for i in reversed(small_terms):
                    del hypothesis.function.compound_terms[i]
                numpy.seterr(**previous)
                hypothesis.compute_cost(measurement_sequence)

                runtime_model = Model(hypothesis, callpath, measurement_sequence[0].metric)
                runtime_model.measurements = measurement_sequence
                models.append(runtime_model)
            else:
                for model in callpath_models[callpath]:
                    if model.metric is metric:
                        models.append(model)

        return models
