# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from itertools import accumulate
from subprocess import call
import warnings
import numpy as np
import copy

from typing import Sequence, Optional, Tuple
from extrap.entities import measurement

from extrap.entities.measurement import Measurement
from extrap.entities.model import Model
from extrap.entities.functions import Function, SingleParameterFunction
from extrap.entities.metric import Metric
from extrap.entities.hypotheses import Hypothesis, SingleParameterHypothesis
from extrap.entities.callpath import Callpath
from extrap.modelers.abstract_modeler import AbstractModeler
from extrap.modelers.single_parameter.basic import SingleParameterModeler
from extrap.modelers.single_parameter.abstract_base import AbstractSingleParameterModeler

class HardwareCounterModeler(AbstractSingleParameterModeler):
    NAME = 'Hardware-Counter'
    DESCRIPTION = 'Creates a runtime model based on data of hardware performance counters'
    
    counter_data = {}

    def __init__(self):
        """
        Initialize HardwareCounterModeler object.
        """
        super().__init__(use_median=False)

        self.min_measurement_points = 5
        self.counter_list = [
            "PAPI_TOT_INS",
            "PAPI_LD_INS",
            "PAPI_SR_INS",
            #"PAPI_BR_INS",
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

        for callpath in measurements:
            current_metric = str(callpath[0].metric)
            if current_metric in counter_data:
                counter_data[current_metric] = list(map(lambda x: x[0] + x[1].mean, zip(counter_data[current_metric], callpath)))
                
        return counter_data

    def determine_relevant_counters(self, callpath) -> list:
        callpath_data = self.callpath_counter_data[callpath]
        tot_ins = next(filter(
            lambda x: str(x[0].metric) == "PAPI_TOT_INS",
            callpath_data
        ))
        tot_ins_sum = sum([measurement.mean for measurement in tot_ins])
    
        relevant_counters = []
        max_counter_amount = 3
        counter_weights = []
        for measurements in callpath_data:
            metric = measurements[0].metric
            if str(metric) in self.counter_list and str(metric) != "PAPI_TOT_INS":
                val_sum = sum([measurement.mean for measurement in measurements])
                weight = val_sum / tot_ins_sum
                counter_weights.append((weight, metric))
                
        counter_weights.sort(
            key = lambda x: x[0],
            reverse = True
        )
        relevant_counters = [tpl[1] for tpl in counter_weights[:max_counter_amount]]
        return relevant_counters

    def accumulate_callpath_data(self, measurements: Sequence[Sequence[Measurement]]) -> dict:
        callpath_data = {}
        for measurement_sequence in measurements:
            callpath = measurement_sequence[0].callpath
            if not callpath in callpath_data:
                callpath_data[callpath] = []
            callpath_data[callpath].append(measurement_sequence)
        return callpath_data

    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar:...) -> Sequence[Model]:
        self.coordinates = [measurement.coordinate for measurement in measurements[0]]
        #self.global_counter_data = self.accumulate_counter_data(measurements)
        #self.counter_weights = self.determine_counter_weights(measurements)
        self.callpath_counter_data = self.accumulate_callpath_data(measurements)
        counter_weight_distribution = {}
        callpath_models = {}
        single_param_modeler = SingleParameterModeler()
        models = []

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
                    if model.metric in desired_metrics:
                        for term in model.hypothesis.function.compound_terms:
                            runtime_function.add_compound_term(copy.copy(term))
                hypothesis = SingleParameterHypothesis(runtime_function, self.use_median)
                hypothesis.function.reset_coefficients()
                hypothesis.compute_coefficients(measurement_sequence)
                hypothesis.compute_cost(measurement_sequence)
                # Remove terms with small coefficients
                for term in hypothesis.function.compound_terms:
                    if term.coefficient < 1e-5:
                        hypothesis.function.compound_terms.remove(term)
                hypothesis.compute_coefficients(measurement_sequence)
                hypothesis.compute_cost(measurement_sequence)
                runtime_model = Model(hypothesis, callpath, measurement_sequence[0].metric)
                runtime_model.measurements = measurement_sequence
                models.append(runtime_model)
            else:
                for model in callpath_models[callpath]:
                    if model.metric is metric:
                        models.append(model)

        return models
            


        
