# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import copy
from typing import Tuple, Dict, Union, TYPE_CHECKING

from extrap.entities.calculation_element import divide_no0
from extrap.entities.callpath import Callpath
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.hypotheses import Hypothesis
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.modelers.postprocessing import PostProcess, PostProcessedModel
from extrap.util.dynamic_options import DynamicOptions
from extrap.util.progress_bar import DUMMY_PROGRESS

if TYPE_CHECKING:
    from extrap.entities.experiment import Experiment


class ArithmeticIntensity(PostProcess):
    NAME = "Arithmetic Intensity"

    flops_metric = DynamicOptions.add(Metric('PAPI_DP_OPS'), Metric)
    unc_m_cas_count_metric = DynamicOptions.add(Metric("UNC_M_CAS_COUNT:ALL"), Metric)
    bytes_per_cas = DynamicOptions.add(8, int)

    arithmetic_intensity_metric = Metric("Arithmetic Intensity")

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

    def process(self, current_model_set: Dict[Tuple[Callpath, Metric], Model], progress_bar=DUMMY_PROGRESS) -> Dict[
        Tuple[Callpath, Metric], Union[Model, PostProcessedModel]]:
        model_set = self.generate_arithmetic_intensity_models(current_model_set, progress_bar)
        model_set.update(current_model_set)
        self.experiment.add_metric(self.arithmetic_intensity_metric)
        return model_set

    def generate_arithmetic_intensity_models(self, current_model_set, progress_bar):
        model_set = {}
        for (callpath, metric), v in progress_bar(current_model_set.items(), length=len(current_model_set)):
            if metric != self.flops_metric:
                continue
            cas_count_model = current_model_set.get((callpath, self.unc_m_cas_count_metric))
            if cas_count_model is None:
                continue

            flops_function = ComputationFunction(v.hypothesis.function)
            bytes_function = ComputationFunction(cas_count_model.hypothesis.function) * self.bytes_per_cas
            ai_function = divide_no0(flops_function, bytes_function)
            hypothesis = copy.copy(v.hypothesis)
            hypothesis.function = ai_function

            measurements_dict = {m.coordinate: m for m in v.measurements}
            for m in cas_count_model.measurements:
                measurements_dict[m.coordinate] = divide_no0(measurements_dict[m.coordinate], m)

            measurements = list(measurements_dict.values())
            hypothesis.compute_cost(measurements)
            if hasattr(hypothesis, "compute_adjusted_rsquared"):
                _, constant_cost = Hypothesis.calculate_constant_indicators(measurements, hypothesis.use_median)
                hypothesis.compute_adjusted_rsquared(constant_cost, measurements)

            model = Model(hypothesis, callpath, self.arithmetic_intensity_metric)
            model.measurements = measurements
            model_set[callpath, self.arithmetic_intensity_metric] = model
        return model_set

    def supports_processing(self, post_processing_history: list[PostProcess]) -> bool:
        return not post_processing_history

    @property
    def modifies_experiment(self) -> bool:
        return True
