from __future__ import annotations

import copy
from typing import Dict, Tuple, Union

from extrap.entities.callpath import Callpath
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.hypotheses import Hypothesis
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.modelers.postprocessing import PostProcess, PostProcessedModel, PostProcessSchema
from extrap.util.progress_bar import DUMMY_PROGRESS


class CalculateTotalEnergy(PostProcess):
    def process(self, current_model_set: Dict[Tuple[Callpath, Metric], Model], progress_bar=DUMMY_PROGRESS) -> Dict[
        Tuple[Callpath, Metric], Union[Model, PostProcessedModel]]:
        model_set = {}
        data_set = {}
        metric_energy = Metric("Energy", postprocess_generated=True)

        for (callpath, metric), v in progress_bar(current_model_set.items(), length=len(current_model_set)):
            model_set[callpath, metric] = v
            if metric.name.startswith('energy'):
                if callpath not in data_set:
                    function = ComputationFunction(v.hypothesis.function)
                    hypothesis = copy.copy(v.hypothesis)
                    hypothesis.function = function

                    data_set[callpath] = (hypothesis, {m.coordinate: m for m in v.measurements})
                else:
                    hypothesis, measurements_dict = data_set[callpath]
                    hypothesis.function += v.hypothesis.function
                    for m in v.measurements:
                        measurements_dict[m.coordinate] += m

        for callpath, (hypothesis, measurements_dict) in data_set.items():
            measurements = list(measurements_dict.values())
            hypothesis.compute_cost(measurements)
            if hasattr(hypothesis, "compute_adjusted_rsquared"):
                _, constant_cost = Hypothesis.calculate_constant_indicators(measurements, hypothesis.use_median)
                hypothesis.compute_adjusted_rsquared(constant_cost, measurements)

            model = Model(hypothesis, callpath, metric_energy)
            model.measurements = measurements
            model_set[callpath, metric_energy] = model

        self.experiment.add_metric(metric_energy)
        return model_set

    NAME = "Calculate Total Energy"

    def supports_processing(self, post_processing_history: list[PostProcess]) -> bool:
        return any(metric.name.startswith('energy') for metric in self.experiment.metrics)

    @property
    def modifies_experiment(self) -> bool:
        return True


class CalculateTotalEnergySchema(PostProcessSchema):
    def create_object(self):
        return CalculateTotalEnergy(None)
