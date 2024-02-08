from __future__ import annotations

import copy
from typing import Dict, Tuple, Union

from extrap.entities.callpath import Callpath
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.hypotheses import Hypothesis
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.modelers.postprocessing import PostProcess, PostProcessedModel, PostProcessSchema
from extrap.util.dynamic_options import DynamicOptions
from extrap.util.progress_bar import DUMMY_PROGRESS


class EstimateTimeRelatedMetric(PostProcess):
    metric_name = DynamicOptions.add('Energy', str)
    conversion_factor = DynamicOptions.add(1.0, float)

    def process(self, current_model_set: Dict[Tuple[Callpath, Metric], Model], progress_bar=DUMMY_PROGRESS) -> Dict[
        Tuple[Callpath, Metric], Union[Model, PostProcessedModel]]:
        model_set = {}
        data_set = {}
        metric_estimated = Metric(self.metric_name, postprocess__generated=True)

        for (callpath, metric), v in progress_bar(current_model_set.items(), length=len(current_model_set)):
            model_set[callpath, metric] = v
            if metric.name == 'time':
                if callpath not in data_set:
                    function = ComputationFunction(v.hypothesis.function)
                    hypothesis = copy.copy(v.hypothesis)
                    hypothesis.function = function

                    data_set[callpath] = (
                        hypothesis, {m.coordinate: m * self.conversion_factor for m in v.measurements})

        for callpath, (hypothesis, measurements_dict) in data_set.items():
            measurements = list(measurements_dict.values())
            hypothesis.compute_cost(measurements)
            if hasattr(hypothesis, "compute_adjusted_rsquared"):
                _, constant_cost = Hypothesis.calculate_constant_indicators(measurements, hypothesis.use_median)
                hypothesis.compute_adjusted_rsquared(constant_cost, measurements)

            model = Model(hypothesis, callpath, metric_estimated)
            model.measurements = measurements
            model_set[callpath, metric_estimated] = model

        self.experiment.add_metric(metric_estimated)
        return model_set

    NAME = "Estimate time related metric"

    def supports_processing(self, post_processing_history: list[PostProcess]) -> bool:
        return any(metric.name == 'time' for metric in self.experiment.metrics)

    @property
    def modifies_experiment(self) -> bool:
        return True


class EstimateTimeRelatedMetricSchema(PostProcessSchema):
    def create_object(self):
        return EstimateTimeRelatedMetric(None)
