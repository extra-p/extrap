from __future__ import annotations

import copy
import json
from typing import Dict, Tuple, Union

from extrap.entities.callpath import Callpath
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.hypotheses import Hypothesis
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.entities.parameter import Parameter
from extrap.modelers.postprocessing import PostProcess, PostProcessedModel, PostProcessSchema
from extrap.util import measurement_set
from extrap.util.dynamic_options import DynamicOptions
from extrap.util.exceptions import RecoverableError
from extrap.util.progress_bar import DUMMY_PROGRESS


class ConcretizationData:
    def __init__(self, data_json: str = None):
        if data_json is not None:
            self.data = json.loads(data_json)
        else:
            self.data = {}

    @property
    def cu_count(self) -> int:
        return self.data['cu_count']

    def __getitem__(self, item):
        return [self.data[key][item] for key in
                ['load', 'store', 'double_precision', 'single_precision', 'branch', 'other']]

    def __str__(self):
        return json.dumps(self.data)


class ModelConcretization(PostProcess):
    # data = [0.02943982122405140, 0.42147964499416400, 0.01831283442512410, 0.00462516829449892, 0.00354319719267549,
    #         0.00807172719139877]
    # data = [0.03148720464707880, 0.61428993734171700, 0.02127634561947720, 0.00510115567866146, 0.00522538230621277,
    #         0.01048028883814250]
    data = DynamicOptions.add(ConcretizationData(), ConcretizationData)
    approximation = DynamicOptions.add('mean', str, range={m: m for m in ['mean', 'min', 'max']})
    rank_parameter = DynamicOptions.add(Parameter("p"), Parameter)

    def process(self, current_model_set: Dict[Tuple[Callpath, Metric], Model], progress_bar=DUMMY_PROGRESS) -> Dict[
        Tuple[Callpath, Metric], Union[Model, PostProcessedModel]]:
        reference_metric = Metric('time')
        model_set = {}
        progress_bar.total += 1
        metrics = set(met for cp, met in current_model_set.keys())
        if Metric('PAPI_DP_OPS') not in metrics and Metric('PAPI_SP_OPS') not in metrics:
            raise RecoverableError()
        if Metric('PAPI_VEC_DP') not in metrics and Metric('PAPI_VEC_SP') not in metrics:
            raise RecoverableError()
        if Metric('PAPI_LD_INS') not in metrics or Metric('PAPI_SR_INS') not in metrics:
            raise RecoverableError()
        if Metric('PAPI_BR_INS') not in metrics:
            raise RecoverableError()
        if Metric('PAPI_TOT_INS') not in metrics:
            raise RecoverableError()

        rank_param_idx = self.experiment.parameters.index(self.rank_parameter)

        progress_bar.update()
        for (cp, met), v in progress_bar(current_model_set.items()):
            if met != reference_metric:
                continue

            thread_count = max(m.count.count / (m.coordinate[rank_param_idx] * m.count.repetitions) for m in
                               current_model_set[cp, Metric('PAPI_TOT_INS')].measurements)

            known_instructions = measurement_set.sum(*[current_model_set[cp, m].measurements for m in
                                                       [Metric('PAPI_VEC_DP'), Metric('PAPI_VEC_SP'),
                                                        Metric('PAPI_LD_INS'),
                                                        Metric('PAPI_SR_INS'), Metric('PAPI_BR_INS')]
                                                       if (cp, m) in current_model_set])
            other_instructions = measurement_set.subtract(current_model_set[cp, Metric('PAPI_TOT_INS')].measurements,
                                                          known_instructions)

            times = [measurement_set.multiply_factor(f, current_model_set[cp, m].measurements) for f, m in
                     zip(self.data[self.approximation],
                         [Metric('PAPI_LD_INS'), Metric('PAPI_SR_INS'), Metric('PAPI_DP_OPS'), Metric('PAPI_SP_OPS'),
                          Metric('PAPI_BR_INS')]) if (cp, m) in current_model_set]
            times.append(measurement_set.multiply_factor(self.data[self.approximation][-1], other_instructions))

            time = measurement_set.multiply_factor(10 ** -9 * thread_count / self.data.cu_count,
                                                   measurement_set.sum(*times))

            known_instructions_f = sum(ComputationFunction(current_model_set[cp, m].hypothesis.function) for m in
                                       [Metric('PAPI_VEC_DP'), Metric('PAPI_VEC_SP'), Metric('PAPI_LD_INS'),
                                        Metric('PAPI_SR_INS'), Metric('PAPI_BR_INS')]
                                       if (cp, m) in current_model_set)
            other_instructions_f = (
                    ComputationFunction(current_model_set[cp, Metric('PAPI_TOT_INS')].hypothesis.function)
                    - known_instructions_f)

            times_f = [f * ComputationFunction(current_model_set[cp, m].hypothesis.function) for f, m in
                       zip(self.data[self.approximation],
                           [Metric('PAPI_LD_INS'), Metric('PAPI_SR_INS'), Metric('PAPI_DP_OPS'), Metric('PAPI_SP_OPS'),
                            Metric('PAPI_BR_INS')]) if (cp, m) in current_model_set]
            times_f.append(other_instructions_f * self.data[self.approximation][-1])

            time_f = sum(times_f) / 10 ** 9 * thread_count / self.data.cu_count

            hypothesis = copy.copy(current_model_set[cp, reference_metric].hypothesis)
            hypothesis.function = time_f
            hypothesis.compute_cost(time)
            if hasattr(hypothesis, "compute_adjusted_rsquared"):
                _, constant_cost = Hypothesis.calculate_constant_indicators(time, hypothesis.use_median)
                hypothesis.compute_adjusted_rsquared(constant_cost, time)
            model = PostProcessedModel(hypothesis, cp, reference_metric)
            model.measurements = time
            model_set[cp, reference_metric] = model
        return model_set

    NAME = "Model Concretization"

    def supports_processing(self, post_processing_history: list[PostProcess]) -> bool:
        return not bool(post_processing_history)


class ModelConcretizationSchema(PostProcessSchema):
    def create_object(self) -> ModelConcretization:
        return ModelConcretization(None)
