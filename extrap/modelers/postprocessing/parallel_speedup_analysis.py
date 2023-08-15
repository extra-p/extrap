# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import copy
from typing import Dict, Tuple, Union

from extrap.entities.callpath import Callpath
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.entities.parameter import Parameter
from extrap.modelers.postprocessing import PostProcess, PostProcessedModel
from extrap.util.dynamic_options import DynamicOptions
from extrap.util.exceptions import RecoverableError
from extrap.util.progress_bar import DUMMY_PROGRESS


class ParallelSpeedupAnalysis(PostProcess):
    NAME = "Parallel Speedup"

    resource_parameter = DynamicOptions.add(Parameter('gflops'), Parameter)

    def process(self, current_model_set: Dict[Tuple[Callpath, Metric], Model], progress_bar=DUMMY_PROGRESS) -> Dict[
        Tuple[Callpath, Metric], Union[Model, PostProcessedModel]]:
        try:
            param_idx = self.experiment.parameters.index(self.resource_parameter)
        except ValueError:
            raise RecoverableError(f"Parameter {self.resource_parameter.name} could not be found.")

        model_set = {}
        for k, v in progress_bar(current_model_set.items(), length=len(current_model_set)):
            function = ComputationFunction(v.hypothesis.function)
            res_param = ComputationFunction.get_param(param_idx)
            serial_function = function.sympy_function.subs(res_param, 1)
            function = serial_function / function
            hypothesis = copy.copy(v.hypothesis)
            hypothesis.function = function

            measurements = [
                float(serial_function.subs(
                    [(ComputationFunction.get_param(i), c) for i, c in enumerate(m.coordinate)])) / m /
                m.coordinate[param_idx] for m in v.measurements]
            hypothesis.compute_cost(measurements)

            model = Model(hypothesis, *k)
            model.measurements = measurements
            model_set[k] = model

        return model_set

    def supports_processing(self, post_processing_history: list[PostProcess]) -> bool:
        return not any(isinstance(p, ParallelSpeedupAnalysis) for p in post_processing_history)
