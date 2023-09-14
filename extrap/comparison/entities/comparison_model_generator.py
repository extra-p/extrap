# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from typing import TYPE_CHECKING

from extrap.comparison.entities.comparison_model import ComparisonModel
from extrap.entities.callpath import Callpath
from extrap.entities.metric import Metric
from extrap.modelers.abstract_modeler import EMPTY_MODELER
from extrap.modelers.model_generator import ModelGenerator, ModelGeneratorSchema
from extrap.modelers.postprocessing import PostProcess
from extrap.modelers.postprocessing.aggregation import Aggregation
from extrap.util.exceptions import RecoverableError
from extrap.util.progress_bar import DUMMY_PROGRESS, ScaledSubProgressBar

if TYPE_CHECKING:
    from extrap.entities.experiment import Experiment


class ComparisonModelGenerator(ModelGenerator):
    models: dict[tuple[Callpath, Metric], ComparisonModel]

    def __init__(self, experiment: Experiment,
                 name: str = "New Modeler",
                 use_median: bool = False):
        super().__init__(experiment, NotImplemented, name, use_median)
        self._modeler = EMPTY_MODELER
        self.post_processing_history = []

    def post_process(self, post_process: PostProcess, progress_bar=DUMMY_PROGRESS,
                     auto_append=True) -> ComparisonModelGenerator:
        if not post_process.supports_processing(self.post_processing_history):
            raise RecoverableError(f"Processing this model with {post_process.NAME} is not supported.")

        post_process.experiment = self.experiment

        if hasattr(post_process, 'supports_comparison_models') and getattr(post_process, 'supports_comparison_models'):
            post_processed_comparison_set = ComparisonModelGenerator(post_process.experiment,
                                                                     post_process.NAME + ' ' + self.name)
            post_processed_comparison_set.models = post_process.process(self.models, progress_bar)
        else:
            number_models = len(next(iter(self.models.values())).models)
            progress_bar.total = number_models * len(self.models)
            post_processed_comparison_set = ComparisonModelGenerator(post_process.experiment,
                                                                     post_process.NAME + ' ' + self.name)

            all_models = []
            for c in range(number_models):
                with ScaledSubProgressBar(progress_bar, len(self.models)) as sub_progress_bar:
                    partial_models = {key: value.models[c] for key, value in self.models.items() if
                                      isinstance(value, ComparisonModel)}
                    all_models.append(post_process.process(partial_models, sub_progress_bar))

            post_processed_comparison_set.models = {key: ComparisonModel(*key, [m[key] for m in all_models]) for
                                                    key, value in progress_bar(self.models.items())
                                                    if isinstance(value, ComparisonModel)}

        post_processed_comparison_set.post_processing_history = self.post_processing_history + [post_process]
        if auto_append:
            self.experiment.add_modeler(post_processed_comparison_set)
        return post_processed_comparison_set

    def aggregate(self, aggregation: Aggregation, progress_bar=DUMMY_PROGRESS, auto_append=True):
        raise RecoverableError("Aggregation is not supported using a comparison model set.")

    def model_all(self, progress_bar=DUMMY_PROGRESS, auto_append=True):
        raise RecoverableError("Modelling is not supported using a comparison model set.")

    def restore_from_exp(self, experiment):
        self.experiment = experiment
        for key, model in self.models.items():
            if not model.measurements:
                model.measurements = experiment.measurements.get(key)
            if isinstance(model, ComparisonModel):
                for m in model.models:
                    if not m.measurements:
                        m.measurements = experiment.measurements.get((m.callpath, m.metric))


class ComparisonModelGeneratorSchema(ModelGeneratorSchema):
    def create_object(self):
        return ComparisonModelGenerator(None, None)
