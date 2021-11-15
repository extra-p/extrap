# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from typing import TYPE_CHECKING

from extrap.comparison.entities.comparison_model import ComparisonModel
from extrap.modelers.abstract_modeler import EMPTY_MODELER
from extrap.modelers.aggregation import Aggregation
from extrap.modelers.model_generator import ModelGenerator, ModelGeneratorSchema
from extrap.util.exceptions import RecoverableError
from extrap.util.progress_bar import DUMMY_PROGRESS

if TYPE_CHECKING:
    from extrap.entities.experiment import Experiment


class ComparisonModelGenerator(ModelGenerator):
    def __init__(self, experiment: Experiment,
                 name: str = "New Modeler",
                 use_median: bool = False):
        super().__init__(experiment, NotImplemented, name, use_median)
        self._modeler = EMPTY_MODELER

    def aggregate(self, aggregation: Aggregation, progress_bar=DUMMY_PROGRESS):
        raise RecoverableError("Aggregation is not supported using a comparison model set.")

    def model_all(self, progress_bar=DUMMY_PROGRESS):
        raise RecoverableError("Modelling is not supported using a comparison model set.")

    def restore_from_exp(self, experiment):
        self.experiment = experiment
        for key, model in self.models.items():
            if not model.measurements:
                model.measurements = experiment.measurements.get(key)
            if isinstance(model, ComparisonModel):
                for m in model.models:
                    if not model.measurements:
                        model.measurements = experiment.measurements.get((m.callpath, m.metric))


class ComparisonModelGeneratorSchema(ModelGeneratorSchema):
    def create_object(self):
        return ComparisonModelGenerator(None, None)
