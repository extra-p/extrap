from __future__ import annotations

from typing import TYPE_CHECKING

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


class ComparisonModelGeneratorSchema(ModelGeneratorSchema):
    def create_object(self):
        return ComparisonModelGenerator(None, None)
