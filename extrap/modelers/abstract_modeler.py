import copy
from abc import ABC, abstractmethod
from typing import Sequence

from marshmallow import fields

from extrap.entities.measurement import Measurement
from extrap.entities.model import Model
from extrap.util.progress_bar import DUMMY_PROGRESS
from extrap.util.serialization_schema import BaseSchema


class AbstractModeler(ABC):
    def __init__(self, use_median: bool):
        # use mean or median measurement values to calculate models
        self._use_median = use_median

    @property
    def use_median(self) -> bool:
        return self._use_median

    @use_median.setter
    def use_median(self, value: bool):
        self._use_median = value

    @abstractmethod
    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:
        """ This method is the core of the modeling system.
        It receives a sequence of measurement point sequences and returns a sequence of models.
        For each sequence of measurement points one model is generated.
        The measurement sequences are not guaranteed to have similar coordinates."""
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def NAME(cls):
        raise NotImplementedError


class SingularModeler(AbstractModeler, ABC):
    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:
        return [self.create_model(m) for m in progress_bar(measurements)]

    @abstractmethod
    def create_model(self, measurements: Sequence[Measurement]) -> Model:
        raise NotImplementedError


class MultiParameterModeler(AbstractModeler, ABC):

    def __init__(self, use_median, single_parameter_modeler):
        super().__init__(use_median)
        single_parameter_modeler.use_median = use_median
        self._default_single_parameter_modeler = single_parameter_modeler
        self.single_parameter_modeler: AbstractModeler = copy.copy(single_parameter_modeler)

    def reset_single_parameter_modeler(self):
        self.single_parameter_modeler = copy.copy(self._default_single_parameter_modeler)

    @AbstractModeler.use_median.setter
    def use_median(self, value):
        self._use_median = value
        self.single_parameter_modeler.use_median = value


class ModelerSchema(BaseSchema):
    use_median = fields.Bool()

    def create_object(self):
        raise NotImplementedError()