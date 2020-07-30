from abc import ABC, abstractmethod
from typing import Sequence

from entities.measurement import Measurement
from entities.model import Model
from util.progress_bar import DUMMY_PROGRESS


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
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def NAME(cls):
        raise NotImplementedError


class LegacyModeler(AbstractModeler, ABC):
    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:
        return [self.create_model(m) for m in progress_bar(measurements)]

    @abstractmethod
    def create_model(self, measurements: Sequence[Measurement]):
        raise NotImplementedError


class MultiParameterModeler(AbstractModeler, ABC):

    def __init__(self, use_median, single_parameter_modeler):
        super().__init__(use_median)
        single_parameter_modeler.use_median = use_median
        self.single_parameter_modeler = single_parameter_modeler

    @AbstractModeler.use_median.setter
    def use_median(self, value):
        self._use_median = value
        self.single_parameter_modeler.use_median = value
