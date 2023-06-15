# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Sequence, Optional

from marshmallow import fields, post_load

from extrap.entities.measurement import Measurement
from extrap.entities.model import Model
from extrap.util.classproperty import classproperty
from extrap.util.progress_bar import DUMMY_PROGRESS
from extrap.util.serialization_schema import BaseSchema


class AbstractModeler(ABC):
    def __init__(self, use_median: bool):
        """Creates a new modeler object, that uses either the median or the mean when modeling.

           :param use_median: use mean or median measurement values to calculate models
        """
        self._use_median = use_median

    @property
    def use_median(self) -> bool:
        return self._use_median

    @use_median.setter
    def use_median(self, value: bool):
        self._use_median = value

    @abstractmethod
    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:
        """ Creates a model for each sequence of measurements.

        This method is the core of the modeling system.
        It receives a sequence of measurement point sequences and returns a sequence of models.
        For each sequence of measurement points one model is generated.
        The measurement sequences are not guaranteed to have similar coordinates."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def NAME(cls) -> str:  # noqa
        """ This attribute is the unique display name of the modeler.

        It is used for selecting the modeler in the GUI and CLI.
        You must override this only in concrete modelers, you should do so by setting the class variable NAME."""
        raise NotImplementedError

    @classproperty
    def DESCRIPTION(cls) -> Optional[str]:  # noqa
        """ This attribute is the description of the modeler.

        It is shown as additional information in the GUI and CLI.
        You should override this by setting the class variable DESCRIPTION."""
        return None


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

    def on_missing_sub_schema(self, type_, data, **kwargs):
        warnings.warn(f"Loaded unknown modeler of type {type_}")
        data['type_'] = type_
        return super(BaseSchema, self).load(data, **kwargs)

    def create_object(self):
        logging.debug(f"Created placeholder for unknown modeler")
        return _EmptyModeler(False)


class _EmptyModeler(AbstractModeler):
    NAME = "<Empty>"

    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:
        raise NotImplementedError()

    def __eq__(self, o: object) -> bool:
        return o is self or isinstance(o, _EmptyModeler)


EMPTY_MODELER = _EmptyModeler(False)


class _EmptyModelerSchema(ModelerSchema):
    use_median = fields.Constant(False, load_only=True, dump_only=True)

    @post_load
    def unpack_to_object(self, data, **kwargs):
        return EMPTY_MODELER

    def create_object(self):
        return EMPTY_MODELER
