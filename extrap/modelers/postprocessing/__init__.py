# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import logging
import typing
from abc import abstractmethod, ABC
from typing import Dict, Tuple, Optional, Union

from marshmallow import fields, post_load

from extrap.entities.callpath import Callpath
from extrap.entities.measurement import MeasurementSchema
from extrap.entities.metric import Metric
from extrap.entities.model import Model, ModelSchema
from extrap.util.classproperty import classproperty
from extrap.util.dynamic_options import DynamicOptions
from extrap.util.extension_loader import load_extensions
from extrap.util.progress_bar import DUMMY_PROGRESS
from extrap.util.serialization_schema import BaseSchema

if typing.TYPE_CHECKING:
    from extrap.entities.experiment import Experiment


class PostProcessedModel(Model):
    pass


class PostProcessedModelSchema(ModelSchema):
    measurements = fields.List(fields.Nested(MeasurementSchema))

    def create_object(self):
        return PostProcessedModel(None)

    def report_progress(self, data, **kwargs):
        return data


class PostProcess(DynamicOptions, ABC):

    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    @abstractmethod
    def process(self, current_model_set: Dict[Tuple[Callpath, Metric], Model],
                progress_bar=DUMMY_PROGRESS) -> Dict[Tuple[Callpath, Metric], Union[Model, PostProcessedModel]]:
        """Post-processes models provided by current_model_set.

        This method is the core of the post-processing system.
        It receives all models and returns post-processed versions of the models.
        It should not modify the experiment directly to add the models to the experiment.
        """
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def NAME(cls) -> str:  # noqa
        """ This attribute is the unique display name of the post process.

        It is used for selecting the post process in the GUI and CLI.
        You must override this only in concrete post processes, you should do so by setting the class variable NAME."""
        raise NotImplementedError

    @classproperty
    def DESCRIPTION(cls) -> Optional[str]:  # noqa
        """ This attribute is the description of the post process.

        It is shown as additional information in the GUI and CLI.
        You should override this by setting the class variable DESCRIPTION."""
        return None

    @abstractmethod
    def supports_processing(self, post_processing_history: list[PostProcess]) -> bool:
        """Returns true when post-processing is supported for a model set with the given post-processing history.
        Otherwise, returns false.
        """
        raise NotImplementedError

    @property
    def modifies_experiment(self) -> bool:
        return False


class PostProcessSchema(BaseSchema):
    def create_object(self):
        logging.debug(f"Created placeholder for unknown post process")
        return _EmptyPostProcess(None)


class _EmptyPostProcess(PostProcess):
    NAME = "<Empty>"

    def process(self, current_model_set: Dict[Tuple[Callpath, Metric], Model], progress_bar=DUMMY_PROGRESS) -> Dict[
        Tuple[Callpath, Metric], Union[Model, PostProcessedModel]]:
        raise NotImplementedError

    def supports_processing(self, post_processing_history: list[PostProcess]) -> bool:
        return False

    def __eq__(self, o: object) -> bool:
        return o is self or isinstance(o, _EmptyPostProcess)


EMPTY_POST_PROCESS = _EmptyPostProcess(False)


class _EmptyPostProcessSchema(PostProcessSchema):
    @post_load
    def unpack_to_object(self, data, **kwargs):
        return EMPTY_POST_PROCESS

    def create_object(self):
        return EMPTY_POST_PROCESS


all_post_processes = load_extensions(__path__, __name__, PostProcess)
