# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import typing
from abc import abstractmethod
from typing import Sequence, Dict, Tuple, Optional, Union

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.modelers.postprocessing import PostProcessedModel, PostProcessedModelSchema, PostProcess, PostProcessSchema
from extrap.util.classproperty import classproperty
from extrap.util.extension_loader import load_extensions
from extrap.util.progress_bar import DUMMY_PROGRESS

if typing.TYPE_CHECKING:
    pass


class AggregatedModel(PostProcessedModel):
    pass


class AggregatedModelSchema(PostProcessedModelSchema):
    def create_object(self):
        return AggregatedModel(None)


class Aggregation(PostProcess):
    TAG_DISABLED = 'agg__disabled'
    TAG_USAGE_DISABLED = 'agg__usage_disabled'
    TAG_USAGE_DISABLED_agg_model = 'only_agg_model'
    TAG_CATEGORY = 'agg__category'

    def __init__(self, experiment):
        super().__init__(experiment)
        self.tag_suffix: Optional[str] = None
        """
        A suffix appended to the tags controlling the aggregation.

        Allows specifying aggregation behavior for specific use cases, such as comparison.
        A separator will be automatically added between the tag and the suffix.
        """

    @abstractmethod
    def aggregate(self, models: Dict[Tuple[Callpath, Metric], Model], calltree: CallTree, metrics: Sequence[Metric],
                  progress_bar=DUMMY_PROGRESS) -> Dict[Tuple[Callpath, Metric], Union[Model, AggregatedModel]]:
        """ Creates an aggregated model for each model.

        This method is the core of the aggregation system.
        It receives a call-tree and all models and returns aggregated versions of the models.
        It should honor the "agg__disabled" tag on metrics and callpaths,
        thus it should not create aggregations for elements marked with this tag.
        """
        raise NotImplementedError

    def process(self, current_model_set: Dict[Tuple[Callpath, Metric], Model], progress_bar=DUMMY_PROGRESS) -> Dict[
        Tuple[Callpath, Metric], Union[Model, PostProcessedModel]]:
        return self.aggregate(current_model_set, self.experiment.call_tree, self.experiment.metrics, progress_bar)

    def supports_processing(self, post_processing_history: list[PostProcess]) -> bool:
        return not post_processing_history

    @classproperty
    @abstractmethod
    def NAME(cls) -> str:  # noqa
        """ This attribute is the unique display name of the aggregation.

        It is used for selecting the aggregation in the GUI and CLI.
        You must override this only in concrete aggregations, you should do so by setting the class variable NAME."""
        raise NotImplementedError

    @classproperty
    def DESCRIPTION(cls) -> Optional[str]:  # noqa
        """ This attribute is the description of the aggregation.

        It is shown as additional information in the GUI and CLI.
        You should override this by setting the class variable DESCRIPTION."""
        return None


class AggregationSchema(PostProcessSchema):
    pass


all_aggregations = load_extensions(__path__, __name__, Aggregation)
