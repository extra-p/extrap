# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from abc import abstractmethod, ABC
from typing import Sequence, Dict, Tuple, Optional, Union

from marshmallow import fields, post_dump, pre_load

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree
from extrap.entities.measurement import MeasurementSchema
from extrap.entities.metric import Metric
from extrap.entities.model import Model, ModelSchema
from extrap.entities.named_entity import TAG_SEPARATOR
from extrap.util.classproperty import classproperty
from extrap.util.extension_loader import load_extensions
from extrap.util.progress_bar import DUMMY_PROGRESS


class AggregatedModel(Model):
    pass


class AggregatedModelSchema(ModelSchema):
    measurements = fields.List(fields.Nested(MeasurementSchema))

    def create_object(self):
        return AggregatedModel(None)

    def report_progress(self, data, **kwargs):
        return data

    @post_dump
    def intercept(self, data, many, **kwargs):
        return data

    @pre_load
    def intercept2(self, data, many, **kwargs):
        return data


class Aggregation(ABC):
    TAG_DISABLED = 'agg__disabled'
    TAG_USAGE_DISABLED = 'agg__usage_disabled'
    TAG_USAGE_DISABLED_agg_model = 'only_agg_model'
    TAG_CATEGORY = 'agg__category'

    def __init__(self):
        self._tag_suffix = None

    @property
    def tag_suffix(self) -> str:
        """
        A suffix appended to the tags controlling the aggregation.

        Allows specifying aggregation behavior for specific use cases, such as comparison.
        A separator will be automatically added between the tag and the suffix.
        """
        return self._tag_suffix

    @tag_suffix.setter
    def tag_suffix(self, val: str):
        self._tag_suffix = val
        if val:
            self._update_tags(suffix=TAG_SEPARATOR + self._tag_suffix)
        else:
            self._update_tags()

    def _update_tags(self, prefix="", suffix=""):
        """
        Updates the tags with prefix and suffix.

        Should be overridden/extended when new tags are added.
        """
        self.TAG_DISABLED = prefix + type(self).TAG_DISABLED + suffix
        self.TAG_USAGE_DISABLED = prefix + type(self).TAG_USAGE_DISABLED + suffix
        self.TAG_CATEGORY = prefix + type(self).TAG_CATEGORY + suffix

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


all_aggregations = load_extensions(__path__, __name__, Aggregation)
