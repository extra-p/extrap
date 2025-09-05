# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2025, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import itertools
from typing import Dict, Union, Tuple, TYPE_CHECKING

from marshmallow import fields, post_dump, pre_load

from extrap.entities.callpath import Callpath, CallpathSchema
from extrap.entities.measurement import Measure
from extrap.entities.metric import Metric, MetricSchema
from extrap.entities.model import Model, ModelSchema
from extrap.modelers import multi_parameter
from extrap.modelers import single_parameter
from extrap.modelers.abstract_modeler import AbstractModeler, MultiParameterModeler, ModelerSchema
from extrap.modelers.modeler_options import modeler_options
from extrap.modelers.postprocessing import PostProcess, PostProcessSchema
from extrap.modelers.postprocessing.aggregation import Aggregation
from extrap.util.exceptions import RecoverableError
from extrap.util import deprecation
from extrap.util.progress_bar import DUMMY_PROGRESS
from extrap.util.serialization_schema import TupleKeyDict, BaseSchema

if TYPE_CHECKING:
    from extrap.entities.experiment import Experiment


class ModelGenerator:
    """
    Counter for global modeler ids
    """
    ID_COUNTER = itertools.count()

    def __init__(self, experiment: Experiment,
                 modeler: Union[AbstractModeler, str] = "Default",
                 name: str = "New Modeler", use_measure: Measure = Measure.MEAN, *, use_median: bool = None):
        self.experiment = experiment
        self.name = name
        self.id = next(ModelGenerator.ID_COUNTER)

        if isinstance(use_measure, bool) or use_median is not None:
            if use_median is not None:
                use_measure = use_median
            deprecation.deprecated.code("use_median is deprecated, use use_measure instead.")
            use_measure = Measure.from_use_median(use_measure)

        # choose the modeler based on the input data
        self._modeler: AbstractModeler = self._choose_modeler(modeler, use_measure)
        # all models, modeled with this model generator
        self.models: Dict[Tuple[Callpath, Metric], Model] = {}

    @property
    def modeler(self):
        return self._modeler

    def _choose_modeler(self, modeler: Union[AbstractModeler, str], use_measure) -> AbstractModeler:
        if isinstance(modeler, str):

            if len(self.experiment.parameters) == 1:
                # single-parameter model generator init here...
                try:
                    result_modeler = single_parameter.all_modelers[modeler]()
                except KeyError:
                    raise ValueError(
                        f'A modeler with name "{modeler}" does not exist. For single-parameter experiments only the '
                        f'following modelers are available "{", ".join(single_parameter.all_modelers.keys())}".') from None
            else:
                # multi-parameter model generator init here...
                try:
                    result_modeler = multi_parameter.all_modelers[modeler]()
                except KeyError as e:
                    if modeler in single_parameter.all_modelers:
                        raise ValueError(
                            f'A multi-parameter modeler with name "{modeler}" does not exist. '
                            f'You selected a modeler only available for single-parameter experiments. '
                            f'For multi-parameter experiments only the '
                            f'following modelers are available "{", ".join(multi_parameter.all_modelers.keys())}". '
                            f'You can change the underlying single-parameter modeler by setting the '
                            f'single-parameter-modeler option of the multi-parameter modeler.') from e
                    raise ValueError(
                        f'A modeler with name "{modeler}" does not exist. For multi-parameter experiments only the '
                        f'following modelers are available "{", ".join(multi_parameter.all_modelers.keys())}".') from e
                result_modeler.use_measure = use_measure

        elif modeler is NotImplemented:
            result_modeler = NotImplemented
        else:
            if (len(self.experiment.parameters) > 1) == isinstance(modeler, MultiParameterModeler):
                # single-parameter model generator init here...
                result_modeler = modeler
                if use_measure is not None:
                    result_modeler.use_measure = use_measure
            elif len(self.experiment.parameters) > 1:
                raise ValueError("Modeler must use multiple parameters.")
            else:
                raise ValueError("Modeler must use one parameter.")
        return result_modeler

    def model_all(self, progress_bar=DUMMY_PROGRESS, auto_append=True):
        models = self._modeler.model(list(self.experiment.measurements.values()), progress_bar)
        self.models = {
            k: m for k, m in zip(self.experiment.measurements.keys(), models)
        }
        for (callpath, metric), model in self.models.items():
            model.callpath = callpath
            model.metric = metric
            model.measurements = self.experiment.measurements[(callpath, metric)]
        if auto_append:
            # add the modeler with the results to the experiment
            self.experiment.add_modeler(self)

    def aggregate(self, aggregation: Aggregation, progress_bar=DUMMY_PROGRESS, auto_append=True):
        mg = AggregateModelGenerator(self.experiment, aggregation, self._modeler, aggregation.NAME + ' ' + self.name,
                                     self._modeler.use_measure)
        aggregation.experiment = self.experiment
        mg.models = aggregation.aggregate(self.models, self.experiment.call_tree, self.experiment.metrics, progress_bar)
        if auto_append:
            self.experiment.add_modeler(mg)

    def post_process(self, post_process: PostProcess, progress_bar=DUMMY_PROGRESS,
                     auto_append=True) -> PostProcessedModelSet:
        mg = PostProcessedModelSet(self.experiment, post_process, self._modeler,
                                   post_process.NAME + ' ' + self.name, self._modeler.use_measure)
        post_process.experiment = self.experiment
        mg.models = post_process.process(self.models, progress_bar)
        if auto_append:
            self.experiment.add_modeler(mg)
        return mg

    def __eq__(self, other):
        if not isinstance(other, ModelGenerator):
            return NotImplemented
        elif self is other:
            return True
        else:
            return (self.models == other.models and
                    self._modeler.NAME == other._modeler.NAME and
                    self._modeler.use_measure == other._modeler.use_measure and
                    modeler_options.equal(self._modeler, other._modeler))

    def restore_from_exp(self, experiment):
        self.experiment = experiment
        for key, model in self.models.items():
            model.measurements = experiment.measurements.get(key)


class PostProcessedModelSet(ModelGenerator):

    def __init__(self, experiment: Experiment, post_processing: PostProcess,
                 modeler: Union[AbstractModeler, str] = "Default",
                 name: str = "Processed Models",
                 use_measure: Measure = Measure.MEAN):
        super().__init__(experiment, modeler, name, use_measure)
        self.post_processing = post_processing
        self.post_processing_history = [post_processing]

    def model_all(self, progress_bar=DUMMY_PROGRESS, auto_append=True):
        raise RecoverableError("Modelling is not supported using a post-processed model set.")

    def post_process(self, post_process: PostProcess, progress_bar=DUMMY_PROGRESS, auto_append=True):
        if post_process.supports_processing(self.post_processing_history):
            post_processed_model_set = super().post_process(post_process, progress_bar, auto_append=auto_append)
            post_processed_model_set.post_processing_history = self.post_processing_history + [post_process]
            return post_processed_model_set
        else:
            raise RecoverableError(f"Processing {self.name} with {post_process.NAME} is not supported.")

    def restore_from_exp(self, experiment):
        self.experiment = experiment
        for key, model in self.models.items():
            if not model.measurements:
                model.measurements = experiment.measurements.get(key)


class AggregateModelGenerator(PostProcessedModelSet):

    def __init__(self, experiment: Experiment, aggregation: Aggregation,
                 modeler: Union[AbstractModeler, str] = "Default",
                 name: str = "New Modeler",
                 use_measure: Measure = Measure.MEAN):
        super().__init__(experiment, aggregation, modeler, name, use_measure)
        self.aggregation = aggregation

    def aggregate(self, aggregation: Aggregation, progress_bar=DUMMY_PROGRESS, auto_append=True):
        raise RecoverableError("Aggregation is not supported using an aggregated model set.")


class ModelGeneratorSchema(BaseSchema):
    name = fields.Str()
    _modeler = fields.Nested(ModelerSchema, data_key='modeler')
    models = TupleKeyDict(keys=(fields.Pluck(CallpathSchema, 'name'), fields.Pluck(MetricSchema, 'name')),
                          values=fields.Nested(ModelSchema, exclude=('callpath', 'metric')))

    def create_object(self):
        return ModelGenerator(None, NotImplemented)

    def postprocess_object(self, obj: ModelGenerator):
        for (callpath, metric), m in obj.models.items():
            m.callpath = callpath
            m.metric = metric
        return obj

    # @pre_load
    # def intercept(self, val, **kwargs):
    #     return val


class PostProcessedModelSetSchema(ModelGeneratorSchema):
    post_processing_history = fields.List(fields.Nested(PostProcessSchema))

    def create_object(self):
        return PostProcessedModelSet(None, None, NotImplemented)

    # @pre_load
    # def intercept(self, data, many, **kwargs):
    #     return data
    #
    # @post_dump
    # def intercept2(self, val, **kwargs):
    #     return val


class AggregateModelGeneratorSchema(PostProcessedModelSetSchema):
    def create_object(self):
        return AggregateModelGenerator(None, None, NotImplemented)
