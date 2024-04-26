# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from typing import Optional, List

import numpy
from marshmallow import fields, post_load, pre_dump

from extrap.entities.annotations import Annotation, AnnotationSchema
from extrap.entities.callpath import CallpathSchema
from extrap.entities.hypotheses import Hypothesis, HypothesisSchema
from extrap.entities.measurement import Measurement, MeasurementSchema
from extrap.entities.metric import MetricSchema
from extrap.util.caching import cached_property
from extrap.util.serialization_schema import BaseSchema


class Model:

    def __init__(self, hypothesis, callpath=None, metric=None):
        self.hypothesis: Hypothesis = hypothesis
        self.callpath = callpath
        self.metric = metric
        self.measurements: Optional[List[Measurement]] = None
        self.annotations: list[Annotation] = []

    @cached_property
    def predictions(self):
        coordinates = numpy.array([m.coordinate for m in self.measurements])
        return self.hypothesis.function.evaluate(coordinates.transpose())

    def __eq__(self, other):
        if not isinstance(other, Model):
            return NotImplemented
        elif self is other:
            return True
        else:
            return self.callpath == other.callpath and \
                self.metric == other.metric and \
                self.hypothesis == other.hypothesis and \
                self.measurements == other.measurements


class SegmentedModel(Model):
    def __init__(self, hypothesis, segment_models, changing_points, callpath=None, metric=None):
        self._callpath = None
        self._metric = None
        self._measurements = None
        self.changing_points: list[Measurement] = changing_points
        self.segment_models: list[Model] = segment_models
        super().__init__(hypothesis, callpath, metric)

    @property
    def measurements(self):
        return self._measurements

    @measurements.setter
    def measurements(self, value):
        self._measurements = value
        if value is None:
            return
        index = value.index(self.changing_points[0])
        self.segment_models[0].measurements = value[:index]
        if len(self.changing_points) == 1:
            self.segment_models[1].measurements = value[index:]
        elif len(self.changing_points) == 2:
            index2 = value.index(self.changing_points[1])
            self.segment_models[1].measurements = value[index2:]
        else:
            raise NotImplementedError()

    @property
    def callpath(self):
        return self._callpath

    @callpath.setter
    def callpath(self, value):
        self._callpath = value
        for model in self.segment_models:
            model.callpath = value

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = value
        for model in self.segment_models:
            model.metric = value


class ModelSchema(BaseSchema):
    def create_object(self):
        return Model(None)

    @post_load
    def report_progress(self, data, **kwargs):
        if 'progress_bar' in self.context:
            self.context['progress_bar'].update()
        return data

    @pre_dump
    def report_progress(self, data, **kwargs):
        if 'progress_bar' in self.context:
            self.context['progress_bar'].update()
        return data

    hypothesis = fields.Nested(HypothesisSchema)
    callpath = fields.Nested(CallpathSchema)
    metric = fields.Nested(MetricSchema)
    annotations = fields.List(fields.Nested(AnnotationSchema))


class SegmentedModelSchema(ModelSchema):
    segment_models = fields.List(fields.Nested(ModelSchema))
    changing_points = fields.List(fields.Nested(MeasurementSchema))

    def create_object(self):
        return SegmentedModel(None, [], [])
