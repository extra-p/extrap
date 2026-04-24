# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2025, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import copy
from typing import Optional, List
from typing import Sequence

import numpy
from marshmallow import fields, post_load, pre_dump

from extrap.entities.annotations import Annotation, AnnotationSchema
from extrap.entities.callpath import CallpathSchema
from extrap.entities.functions import ConstantFunction
from extrap.entities.hypotheses import ConstantHypothesis
from extrap.entities.hypotheses import Hypothesis, HypothesisSchema
from extrap.entities.measurement import Measurement, MeasurementSchema
from extrap.entities.metric import MetricSchema
from extrap.util.caching import cached_property
from extrap.util.serialization_schema import BaseSchema


class Model:
    ZERO: Model

    def __init__(self, hypothesis, callpath=None, metric=None):
        self.hypothesis: Hypothesis = hypothesis
        self.callpath = callpath
        self.metric = metric
        self.measurements: Optional[List[Measurement]] = None
        self.annotations: list[Annotation] = []

    @cached_property
    def predictions(self):
        if not self.measurements:
            return []
        coordinates = numpy.array([m.coordinate for m in self.measurements])
        return self.hypothesis.function.evaluate(coordinates.transpose())

    def with_callpath(self, callpath):
        model = copy.copy(self)
        model.callpath = callpath
        if self.measurements:
            model.measurements = []
            for m in self.measurements:
                m = copy.copy(m)
                m.callpath = callpath
                model.measurements.append(m)
        return model

    def __eq__(self, other):
        if not isinstance(other, Model):
            return NotImplemented
        elif self is other:
            return True
        else:
            return (self.callpath == other.callpath and
                    self.metric == other.metric and
                    self.hypothesis == other.hypothesis and
                    self.measurements == other.measurements)


class SegmentedModel(Model):
    def __init__(self, hypothesis, segment_models, changing_points, callpath=None, metric=None):
        self._callpath = None
        self._metric = None
        self._measurements = None
        self.changing_points: list[Measurement] = changing_points
        self.segment_models: list[Model] = segment_models
        super().__init__(hypothesis, callpath, metric)

    class MeasurementSegmentView(Sequence[Measurement]):
        def __init__(self, measurements: Sequence[Measurement], changing_points: Sequence[Measurement], segment_id):
            self._measurements = measurements
            self._changing_points = changing_points
            if segment_id > 1:
                raise ValueError("Segment ID must be less than 2.")
            self._segment_id = segment_id

        def __len__(self):
            if self._segment_id == 0:
                index = self._measurements.index(self._changing_points[0])
                return index + 1
            elif self._segment_id == 1:
                index = self._get_segment1_index()
                return len(self._measurements) - index
            else:
                raise NotImplementedError()

        def _get_segment1_index(self) -> int:
            if len(self._changing_points) == 1:
                index = self._measurements.index(self._changing_points[0])
            elif len(self._changing_points) == 2:
                index = self._measurements.index(self._changing_points[1])
            else:
                raise NotImplementedError()
            return index

        def __getitem__(self, i):
            measurements = self._measurements
            if self._segment_id == 0:
                index = measurements.index(self._changing_points[0])
                return measurements[:index + 1][i]
            elif self._segment_id == 1:
                index = self._get_segment1_index()
                return measurements[index:][i]
            else:
                raise NotImplementedError()

        def __repr__(self):
            return f"MS({self._segment_id}: {list(self)})"

    @property
    def measurements(self):
        return self._measurements

    @measurements.setter
    def measurements(self, value):
        self._measurements = value
        if value is None:
            return
        self.segment_models[0].measurements = self.MeasurementSegmentView(self._measurements, self.changing_points, 0)
        self.segment_models[1].measurements = self.MeasurementSegmentView(self._measurements, self.changing_points, 1)

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


class _NullModel(Model):
    def __init__(self):
        hypothesis = ConstantHypothesis(ConstantFunction(0), False)
        hypothesis.compute_cost([])
        super(_NullModel, self).__init__(hypothesis)

    def __eq__(self, o: object) -> bool:
        return o is self or isinstance(o, _NullModel)

    @property
    def predictions(self):
        return None


Model.ZERO = _NullModel()


class _NullModelSchema(ModelSchema):
    callpath = fields.Constant(None, dump_only=True, load_only=True)
    metric = fields.Constant(None, dump_only=True, load_only=True)
    hypothesis = fields.Constant(None, dump_only=True, load_only=True)
    annotations = fields.Constant(None, dump_only=True, load_only=True)

    @post_load
    def unpack_to_object(self, data, **kwargs):
        return Model.ZERO

    def create_object(self):
        return Model.ZERO
