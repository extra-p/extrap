# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
from __future__ import annotations

import copy
from typing import Optional, List

import numpy
from marshmallow import fields, post_load, pre_dump

from extrap.entities.callpath import CallpathSchema
from extrap.entities.functions import ConstantFunction
from extrap.entities.hypotheses import Hypothesis, HypothesisSchema, ConstantHypothesis
from extrap.entities.measurement import Measurement
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

    @cached_property
    def predictions(self):
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
            return self.callpath == other.callpath and \
                   self.metric == other.metric and \
                   self.hypothesis == other.hypothesis and \
                   self.measurements == other.measurements


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

    @post_load
    def unpack_to_object(self, data, **kwargs):
        return Model.ZERO

    def create_object(self):
        return Model.ZERO
