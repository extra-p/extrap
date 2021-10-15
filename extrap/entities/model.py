# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Optional, List

import numpy
from marshmallow import fields, post_load

from extrap.entities.callpath import CallpathSchema
from extrap.entities.functions import ConstantFunction
from extrap.entities.hypotheses import Hypothesis, HypothesisSchema, ConstantHypothesis
from extrap.entities.measurement import Measurement
from extrap.entities.metric import MetricSchema
from extrap.util.caching import cached_property
from extrap.util.serialization_schema import BaseSchema


class Model:

    def __init__(self, hypothesis, callpath=None, metric=None):
        self.hypothesis: Hypothesis = hypothesis
        self.callpath = callpath
        self.metric = metric
        self.measurements: Optional[List[Measurement]] = None

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


class ModelSchema(BaseSchema):
    def create_object(self):
        return Model(None)

    @post_load
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


NULL_MODEL = _NullModel()


class _NullModelSchema(ModelSchema):
    callpath = fields.Constant(None, dump_only=True, load_only=True)
    metric = fields.Constant(None, dump_only=True, load_only=True)
    hypothesis = fields.Constant(None, dump_only=True, load_only=True)

    def unpack_to_object(self, data, **kwargs):
        return NULL_MODEL

    def create_object(self):
        return NULL_MODEL
