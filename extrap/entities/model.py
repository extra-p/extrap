# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Optional, List

import numpy
from marshmallow import fields, post_load

from extrap.entities.callpath import CallpathSchema
from extrap.entities.hypotheses import Hypothesis, HypothesisSchema
from extrap.entities.measurement import Measurement
from extrap.entities.metric import MetricSchema
from extrap.util.caching import cached_property
from extrap.util.deprecation import deprecated
from extrap.util.serialization_schema import Schema


class Model:

    def __init__(self, hypothesis, callpath=None, metric=None):
        self.hypothesis: Hypothesis = hypothesis
        self.callpath = callpath
        self.metric = metric
        self.measurements: Optional[List[Measurement]] = None

    @deprecated("Use property directly.")
    def get_hypothesis(self):
        return self.hypothesis

    @deprecated("Use property directly.")
    def get_callpath_id(self):
        return self.callpath.id

    @deprecated("Use property directly.")
    def get_metric_id(self):
        return self.metric.id

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


class ModelSchema(Schema):
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
