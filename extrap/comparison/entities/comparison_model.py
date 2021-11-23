# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from functools import reduce
from typing import Sequence

from marshmallow import fields, pre_load

from extrap.comparison.entities.comparison_function import ComparisonFunction
from extrap.entities.model import Model, ModelSchema


class ComparisonModel(Model):
    def __init__(self, callpath, metric, models: Sequence[Model]):
        if models is None:
            super().__init__(None)
            return
        super().__init__(self._make_comparison_hypothesis(models), callpath, metric)
        self.models = models

    def predictions(self):
        raise NotImplementedError()

    @staticmethod
    def _make_comparison_hypothesis(models):
        function = ComparisonFunction([m.hypothesis.function for m in models])
        hypothesis_type = type(models[0].hypothesis)
        hypothesis = hypothesis_type(function, models[0].hypothesis._use_median)
        hypothesis._RSS = sum(m.hypothesis.RSS for m in models)
        hypothesis._RE = reduce(lambda a, b: abs(a * b), (m.hypothesis.RE for m in models)) ** (1 / len(models))
        hypothesis._rRSS = reduce(lambda a, b: a * b, (m.hypothesis.rRSS for m in models)) ** (1 / len(models))
        hypothesis._AR2 = reduce(lambda a, b: a * b,
                                 (m.hypothesis.AR2 if m.hypothesis.AR2 > 0 else 0 for m in models)) ** (1 / len(models))
        hypothesis._SMAPE = reduce(lambda a, b: a * b, (m.hypothesis.SMAPE for m in models)) ** (1 / len(models))
        hypothesis._costs_are_calculated = True
        return hypothesis


class ComparisonModelSchema(ModelSchema):
    def create_object(self):
        return ComparisonModel(None, None, None)

    models = fields.List(fields.Nested(ModelSchema))

    @pre_load
    def intercept(self, obj, **kwargs):
        return obj

    def postprocess_object(self, obj: object) -> object:
        return obj
