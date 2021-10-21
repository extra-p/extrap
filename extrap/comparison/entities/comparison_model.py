from functools import reduce
from typing import Sequence

from marshmallow import fields

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
        hypothesis._RE = reduce(lambda a, b: a * b, (m.hypothesis.RE for m in models)) ** (1 / len(models))
        hypothesis._rRSS = reduce(lambda a, b: a * b, (m.hypothesis.rRSS for m in models)) ** (1 / len(models))
        hypothesis._AR2 = reduce(lambda a, b: a * b, (m.hypothesis.AR2 for m in models)) ** (1 / len(models))
        hypothesis._SMAPE = reduce(lambda a, b: a * b, (m.hypothesis.SMAPE for m in models)) ** (1 / len(models))
        hypothesis._costs_are_calculated = True
        return hypothesis


class ComparisonModelSchema(ModelSchema):
    def create_object(self):
        return ComparisonModel(None, None, None)

    models = fields.List(fields.Nested(ModelSchema))

    def postprocess_object(self, obj: object) -> object:
        return obj
