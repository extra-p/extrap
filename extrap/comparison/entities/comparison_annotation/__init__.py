# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import importlib.resources
import math

import numpy
from marshmallow import fields

from extrap.entities.annotations import Annotation, AnnotationSchema, AnnotationIconSVG


class ComparisonAnnotation(Annotation):
    NAME = 'Comparison'
    relative_tolerance = 1e-3

    def __init__(self):
        self.function_a = None
        self.function_b = None

    def title(self, **context) -> str:
        parameter_values = context.get('parameter_values')
        if parameter_values is None:
            return ''

        if self.function_a is None or self.function_b is None:
            return ''

        previous = numpy.seterr(divide='ignore', invalid='ignore')
        res_a = self.function_a.evaluate(parameter_values)
        res_b = self.function_b.evaluate(parameter_values)
        comparison = res_a - res_b
        numpy.seterr(**previous)

        if math.isclose(res_a, res_b, rel_tol=self.relative_tolerance):
            return f'Comparison result: {res_a} ≈ {res_b}'
        elif comparison > 0:
            return f'Comparison result: {res_a} > {res_b}\nDifference: {abs(res_a - res_b)}'
        elif comparison < 0:
            return f'Comparison result: {res_a} < {res_b}\nDifference: {abs(res_a - res_b)}'
        else:
            return ''

    def icon(self, **context) -> AnnotationIconSVG:
        parameter_values = context.get('parameter_values')
        if parameter_values is None:
            return AnnotationIconSVG('')

        if self.function_a is None or self.function_b is None:
            return AnnotationIconSVG('')

        previous = numpy.seterr(divide='ignore', invalid='ignore')
        res_a = self.function_a.evaluate(parameter_values)
        res_b = self.function_b.evaluate(parameter_values)
        comparison = res_a - res_b
        numpy.seterr(**previous)

        if math.isclose(res_a, res_b, rel_tol=self.relative_tolerance):
            data = importlib.resources.read_text(__name__, 'approx.svg')
        elif comparison > 0:
            data = importlib.resources.read_text(__name__, 'gt.svg')
        elif comparison < 0:
            data = importlib.resources.read_text(__name__, 'lt.svg')
        else:
            data = ''
        return AnnotationIconSVG(data)

    def content(self, **context):
        return self.title(**context)

    def init_with_comparison_model(self, model):
        functions = [m.hypothesis.function for m in model.models]
        self.function_a = functions[0]
        self.function_b = functions[1]


class ComparisonAnnotationSchema(AnnotationSchema):
    def create_object(self):
        return ComparisonAnnotation()

    comparison = fields.Integer()
