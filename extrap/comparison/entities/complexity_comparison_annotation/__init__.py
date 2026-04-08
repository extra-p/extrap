# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import importlib.resources
import itertools

from marshmallow import fields

from extrap.entities.annotations import Annotation, AnnotationSchema, AnnotationIconSVG


class ComplexityComparisonAnnotation(Annotation):
    NAME = "Complexity Comparison"

    def __init__(self):
        self.comparison = None

    def title(self, **context) -> str:
        if self.comparison is None:
            return ""
        if isinstance(self.comparison, tuple):
            result = "Comparison result for complexity: "
            parameters = context.get('parameters', (f'p{i}' for i in itertools.count(1)))
            for p, c in zip(parameters, self.comparison):
                if isinstance(c, str):
                    result += f"{p.name}: {c} "
                    continue
                if c > 0:
                    cr = '<'
                elif c < 0:
                    cr = '>'
                else:
                    cr = '='
                result += f"{p.name}: {cr} "
            return result
        elif self.comparison > 0:
            return "Comparison result for complexity: >"
        elif self.comparison == 0:
            return "Comparison result for complexity: ="
        elif self.comparison < 0:
            return "Comparison result for complexity: <"
        raise ValueError(f"Unknown comparison value: {self.comparison}")

    def icon(self, **context) -> AnnotationIconSVG:
        if self.comparison is None:
            data = ''
        elif isinstance(self.comparison, tuple):
            data = importlib.resources.read_text(__name__, 'mult.svg')
        elif self.comparison > 0:
            data = importlib.resources.read_text(__name__, 'gt.svg')
        elif self.comparison == 0:
            data = importlib.resources.read_text(__name__, 'eq.svg')
        elif self.comparison < 0:
            data = importlib.resources.read_text(__name__, 'lt.svg')
        else:
            data = ''
        return AnnotationIconSVG(data)

    def content(self, **context):
        return self.title(**context)


class ComplexityComparisonAnnotationSchema(AnnotationSchema):
    def create_object(self):
        return ComplexityComparisonAnnotation()

    @staticmethod
    def _comparison_serialize(value):
        data = value.comparison
        return data

    @staticmethod
    def _comparison_deserialize(value):
        if isinstance(value, list):
            data = tuple(value)
        else:
            data = value
        return data

    comparison = fields.Method("_comparison_serialize", "_comparison_deserialize")
