# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2026, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from marshmallow import fields

from extrap.entities.annotations import Annotation, AnnotationSchema, AnnotationIconSVG

_ICON = AnnotationIconSVG(
    '<svg width="16" height="16" version="1.1" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="8" cy="8" r="7" fill="#dbe9ff" stroke="#000" stroke-width="1"/>'
    '<text x="8" y="11" fill="#000" font-family="sans-serif" font-size="8px" '
    'text-anchor="middle">HW</text></svg>'
)

_ICON_FLAGGED = AnnotationIconSVG(
    '<svg width="16" height="16" version="1.1" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="8" cy="8" r="7" fill="#ffdbdb" stroke="#c00" stroke-width="1"/>'
    '<text x="8" y="11" fill="#c00" font-family="sans-serif" font-size="8px" '
    'text-anchor="middle">HW</text></svg>'
)

EXPONENT_DEVIATION_FLAG_THRESHOLD = 0.5


class CounterSelectionAnnotation(Annotation):
    """Records which hardware-counter selection strategy and counters produced a runtime model,
    and flags a scalability mismatch if the model's leading exponent deviates too much from the
    original runtime-only model's leading exponent."""
    NAME = "Counter Selection"

    def __init__(self, strategy: str = '', counters: list[str] = None, exponent_deviation: float = None):
        self.strategy = strategy
        self.counters: list[str] = counters if counters is not None else []
        self.exponent_deviation = exponent_deviation

    @property
    def flagged(self) -> bool:
        return self.exponent_deviation is not None and self.exponent_deviation > EXPONENT_DEVIATION_FLAG_THRESHOLD

    def title(self, **context) -> str:
        title = f"Counter selection ({self.strategy}): {', '.join(self.counters)}"
        if self.flagged:
            title += f" -- exponent deviation {self.exponent_deviation:.2f} from runtime-only model"
        return title

    def icon(self, **context) -> AnnotationIconSVG:
        return _ICON_FLAGGED if self.flagged else _ICON

    def content(self, **context):
        return self.title(**context)


class CounterSelectionAnnotationSchema(AnnotationSchema):
    strategy = fields.Str()
    counters = fields.List(fields.Str())
    exponent_deviation = fields.Float(allow_none=True)

    def create_object(self):
        return CounterSelectionAnnotation()
