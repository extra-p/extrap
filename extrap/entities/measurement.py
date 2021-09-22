# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
from __future__ import annotations

import enum
import numbers
from collections.abc import Iterable
from typing import Union, Generator

import numpy as np
from marshmallow import fields, post_load

from extrap.entities.callpath import Callpath, CallpathSchema
from extrap.entities.coordinate import Coordinate, CoordinateSchema
from extrap.entities.metric import Metric, MetricSchema
from extrap.util.serialization_schema import Schema, NumberField


class Measure(enum.Enum):
    UNKNOWN = -1
    MEAN = 0
    MEDIAN = 1
    MINIMUM = enum.auto()
    MAXIMUM = enum.auto()

    @classmethod
    def from_str(cls, name: str):
        return cls[name.upper()]

    @classmethod
    def from_use_median(cls, use_median: bool):
        if use_median:
            return Measure.MEDIAN
        else:
            return Measure.MEAN

    @classmethod
    def choices(cls):
        return [m for m in Measure if m != cls.UNKNOWN]


class Measurement:
    """
    This class represents a measurement, i.e. the value measured for a specific metric and callpath at a coordinate.
    """

    def __init__(self, coordinate: Coordinate, callpath: Callpath, metric: Metric, values):
        """
        Initialize the Measurement object.
        """
        self.coordinate: Coordinate = coordinate
        self.callpath: Callpath = callpath
        self.metric: Metric = metric
        if values is None:
            return
        values = np.array(values)
        self.median: float = np.median(values)
        self.mean: float = np.mean(values)
        self.minimum: float = np.min(values)
        self.maximum: float = np.max(values)
        self.std: float = np.std(values)

    def value(self, measure: Union[bool, Measure]):
        if measure == Measure.MEAN:
            return self.mean
        elif measure == Measure.MEDIAN:
            return self.median
        elif measure == Measure.MINIMUM:
            return self.minimum
        elif measure == Measure.MAXIMUM:
            return self.maximum
        elif measure is True:
            return self.median
        elif measure is False:
            return self.mean
        else:
            raise ValueError("Unknown measure.")

    def merge(self, other: 'Measurement') -> None:
        """Approximately merges the other measurement into this measurement."""
        if self.coordinate != other.coordinate:
            raise ValueError("Coordinate does not match while merging measurements.")
        self.median += other.median
        self.mean += other.mean
        self.minimum += other.minimum
        self.maximum += other.maximum
        self.std = np.sqrt(self.std ** 2 + other.std ** 2)

    def __repr__(self):
        return f"Measurement({self.coordinate}: {self.mean:0.6} median={self.median:0.6})"

    def __eq__(self, other):
        if not isinstance(other, Measurement):
            return False
        elif self is other:
            return True
        else:
            return self.coordinate == other.coordinate and \
                   self.metric == other.metric and \
                   self.callpath == other.callpath and \
                   self.mean == other.mean and \
                   self.median == other.median

    @staticmethod
    def select_measure(measurements: Iterable[Measurement], measure: Union[bool, Measure]) -> Generator[numbers.Real]:
        if measure == Measure.MEAN:
            return (m.mean for m in measurements)
        elif measure == Measure.MEDIAN:
            return (m.median for m in measurements)
        elif measure == Measure.MINIMUM:
            return (m.minimum for m in measurements)
        elif measure == Measure.MAXIMUM:
            return (m.maximum for m in measurements)
        elif measure is True:
            return (m.median for m in measurements)
        elif measure is False:
            return (m.mean for m in measurements)
        else:
            raise ValueError("Unknown measure.")


class MeasurementSchema(Schema):
    coordinate = fields.Nested(CoordinateSchema)
    metric = fields.Nested(MetricSchema)
    callpath = fields.Nested(CallpathSchema)
    median = NumberField()
    mean = NumberField()
    minimum = NumberField()
    maximum = NumberField()
    std = NumberField()

    @post_load
    def report_progress(self, data, **kwargs):
        if 'progress_bar' in self.context:
            self.context['progress_bar'].update()
        return data

    def create_object(self):
        return Measurement(None, None, None, None)
