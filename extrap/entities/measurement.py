# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import enum
import math
import numbers
from collections.abc import Iterable, Sequence
from itertools import chain, product
from typing import Union, Generator, Optional

import numpy as np
from marshmallow import fields, post_load
from numpy import ma

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

    def __init__(self, coordinate: Coordinate, callpath: Callpath, metric: Metric, values, *, keep_values=False,
                 repetitions=None):
        """
        Initialize the Measurement object.
        """
        self.coordinate: Coordinate = coordinate
        self.callpath: Callpath = callpath
        self.metric: Metric = metric
        if values is None:
            self.repetitions = None
            self.values = None
            return
        values = self._convert_values_to_ndarray(values)
        if keep_values:
            self.values: Optional[np.ndarray] = values
        else:
            self.values = None
        self.median: float = ma.median(values)
        self.mean: float = ma.mean(values)
        self.minimum: float = ma.min(values)
        self.maximum: float = ma.max(values)
        self.std: float = ma.std(values)
        if repetitions is not None:
            self.repetitions = repetitions
        else:
            try:
                self.repetitions: int = len(values)
            except TypeError:
                self.repetitions = 1

    @staticmethod
    def _convert_values_to_ndarray(values):
        if not isinstance(values, Sequence):
            return values
        if isinstance(values, np.ndarray):
            return values
        if all(isinstance(v, np.ndarray) for v in values):
            return np.ma.array(values)

        dim_max_len = [(len(values), len(values))]

        current_dimension = values
        while current_dimension:
            max_len, min_len = 0, math.inf
            for entry in current_dimension:
                if isinstance(entry, Sequence):
                    max_len = max(len(entry), max_len)
                    min_len = min(len(entry), min_len)
            if min_len < math.inf:
                dim_max_len.append((min_len, max_len))
                current_dimension = chain.from_iterable(current_dimension)
            else:
                break

        if all(a == b for a, b in dim_max_len):
            return np.array(values)

        base_array = np.zeros(tuple(l for _, l in dim_max_len), dtype=float)
        m_array = ma.array(base_array, mask=True)

        if len(dim_max_len) == 2:
            for index in range(dim_max_len[0][1]):
                v = values[index]
                m_array[index, slice(0, len(v))] = v
        else:
            main_indices = product(*(range(l) for _, l in dim_max_len[:-1]))
            for main_index in main_indices:
                v = values
                try:
                    for partial_index in main_index:
                        v = v[partial_index]
                except IndexError:
                    continue
                sl = slice(0, len(v))
                m_array[(*main_index, sl)] = v
        return m_array

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

    def add_repetition(self, value):
        if self.values is None:
            raise RuntimeError("Cannot add value, because the list of original values does not exist.")
        if isinstance(value, Sequence):
            self.values = ma.append(self.values, value)
        else:
            axis = tuple(range(1, self.values.ndim))
            counts = self.values.count(axis=axis)
            avg_count = int(round(np.mean(counts), 0))
            small_counts = counts.copy()
            small_counts[small_counts > avg_count] = 0
            idx = np.argmax(small_counts)
            if small_counts[idx] == avg_count:
                mask = self.values[idx].mask
            else:
                needed = avg_count - small_counts[idx]
                counts[counts <= avg_count] = counts.max() + 1
                larger_idx = np.argmin(counts)
                mask = self.values[idx].mask.copy()
                candidates = self.values[larger_idx].mask ^ mask
                positions = np.argwhere(candidates)[:needed]
                for pos in positions:
                    mask[tuple(pos)] = False

            new_value = ma.array(np.full_like(self.values[0], value), mask=mask)
            new_value = new_value.reshape(1, *new_value.shape)
            self.values = ma.append(self.values, new_value, axis=0)
        self.median = ma.median(self.values).item()
        self.mean = ma.mean(self.values).item()
        self.minimum = ma.min(self.values).item()
        self.maximum = ma.max(self.values).item()
        self.std = ma.std(self.values).item()
        self.repetitions += 1

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
            return (self.coordinate == other.coordinate and
                    self.metric == other.metric and
                    self.callpath == other.callpath and
                    self.mean == other.mean and
                    self.median == other.median)

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

    def __imul__(self, other):
        if isinstance(other, Measurement):
            if self.coordinate != other.coordinate:
                raise ValueError("Coordinate does not match while merging measurements.")
            self.median *= other.median
            self.mean *= other.mean
            self.minimum *= other.minimum
            self.maximum *= other.maximum
            # Var(XY) = E(X²Y²) - (E(XY))² = Var(X)Var(Y) + Var(X)(E(Y))² + Var(Y)(E(X))²
            self_var, other_var = self.std ** 2, other.std ** 2
            variance = self_var * other_var + self_var * other.mean ** 2 + other_var * self.mean ** 2
            self.std = np.sqrt(variance)
        else:

            self.median *= other
            self.mean *= other
            self.minimum *= other
            self.maximum *= other
            self.std *= abs(other)
        return self


class MeasurementSchema(Schema):
    coordinate = fields.Nested(CoordinateSchema)
    metric = fields.Nested(MetricSchema)
    callpath = fields.Nested(CallpathSchema)
    median = NumberField()
    mean = NumberField()
    minimum = NumberField()
    maximum = NumberField()
    std = NumberField()
    repetitions = fields.Int(allow_none=True)
    values = fields.Method('_store_values', '_load_values', allow_none=True, load_default=None)

    def _load_values(self, value):
        if value is None:
            return None
        return self.context['value_io'].read_values(*value)

    def _store_values(self, obj: Measurement):
        if obj.values is None:
            return None
        return self.context['value_io'].write_values(obj.values)

    @post_load
    def report_progress(self, data, **kwargs):
        if 'progress_bar' in self.context:
            self.context['progress_bar'].update()
        return data

    def create_object(self):
        return Measurement(None, None, None, None)
