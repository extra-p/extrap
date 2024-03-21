# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Optional, Union, Tuple, Type

import numpy as np
from marshmallow import fields, post_load

from extrap.entities.calculation_element import CalculationElement
from extrap.entities.callpath import Callpath, CallpathSchema
from extrap.entities.coordinate import Coordinate, CoordinateSchema
from extrap.entities.metric import Metric, MetricSchema
from extrap.util.serialization_schema import Schema, NumberField


@dataclasses.dataclass
class ValueCount:
    count: int = 0
    repetitions: int = 1

    @property
    def data(self) -> tuple:
        return self.count, self.repetitions


class Measurement(CalculationElement):
    """
    This class represents a measurement, i.e. the value measured for a specific metric and callpath at a coordinate.
    """

    def __init__(self, coordinate: Coordinate, callpath: Callpath, metric: Metric, values, *, keep_values=False):
        """
        Initialize the Measurement object.
        """
        self.coordinate: Coordinate = coordinate
        self.callpath: Callpath = callpath
        self.metric: Metric = metric
        if values is None:
            self.count: Optional[ValueCount] = None
            self.values = None
            return
        values = np.array(values)
        if keep_values:
            self.values: Optional[np.typing.NDArray] = values
        else:
            self.values = None
        self.median: float = np.median(values)
        self.mean: float = np.mean(values)
        self.minimum: float = np.min(values)
        self.maximum: float = np.max(values)
        self.std: float = np.std(values)
        self.count: Optional[ValueCount] = ValueCount(values.size)

    def value(self, use_median):
        return self.median if use_median else self.mean

    def add_value(self, value):
        if not self.values:
            raise RuntimeError("Cannot add value, because the list of original values does not exist.")
        self.values = np.append(self.values, value)
        self.median = np.median(self.values)
        self.mean = np.mean(self.values)
        self.minimum = np.min(self.values)
        self.maximum = np.max(self.values)
        self.std = np.std(self.values)
        self.count = ValueCount(len(self.values))

    def merge(self, other: 'Measurement') -> None:
        """Approximately merges the other measurement into this measurement."""
        if self.coordinate != other.coordinate:
            raise ValueError("Coordinate does not match while merging measurements.")
        self.median += other.median
        self.mean += other.mean
        self.minimum += other.minimum
        self.maximum += other.maximum
        self.std = np.sqrt(self.std ** 2 + other.std ** 2)
        if self.values and other.values:
            self.values += other.values
        else:
            self.values = None
        # TODO merge count

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

    def __add__(self, other):
        if isinstance(other, Measurement):
            result = copy.copy(self)
            result.merge(other)
        else:
            result = copy.copy(self)
            result.median += other
            result.mean += other
            result.minimum += other
            result.maximum += other
            result.std = self.std
            if self.values:
                self.values += other
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __rsub__(self, other):
        return (self * -1).__add__(other)

    def __mul__(self, other):
        result = copy.copy(self)
        result *= other
        return result

    def __imul__(self, other):
        if isinstance(other, Measurement):
            if self.coordinate != other.coordinate:
                raise ValueError("Coordinate does not match while merging measurements.")
            self.median *= other.median
            self.mean *= other.mean
            self.minimum *= other.minimum
            self.maximum *= other.maximum
            # Var(XY) = E(X²Y²) − (E(XY))² = Var(X)Var(Y) + Var(X)(E(Y))² + Var(Y)(E(X))²
            self_var, other_var = self.std ** 2, other.std ** 2
            variance = self_var * other_var + self_var * other.mean ** 2 + other_var * self.mean ** 2
            self.std = np.sqrt(variance)
            if self.values and other.values:
                self.values *= other.values
            else:
                self.values = None
        else:

            self.median *= other
            self.mean *= other
            self.minimum *= other
            self.maximum *= other
            self.std *= abs(other)
            if self.values:
                self.values *= other
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        result = copy.copy(self)
        result.median **= other
        result.mean **= other
        result.minimum **= other
        result.maximum **= other
        variance = (self.std ** 2 + self.mean ** 2) ** other - (self.mean ** 2) ** other
        result.std = np.sqrt(variance)
        if result.values:
            result.values **= other
        return result

    def __itruediv__(self, other):
        if isinstance(other, Measurement):
            if self.coordinate != other.coordinate:
                raise ValueError("Coordinate does not match while merging measurements.")
            self.median /= other.median
            self.mean /= other.mean
            minimum = self.minimum / other.minimum
            maximum = self.maximum / other.maximum
            self.minimum = min(minimum, maximum, self.median)
            self.maximum = max(minimum, maximum, self.median)
            # Var(Y/X)~= Var(Y)/E(X)^2 + (E(Y)^2*Var(X))/E(X)^4 - (2*E(Y)*Cov(X,Y))/E(X)^3
            variance = self.std ** 2 / other.mean ** 2 + (self.mean ** 2 * other.std ** 2) / other.mean ** 4 \
                       - (2 * self.mean * self.std * other.std) / other.mean ** 3
            self.std = math.sqrt(variance)
            if self.values and other.values:
                self.values /= other.values
                self.std = np.std(self.values)
            else:
                self.values = None
        else:
            self.median /= other
            self.mean /= other
            self.minimum /= other
            self.maximum /= other
            variance = self.std ** 2 / other ** 2
            self.std = math.sqrt(variance)
            if self.values:
                self.values /= other
        return self

    def __truediv__(self, other):
        result = copy.copy(self)
        result /= other
        return result

    @staticmethod
    def divide_no0(a: Measurement, b: Measurement) -> Measurement:
        result = copy.copy(a)
        if b.mean == 0:
            result.mean = 1 if a.mean == 0 else math.inf
        else:
            result.mean /= b.mean
        if b.median == 0:
            result.median = 1 if a.median == 0 else math.inf
        else:
            result.median /= b.median

        if b.minimum == 0:
            minimum = 1 if a.minimum == 0 else math.inf
        else:
            minimum = a.minimum / b.minimum
        if b.maximum == 0:
            maximum = 1 if a.maximum == 0 else math.inf
        else:
            maximum = a.maximum / b.maximum
        result.minimum = min(minimum, maximum, result.median)
        result.maximum = max(minimum, maximum, result.median)
        # Var(Y/X)~= Var(Y)/E(X)^2 + (E(Y)^2*Var(X))/E(X)^4 - (2*E(Y)*Cov(X,Y))/E(X)^3
        if b.mean == 0:
            result.std = math.nan
        else:
            variance = a.std ** 2 / b.mean ** 2 + (a.mean ** 2 * b.std ** 2) / b.mean ** 4 \
                       - (2 * a.mean * a.std * b.std) / b.mean ** 3
            result.std = math.sqrt(variance)

        if a.values and b.values:
            divisor = b.values.copy()
            divisor[b.values == 0 & a.values == 0] = 1
            result.values = a.values / divisor
        else:
            result.values = None
        return result

    def __rtruediv__(self, other):
        if isinstance(other, Measurement):
            if self.coordinate != other.coordinate:
                raise ValueError("Coordinate does not match while merging measurements.")
            # TODO implement correct division
            return (self ** -1).__mul__(other)
        else:
            result = copy.copy(self)
            if self.median == 0:
                result.median = math.inf
            else:
                result.median = other / self.median
            if self.mean == 0:
                result.mean = math.inf
            else:
                result.mean = other / self.mean
            if self.maximum == 0:
                result.minimum = math.inf
            else:
                result.minimum = other / self.maximum
            if self.minimum == 0:
                result.maximum = math.inf
            else:
                result.maximum = other / self.minimum
            # Var(Y/X)~= Var(Y)/E(X)^2 + (E(Y)^2*Var(X))/E(X)^4 - (2*E(Y)*Cov(X,Y))/E(X)^3
            if self.mean == 0:
                result.std = math.nan
            else:
                variance = (other ** 2 * self.std ** 2) / self.mean ** 4
                result.std = math.sqrt(variance)

            if self.values:
                result.values = other / self.values
            else:
                result.values = None
        return self

    def __neg__(self):
        return self * -1

    def make_one(self):
        result = copy.copy(self)
        result.median = 1
        result.mean = 1
        result.minimum = 1
        result.maximum = 1
        result.std = 0
        if result.values:
            result.values = np.ones_like(result.values)
        return result

    def copy(self):
        return copy.copy(self)


class ValueCountSchema(Schema):
    def create_object(self) -> Union[object, Tuple[type(NotImplemented), Type]]:
        return ValueCount()

    def dump(self, obj, *, many: bool = None):
        return obj.data

    def load(self, data, *, many: bool = None, partial=None, unknown: str = None):
        return ValueCount(*data)


class MeasurementSchema(Schema):
    coordinate = fields.Nested(CoordinateSchema)
    metric = fields.Nested(MetricSchema)
    callpath = fields.Nested(CallpathSchema)
    median = NumberField()
    mean = NumberField()
    minimum = NumberField()
    maximum = NumberField()
    std = NumberField()
    values = fields.Method('_store_values', '_load_values', allow_none=True, load_default=None)
    count = fields.Nested(ValueCountSchema, allow_none=True)

    def _load_values(self, value):
        if value is None:
            return None
        return self.context['value_io'].read_values(*value)

    def _store_values(self, obj: Measurement):
        if not hasattr(obj, 'values') or obj.values is None:
            return None
        return self.context['value_io'].write_values(obj.values)

    @post_load
    def report_progress(self, data, **kwargs):
        if 'progress_bar' in self.context:
            self.context['progress_bar'].update()
        return data

    def create_object(self):
        return Measurement(None, None, None, None)
