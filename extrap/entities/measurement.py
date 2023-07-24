# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy

import numpy as np
from marshmallow import fields, post_load

from extrap.entities.calculation_element import CalculationElement
from extrap.entities.callpath import Callpath, CallpathSchema
from extrap.entities.coordinate import Coordinate, CoordinateSchema
from extrap.entities.metric import Metric, MetricSchema
from extrap.util.serialization_schema import Schema, NumberField


class Measurement(CalculationElement):
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

    def value(self, use_median):
        return self.median if use_median else self.mean

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
        else:

            self.median *= other
            self.mean *= other
            self.minimum *= other
            self.maximum *= other
            self.std *= abs(other)
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
        return result

    def __truediv__(self, other):
        # TODO implement correct division
        return self.__mul__(other ** -1)

    def __rtruediv__(self, other):
        return (self ** -1).__mul__(other)

    def __neg__(self):
        return self * -1

    def make_one(self):
        result = copy.copy(self)
        result.median = 1
        result.mean = 1
        result.minimum = 1
        result.maximum = 1
        result.std = 0
        return result


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
