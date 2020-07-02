"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""
from util.deprecation import deprecated

import numpy as np
from entities.coordinate import Coordinate
from entities.metric import Metric
from entities.callpath import Callpath


class Measurement:
    """
    This class represents a measurement, i.e. the value measured for a specific metric and callpath at a coordinate.
    """

    def __init__(self, coordinate: Coordinate, callpath, metric, values):
        """
        Initialize the Measurement object.
        """
        self.coordinate: Coordinate = coordinate
        self.callpath: Callpath = callpath
        self.metric: Metric = metric
        self.median: float = np.median(values)
        self.mean: float = np.mean(values)
        self.minimum: float = np.min(values)
        self.maximum: float = np.max(values)
        self.std: float = np.std(values)

    def value(self, use_median):
        return self.median if use_median else self.mean

    @deprecated("Use callpath instead.")
    def get_callpath_id(self):
        """
        Return the callpath id of the measurement.
        """
        return self.callpath.id

    @deprecated("Use metric instead.")
    def get_metric_id(self):
        """
        Return the metric id of the measurement.
        """
        return self.metric.id

    @deprecated("Use coordinate instead.")
    def get_coordinate_id(self):
        """
        Return the coordinate id of the measurement.
        """
        return self.coordinate.id

    @deprecated("Use mean property instead.")
    def get_value_mean(self):
        """
        Return the mean measured value.
        """
        return self.mean

    @deprecated("Use median property instead.")
    def get_value_median(self):
        """
        Return the median measured value.
        """
        return self.median

    @deprecated("Use mean property instead.")
    def set_value_mean(self, value_mean):
        """
        Set the mean measured value to the given value.
        """
        self.mean = value_mean

    @deprecated("Use median property instead.")
    def set_value_median(self, value_median):
        """
        Set the median measured value to the given value.
        """
        self.median = value_median

    def __repr__(self):
        return f"Measurement({self.coordinate}: {self.mean:0.6} median={self.median:0.6})"
