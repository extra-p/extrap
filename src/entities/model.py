from typing import Optional, List

import numpy

from entities.hypotheses import Hypothesis
from entities.measurement import Measurement
from util.caching import cached_property
from util.deprecation import deprecated


class Model:

    def __init__(self, hypothesis, callpath=None, metric=None):
        self.hypothesis: Hypothesis = hypothesis
        self.callpath = callpath
        self.metric = metric
        self.measurements: Optional[List[Measurement]] = None

    @deprecated("Use property directly.")
    def get_hypothesis(self):
        return self.hypothesis

    @deprecated("Use property directly.")
    def get_callpath_id(self):
        return self.callpath.id

    @deprecated("Use property directly.")
    def get_metric_id(self):
        return self.metric.id

    @cached_property
    def predictions(self):
        coordinates = numpy.array([m.coordinate for m in self.measurements])
        return self.hypothesis.function.evaluate(coordinates.transpose())
