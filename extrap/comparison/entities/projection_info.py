# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from typing import TypedDict, Optional

from marshmallow import fields

from extrap.entities.metric import Metric, MetricSchema
from extrap.util.serialization_schema import Schema
from extrap.util.unique_list import UniqueList


class RooflineValue(TypedDict):
    metadata: dict[str, list]
    data: list[tuple[str, float]]


class RooflineData(TypedDict):
    gflops: RooflineValue
    gbytes: RooflineValue


class RooflineInfo(TypedDict):
    empirical: RooflineData
    spec: RooflineData


class ProjectionInfo:

    def __init__(self, number_experiments):
        self.peak_performance_in_gflops_per_s: list[float] = [0] * number_experiments
        self.peak_mem_bandwidth_in_gbytes_per_s: list[float] = [0] * number_experiments
        self.target_experiment_id: int = 1
        self.metrics_to_project: UniqueList = UniqueList()
        self.fp_dp_metric: Optional[Metric] = None
        self.num_mem_transfers_metric: Optional[Metric] = None
        self.bytes_per_mem: int = 0


class ProjectionInfoSchema(Schema):
    peak_performance_in_gflops_per_s = fields.List(fields.Float())
    peak_mem_bandwidth_in_gbytes_per_s = fields.List(fields.Float())
    target_experiment_id = fields.Integer()
    metrics_to_project = fields.List(fields.Nested(MetricSchema), list_type=UniqueList)
    fp_dp_metric = fields.Nested(MetricSchema, load_default=None)
    num_mem_transfers_metric = fields.Nested(MetricSchema, load_default=None)
    bytes_per_mem = fields.Integer()

    def create_object(self):
        return ProjectionInfo(0)
