# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from typing import TypedDict, Optional

from extrap.entities.metric import Metric


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
        self.base_experiment_id: int = 0
        self.metrics_to_project: list = []
        self.fp_dp_metric: Optional[Metric] = None
        self.fp_sp_metric: Optional[Metric] = None
        self.num_mem_transfers: Optional[Metric] = None
        self.bytes_per_mem: int = 0
