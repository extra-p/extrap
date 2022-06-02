# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
import logging
import os
from collections import defaultdict
from itertools import islice
from typing import List

from extrap.fileio.file_reader.nv_reader.binary_parser.nsight_cuprof_report import NsightCuprofReport
from extrap.fileio.file_reader.nv_reader.pb_parser.ProfilerReport_pb2 import ProfileResult
from extrap.fileio.file_reader.nv_reader.pb_parser.ProfilerResults_pb2 import MetricValueMessage


class NcuReport:

    def __init__(self, name):
        logging.info(f"Loading NCU report {name}")
        self.string_table: List[str] = []
        self.source_blocks = []
        self.result_blocks = []
        self.report_data = NsightCuprofReport.from_file(name)
        self._parse_inital()

    def _parse_inital(self):
        for block in self.report_data.blocks:
            if block.header.data.StringTable.Strings:
                self.string_table.extend(block.header.data.StringTable.Strings)
            if block.payload.num_sources > 0:
                for source_raw in block.payload.sources:
                    source = source_raw.entry.parse()
                    if source:
                        self.source_blocks.append(source)
            if block.payload.num_results > 0:
                for results_raw in block.payload.results:
                    self.result_blocks.append(results_raw.entry)

    def get_measurements_unmapped(self, *, ignore_metrics=None):
        ignored_ids = self._calc_ignored_metric_ids(ignore_metrics)
        return _convert_measurements(self.result_blocks, ignore_metric_ids=ignored_ids)

    def _calc_ignored_metric_ids(self, ignore_metrics):
        ignored_ids = set()
        if ignore_metrics:
            for id, s in enumerate(self.string_table):
                if any(s.startswith(p) for p in ignore_metrics):
                    ignored_ids.add(id)
        return ignored_ids

    def get_measurements(self, paths):
        return _convert_and_map_measurements(zip(self.result_blocks, paths))

    def get_measurements_parallel(self, paths, pool, *, ignore_metrics=None):
        aggregated_values = defaultdict(int)
        ignored_ids = self._calc_ignored_metric_ids(ignore_metrics)

        data = zip(self.result_blocks, list(paths))
        chunk_length = int((len(self.result_blocks) + (os.cpu_count() - 1)) / os.cpu_count())
        reduced = pool.imap_unordered(_convert_and_map_measurements,
                                      [islice(data, 0 + i, chunk_length + i) for i in
                                       range(0, len(self.result_blocks), chunk_length)],
                                      1)
        for partition in reduced:
            for key, value in partition.items():
                if key[1] in ignored_ids:
                    continue
                aggregated_values[key] += value

        return aggregated_values

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.report_data.close()
        self.report_data = None


def _convert_metric_value(mv: MetricValueMessage):
    if mv.HasField('DoubleValue'):
        return mv.DoubleValue
    if mv.HasField("FloatValue"):
        return mv.FloatValue
    if mv.HasField("Uint64Value"):
        return mv.Uint64Value
    if mv.HasField("Uint32Value"):
        return mv.Uint32Value
    return None


def _convert_and_map_measurements(data):
    aggregated_values = defaultdict(int)
    for raw, (name, _, _, _, callpath) in data:
        res: ProfileResult = raw.parse()
        assert res.KernelFunctionName == name
        for mv in res.MetricResults:
            value = _convert_metric_value(mv.MetricValue)
            if value is not None:
                aggregated_values[(callpath + '->' + name + '->GPU ' + name, mv.NameId)] += value
    return aggregated_values


def _convert_measurements(raw_data, *, ignore_metric_ids=None):
    if ignore_metric_ids is None:
        ignore_metric_ids = set()
    aggregated_values = defaultdict(int)
    for raw in raw_data:
        res: ProfileResult = raw.parse()
        for mv in res.MetricResults:
            if mv.NameId in ignore_metric_ids:
                continue
            value = _convert_metric_value(mv.MetricValue)
            if value is not None:
                aggregated_values[('main->' + res.KernelFunctionName, mv.NameId)] += value
    return aggregated_values
