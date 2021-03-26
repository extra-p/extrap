# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import os
from collections import defaultdict
from itertools import islice

from extrap.fileio.file_reader.nv_reader.binary_parser.nsight_cuprof_report import NsightCuprofReport
from extrap.fileio.file_reader.nv_reader.pb_parser.ProfilerReport_pb2 import ProfileResult
from extrap.fileio.file_reader.nv_reader.pb_parser.ProfilerResults_pb2 import MetricValueMessage


class NcuReport:

    def __init__(self, name):

        self.string_table = []
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

    def get_measurements(self, paths):
        return self._convert_measurements(zip(self.result_blocks, paths))

    @staticmethod
    def _convert_measurements(data):
        aggregated_values = defaultdict(int)
        for raw, (name, _, _, _, callpath) in data:
            res: ProfileResult = raw.parse()
            assert res.KernelFunctionName == name
            for mv in res.MetricResults:
                aggregated_values[(callpath + '->' + name + '->GPU', mv.NameId)] += NcuReport.convertMetricValue(
                    mv.MetricValue)
        return aggregated_values

    def get_measurements_parallel(self, paths, pool):
        aggregated_values = defaultdict(int)

        data = zip(self.result_blocks, list(paths))
        chunk_length = int((len(self.result_blocks) + (os.cpu_count() - 1)) / os.cpu_count())
        reduced = pool.imap_unordered(self._convert_measurements,
                                      [islice(data, 0 + i, chunk_length + i) for i in
                                       range(0, len(self.result_blocks), chunk_length)],
                                      1)
        for partition in reduced:
            for key, value in partition.items():
                aggregated_values[key] += value

        return aggregated_values

    @staticmethod
    def convertMetricValue(mv: MetricValueMessage):
        if mv.HasField('DoubleValue'):
            return mv.DoubleValue
        if mv.HasField("FloatValue"):
            return mv.FloatValue
        if mv.HasField("Uint64Value"):
            return mv.Uint64Value
        if mv.HasField("Uint32Value"):
            return mv.Uint32Value
        return float('nan')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.report_data = None
