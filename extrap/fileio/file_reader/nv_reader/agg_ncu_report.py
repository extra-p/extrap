# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import logging

import numpy as np

from extrap.util.caching import cached_property
from extrap.util.exceptions import FileFormatError


class AggNcuReport:
    _MAGIC_NUMBER = 'Extra-P NCU Aggregation'
    _BEGIN_PREFIX = '!BEGIN '
    _END_PREFIX = '!END '
    _SEP = '\x1E'
    _USEP = '\x1F'

    def __init__(self, path):
        logging.info(f"Loading aggregated NCU report {path}")
        self.path = path
        self._file = open(path, 'r')
        self.string_table: list[str] = []
        self.kernel_names = {}
        self.info = {}
        self.unmapped_measurements = {}
        self.mapped_measurements = {}
        self._parse()

    @cached_property
    def count(self):
        return int(self.info['count'])

    def _parse(self):
        seen_magic_number = None
        state = None
        for line in self._file:
            line = line[:-1]
            if not seen_magic_number:
                seen_magic_number = self._check_magic_number(line, seen_magic_number)
                continue
            if state is None:
                if line.startswith(self._BEGIN_PREFIX):
                    state = line[len(self._BEGIN_PREFIX):]
                    continue
            elif line.startswith(self._END_PREFIX):
                if state != line[len(self._END_PREFIX):]:
                    raise FileFormatError(f"BEGIN {state} and END {line[len(self._END_PREFIX):]:} did not match.")
                state = None
                continue
            elif state == 'Info':
                key, value = line.split(self._SEP, 1)
                self.info[key] = value
            elif state == 'Strings':
                self.string_table.append(line)
            elif state == 'Kernels':
                key, value = line.split(self._SEP, 1)
                self.kernel_names[int(key)] = value
            elif state == 'Aggregated Measurements Kernel':
                kernel_id, metric_id, value = line.split(self._SEP, 2)
                key = int(kernel_id), int(metric_id)
                self.unmapped_measurements[key] = np.fromstring(value, dtype=np.float, sep=self._USEP)
            elif state == 'Aggregated Measurements Callpath':
                callpath, metricId, value = line.split(self._SEP, 2)
                metricId = int(metricId)
                key = callpath, metricId
                self.mapped_measurements[key] = np.fromstring(value, dtype=np.float, sep=self._USEP)

    def get_measurements(self, *, ignore_metrics=None):
        ignore_metric_ids = self._calc_ignored_metric_ids(ignore_metrics)
        aggregated_values = {(callpath, metric_id): values
                             for (callpath, metric_id), values in self.mapped_measurements.items()
                             if metric_id not in ignore_metric_ids}
        return aggregated_values

    def get_measurements_unmapped(self, *, ignore_metrics=None):
        ignore_metric_ids = self._calc_ignored_metric_ids(ignore_metrics)
        aggregated_values = {(kernel_id, metric_id): values
                             for (kernel_id, metric_id), values in self.unmapped_measurements.items()
                             if metric_id not in ignore_metric_ids}
        assert all(len(v) == self.count for v in self.unmapped_measurements.values())
        return aggregated_values

    def _calc_ignored_metric_ids(self, ignore_metrics):
        ignored_ids = set()
        if ignore_metrics:
            for id, s in enumerate(self.string_table):
                if any(s.startswith(p) for p in ignore_metrics):
                    ignored_ids.add(id)
        return ignored_ids

    def _check_magic_number(self, line, seen_magic_number):
        if line.startswith(self._MAGIC_NUMBER):
            seen_magic_number = line
        else:
            raise FileFormatError("This file is not in Extra-P NCU Aggregation format.")
        return seen_magic_number

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()
        self._file = None
        self.measurements = None
