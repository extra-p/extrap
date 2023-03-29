# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import logging
from enum import Enum, Flag
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Union

import msgpack
import numpy as np

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree, Node
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio import io_helper
from extrap.fileio.file_reader.abstract_directory_reader import AbstractDirectoryReader
from extrap.util.exceptions import FileFormatError
from extrap.util.progress_bar import ProgressBar, DUMMY_PROGRESS

METRIC_TIME = Metric('time')
METRIC_VISITS = Metric('visits')
METRIC_BYTES = Metric('bytes')


class CallTreeNodeType(Enum):
    NONE = 0
    KERNEL_LAUNCH = 1
    KERNEL = 2
    MEMCPY = 3
    MEMSET = 4
    SYNCHRONIZE = 5
    OVERHEAD = 6
    OVERLAP = 7
    MEMMGMT = 8


class CallTreeNodeFlags(Flag):
    NONE = 0
    ASYNC = 1 << 0
    OVERLAP = 1 << 1


class _ExtraProfInputNode(Node):
    childs: _ExtraProfInputNode

    def __init__(self, name: str, path: Callpath, num_values):
        super().__init__(name, path)
        self.duration = np.zeros(num_values)
        self.bytes = np.zeros(num_values)
        self.visits = np.zeros(num_values)
        self.disable_exclusive_conversion = False


class ExtraProf2Reader(AbstractDirectoryReader):
    NAME = "extra-prof"
    GUI_ACTION = "Open set of &Extra-Prof files"
    DESCRIPTION = "Load a set of ExtraProf files and generate a new experiment"
    CMD_ARGUMENT = "--extra-prof"
    LOADS_FROM_DIRECTORY = True

    def read_experiment(self, path: Union[Path, str], progress_bar: ProgressBar = DUMMY_PROGRESS) -> Experiment:
        if isinstance(path, list):
            files = path
        else:
            files = self._find_files_in_directory(path, '*/*.extra-prof.msgpack', progress_bar)
        # iterate over all folders and read the profiles in them
        experiment = Experiment()

        parameter_names, parameter_values = self._determine_parameters_from_paths(files, progress_bar)
        for p in parameter_names:
            experiment.add_parameter(Parameter(p))

        call_tree = None

        reordered_files = sorted(zip(files, parameter_values), key=itemgetter(1))
        for parameter_value, point_group in groupby(reordered_files, key=itemgetter(1)):
            point_group = list(point_group)
            coordinate = Coordinate(parameter_value)
            experiment.add_coordinate(coordinate)

            call_tree = CallTree()
            call_tree.path = Callpath.EMPTY
            for i, (path, _) in enumerate(point_group):
                with open(path, 'rb') as profile_file:
                    profile_data = msgpack.Unpacker(profile_file, raw=False)
                    try:
                        magic_string, version, ep_call_tree = next(iter(profile_data))
                    except ValueError as e:
                        raise FileFormatError(f"File {path} is no valid Extra-Prof file.") from e
                    if magic_string != "EXTRA PROF":
                        raise FileFormatError(f"File {path} is no valid Extra-Prof file.")
                    self._read_calltree(call_tree, ep_call_tree, len(point_group), i)
            for node in call_tree.iterate_nodes():
                node: _ExtraProfInputNode
                progress_bar.update(0)

                child_durations = np.sum([c.duration for c in node.childs if not c.disable_exclusive_conversion],
                                         axis=0)

                if np.any(child_durations > node.duration):
                    logging.info("Extra-Prof overflow", node.path, (node.duration - child_durations) / 10 ** 9)
                    duration = node.duration / 10 ** 9
                    node.childs = []
                else:
                    duration = (node.duration - child_durations) / 10 ** 9

                experiment.add_measurement(
                    Measurement(coordinate, node.path, METRIC_TIME, duration))
                experiment.add_measurement(Measurement(coordinate, node.path, METRIC_VISITS, node.visits))
                if np.any(node.bytes != 0):
                    experiment.add_measurement(Measurement(coordinate, node.path, METRIC_BYTES, node.bytes))
                del node.duration
                del node.bytes
                del node.visits

        experiment.metrics = [METRIC_VISITS, METRIC_TIME, METRIC_BYTES]
        experiment.callpaths = [node.path for node in call_tree.iterate_nodes()]
        experiment.call_tree = call_tree
        io_helper.validate_experiment(experiment, progress_bar)
        return experiment

    @staticmethod
    def _assign_tags(node, n_type, flags):
        tags = node.path.tags
        if n_type == CallTreeNodeType.KERNEL:
            tags["gpu__kernel"] = True
            tags["agg__category"] = 'GPU'
            tags["agg__category__comparison_cpu_gpu"] = None
            node.disable_exclusive_conversion = True
        elif n_type == CallTreeNodeType.MEMCPY:
            if CallTreeNodeFlags.ASYNC in flags:
                tags['agg__category'] = 'GPU MEM'
                tags['gpu__blocking__mem_copy'] = False
                tags['agg__category__comparison_cpu_gpu'] = None
                node.disable_exclusive_conversion = True
            else:
                tags['gpu__blocking__mem_copy'] = True
        elif n_type == CallTreeNodeType.MEMSET:
            if CallTreeNodeFlags.ASYNC in flags:
                tags['agg__category'] = 'GPU'
                tags['gpu__blocking__mem_set'] = False
                tags['agg__category__comparison_cpu_gpu'] = None
                node.disable_exclusive_conversion = True
            else:
                tags['gpu__blocking__mem_set'] = True
        elif n_type == CallTreeNodeType.OVERHEAD:
            tags['agg__category'] = 'GPU OVERHEAD'
        elif n_type == CallTreeNodeType.SYNCHRONIZE:
            tags['agg__category__comparison_cpu_gpu'] = 'GPU SYNC'

    @staticmethod
    def _demangle_name(name):
        from itanium_demangler import parse as demangle
        try:
            demangled = demangle(name)
            if demangled:
                name = str(demangled)
        except NotImplementedError:
            pass
        return name.replace('->', '- >')

    @classmethod
    def _read_calltree(cls, call_tree, ep_root_node, num_values, i):
        def _read_calltree_node(parent_node: _ExtraProfInputNode, ep_node):
            name, childs, raw_type, raw_flags, m_duration, m_visits, m_bytes = ep_node
            n_type, flags = CallTreeNodeType(raw_type), CallTreeNodeFlags(raw_flags)
            if n_type == CallTreeNodeType.KERNEL_LAUNCH:
                name = "LAUNCH " + cls._demangle_name(name)
            elif n_type == CallTreeNodeType.KERNEL:
                name = "GPU " + cls._demangle_name(name)
            elif n_type == CallTreeNodeType.MEMSET or n_type == CallTreeNodeType.MEMCPY:
                name = "GPU " + name
            node = parent_node.find_child(name)
            if not node:
                node = _ExtraProfInputNode(name, parent_node.path.concat(name), num_values)
                if CallTreeNodeFlags.OVERLAP in flags:
                    node.path.tags['gpu__overlap'] = True
                    node.path.tags['validation__ignore__num_measurements'] = True
                    if n_type == CallTreeNodeType.OVERLAP:
                        node.path.tags['gpu__overlap'] = 'agg'
                        node.path.tags['agg__usage_disabled'] = True
                        node.path.tags['agg__disabled'] = True
                    else:
                        node.disable_exclusive_conversion = True
                else:
                    cls._assign_tags(node, n_type, flags)
                parent_node.add_child_node(node)
            node.duration[i] = m_duration
            node.bytes[i] = m_bytes
            node.visits[i] = m_visits
            for child in childs:
                _read_calltree_node(node, child)

        r_name, r_childs, r_type, r_flags, rm_duration, rm_visits, rm_bytes = ep_root_node
        assert r_name == ""
        assert rm_duration == 0
        assert rm_visits == 0
        assert rm_bytes == 0
        for r_child in r_childs:
            _read_calltree_node(call_tree, r_child)
