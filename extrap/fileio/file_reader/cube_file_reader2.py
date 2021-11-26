# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging
import warnings
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Dict, Union, Sequence, Tuple

from pycubexr import CubexParser
from pycubexr.utils.exceptions import MissingMetricError

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio import io_helper
from extrap.fileio.file_reader.abstract_directory_reader import AbstractDirectoryReader
from extrap.util.progress_bar import DUMMY_PROGRESS, ProgressBar

CUBE_CALLSITE_ID_KEY = 'callsite id'


class CubeFileReader2(AbstractDirectoryReader):
    NAME = "cube"
    GUI_ACTION = "Open set of &CUBE files"
    DESCRIPTION = "Load a set of CUBE files and generate a new experiment"
    CMD_ARGUMENT = "--cube"
    LOADS_FROM_DIRECTORY = True

    scaling_type = "weak"
    selected_metrics = None
    demangle_names = True
    use_inclusive_measurements = False

    def read_experiment(self, path: Union[Path, str], progress_bar: ProgressBar = DUMMY_PROGRESS) -> Experiment:
        cubex_files = self._find_files_in_directory(path, '*/[!.]*.cubex', progress_bar)
        # iterate over all folders and read the cube profiles in them
        experiment = Experiment()
        # set scaling flag for experiment
        if self.scaling_type == "weak" or self.scaling_type == "strong":
            experiment.scaling = self.scaling_type

        progress_bar.step("Reading cube files")
        parameter_names, parameter_values = self._determine_parameters_from_paths(cubex_files, progress_bar)

        # determine non-constant parameters and add them to experiment
        for p in parameter_names:
            experiment.add_parameter(Parameter(p))

        # check number of parameters, if > 1 use weak scaling instead
        # since sum values for strong scaling does not work for more than 1 parameter
        if self.scaling_type == 'strong' and len(experiment.parameters) > 1:
            warnings.warn("Strong scaling only works for one parameter. Using weak scaling instead.")
            scaling_type = 'weak'
            experiment.scaling = scaling_type

        progress_bar.step("Reading cube files")

        show_warning_skipped_metrics = set()

        # import data from cube files
        # optimize import memory usage by reordering files and grouping by coordinate
        num_points = 0
        reordered_files = sorted(zip(cubex_files, parameter_values), key=itemgetter(1))
        for parameter_value, point_group in groupby(reordered_files, key=itemgetter(1)):
            num_points += 1
            # create coordinate
            coordinate = Coordinate(parameter_value)
            experiment.add_coordinate(coordinate)

            aggregated_values = self._aggregate_repetitions(point_group, progress_bar, show_warning_skipped_metrics)

            # add measurements to experiment
            for (callpath, metric), values in aggregated_values.items():
                progress_bar.update(0)
                experiment.add_measurement(Measurement(coordinate, callpath, metric, values))

        progress_bar.step("Unify calltrees")
        callpaths_to_merge = self._determine_and_add_common_callpaths(experiment, num_points, progress_bar)
        if self.use_inclusive_measurements:
            self._delete_callpaths_from_experiment(experiment, callpaths_to_merge, progress_bar)
        else:
            self._merge_callpaths_into_existing_calltree(experiment, callpaths_to_merge, progress_bar)

        # determine calltree
        call_tree = io_helper.create_call_tree(experiment.callpaths, progress_bar, progress_scale=0.1)
        experiment.call_tree = call_tree

        if show_warning_skipped_metrics:
            warnings.warn("The following metrics were skipped because they contained no data: "
                          f"{', '.join(show_warning_skipped_metrics)}. For more details see log.")

        io_helper.validate_experiment(experiment, progress_bar)
        progress_bar.update()
        return experiment

    def _aggregate_repetitions(self, point_group, progress_bar, show_warning_skipped_metrics):
        aggregated_values = defaultdict(list)
        for path, _ in point_group:
            progress_bar.update()
            with CubexParser(str(path)) as parsed:
                callpaths = self._make_callpath_mapping(parsed.get_root_cnodes())
                # iterate over all metrics
                for cube_metric in parsed.get_metrics():
                    progress_bar.update(0)
                    # NOTE: here we could choose which metrics to extract
                    if self.selected_metrics and cube_metric.name not in self.selected_metrics:
                        continue
                    try:
                        metric_values = parsed.get_metric_values(metric=cube_metric, cache=False)
                        # create the metrics
                        metric = Metric(cube_metric.name)

                        for cnode_id in metric_values.cnode_indices:
                            progress_bar.update(0)
                            cnode = parsed.get_cnode(cnode_id)
                            callpath = callpaths[cnode_id]
                            # NOTE: here we can use clustering algorithm to select only certain node level values
                            # create the measurements
                            cnode_values = metric_values.cnode_values(cnode,
                                                                      convert_to_exclusive=not self.use_inclusive_measurements,
                                                                      convert_to_inclusive=self.use_inclusive_measurements)

                            # in case of weak scaling calculate mean and median over all mpi process values
                            if self.scaling_type == "weak":
                                # do NOT use generator it is slower
                                aggregated_values[(callpath, metric)].extend(map(float, cnode_values))

                                # in case of strong scaling calculate the sum over all mpi process values
                            elif self.scaling_type == "strong":
                                aggregated_values[(callpath, metric)].append(float(sum(cnode_values)))

                    # Take care of missing metrics
                    except MissingMetricError as e:  # @UnusedVariable
                        show_warning_skipped_metrics.add(e.metric.name)
                        logging.info(
                            f'The cubex file {Path(*path.parts[-2:])} does not contain data for the metric "{e.metric.name}"')
        return aggregated_values

    def _make_callpath_mapping(self, cnodes):
        callpaths = {}
        callsites = {}
        callsite_kernels = {}

        def demangle_name(name):
            if self.demangle_names:
                from itanium_demangler import parse as demangle
                try:
                    demangled = demangle(name)
                    if demangled:
                        name = str(demangled)
                except NotImplementedError as e:
                    pass
            return name.replace('->', '- >')

        def walk_tree(parent_cnode, parent_name):
            for cnode in parent_cnode.get_children():
                name = cnode.region.name
                name = demangle_name(name)
                path_name = '->'.join((parent_name, name))
                callpath = Callpath(path_name)
                if cnode.region.paradigm == 'cuda':
                    if cnode.region.role == 'function':
                        callpath.tags['gpu__kernel'] = True
                        callsite_kernels[cnode.parameters[CUBE_CALLSITE_ID_KEY]] = callpath
                    elif cnode.region.role == 'wrapper':
                        if CUBE_CALLSITE_ID_KEY in cnode.parameters:
                            callsites[cnode.parameters[CUBE_CALLSITE_ID_KEY]] = callpath
                    elif cnode.region.role == 'implicit barrier':
                        callpath.tags['gpu__sync'] = True
                        if CUBE_CALLSITE_ID_KEY in cnode.parameters:
                            callsites[cnode.parameters[CUBE_CALLSITE_ID_KEY]] = callpath
                    elif cnode.region.role == 'artificial':
                        callpath.tags['gpu__overhead'] = True
                        if CUBE_CALLSITE_ID_KEY in cnode.parameters:
                            callsites[cnode.parameters[CUBE_CALLSITE_ID_KEY]] = callpath
                    else:
                        warnings.warn(f"Unknown cuda role {cnode.region.role} in {path_name}.")

                callpaths[cnode.id] = callpath
                walk_tree(cnode, path_name)

        for root_cnode in cnodes:
            name = root_cnode.region.name
            name = demangle_name(name)
            callpath = Callpath(name)
            callpaths[root_cnode.id] = callpath
            walk_tree(root_cnode, name)

        for id, kernel_callpath in callsite_kernels.items():
            if id in callsites:
                callsite_callpath = callsites[id]
                kernel_callpath.name = callsite_callpath.name + '->[GPU]'
            else:
                warnings.warn(f"Could not find call-site ({id}) for the following kernel: {kernel_callpath.name}")

        return callpaths

    @staticmethod
    def _merge_callpaths_into_existing_calltree(experiment: Experiment,
                                                callpaths_to_merge: Sequence[Tuple[Callpath, Metric]],
                                                progress_bar=DUMMY_PROGRESS):
        for key in callpaths_to_merge:
            (callpath, metric) = key
            new_callpath: Callpath = callpath
            new_key = key
            # find parent call-path
            while new_key not in experiment.measurements and '->' in new_callpath.name:
                new_callpath = Callpath(str(new_callpath).rsplit(sep='->', maxsplit=1)[0])
                new_key = (new_callpath, metric)
            # merge parent measurements with the current measurements
            if new_key in experiment.measurements:
                measurements: Dict[Coordinate, Measurement] = {m.coordinate: m for m in
                                                               experiment.measurements[new_key]}
                for m in experiment.measurements[key]:
                    new_m = measurements.get(m.coordinate)
                    if new_m:
                        new_m.merge(m)
                    else:
                        m.callpath = experiment.measurements[new_key][0].callpath
                        experiment.measurements[new_key].append(m)
            else:
                warnings.warn("Some call paths could not be integrated into the common call tree.")
            progress_bar.update(0.1)
            # delete current measurements
            del experiment.measurements[key]

    @staticmethod
    def _determine_and_add_common_callpaths(experiment: Experiment, num_points: int,
                                            progress_bar=DUMMY_PROGRESS) -> Sequence[Tuple[Callpath, Metric]]:
        """
        Determines common callpaths for a common calltree and adds common callpaths and metrics to experiment.
        :param num_points: The number of points/coordinates the input set has.
        :return: Callpaths that are not present in the common calltree.
        """
        callpaths_to_merge = []
        for key, value in progress_bar(experiment.measurements.items(), len(experiment.measurements), scale=0.1):
            if len(value) < num_points:
                callpaths_to_merge.append(key)
                progress_bar.total += 0.1
            else:
                (callpath, metric) = key
                experiment.add_callpath(callpath)
                experiment.add_metric(metric)
        return callpaths_to_merge

    @staticmethod
    def _delete_callpaths_from_experiment(experiment, callpaths_to_merge, progress_bar):
        for key in callpaths_to_merge:
            progress_bar.update(0.1)
            # delete current measurements
            del experiment.measurements[key]

    def read_cube_file(self, dir_name, scaling_type, pbar=DUMMY_PROGRESS, selected_metrics=None):
        self.scaling_type = scaling_type
        self.selected_metrics = selected_metrics
        return self.read_experiment(dir_name, pbar)
