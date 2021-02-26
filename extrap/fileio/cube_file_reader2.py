# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging
import re
import warnings
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Dict

from pycubexr import CubexParser
from pycubexr.utils.exceptions import MissingMetricError

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio import io_helper
from extrap.util.exceptions import FileFormatError
from extrap.util.progress_bar import DUMMY_PROGRESS


def make_callpath_mapping(cnodes):
    callpaths = {}

    def walk_tree(parent_cnode, parent_name):
        for cnode in parent_cnode.get_children():
            name = cnode.region.name
            path_name = '->'.join((parent_name, name))
            callpaths[cnode.id] = Callpath(path_name)
            walk_tree(cnode, path_name)

    for root_cnode in cnodes:
        name = root_cnode.region.name
        callpath = Callpath(name)
        callpaths[root_cnode.id] = callpath
        walk_tree(root_cnode, name)

    return callpaths


def read_cube_file(dir_name, scaling_type, pbar=DUMMY_PROGRESS, selected_metrics=None):
    # read the paths of the cube files in the given directory with dir_name
    path = Path(dir_name)
    if not path.is_dir():
        raise FileFormatError(f'Cube file path must point to a directory: {dir_name}')
    cubex_files = list(path.glob('*/[!.]*.cubex'))
    if not cubex_files:
        raise FileFormatError(f'No cube files were found in: {dir_name}')
    pbar.total += len(cubex_files) + 6
    # iterate over all folders and read the cube profiles in them
    experiment = Experiment()

    pbar.step("Reading cube files")
    parameter_names_initial = []
    parameter_names = []
    parameter_values = []
    parameter_dict = defaultdict(set)
    progress_step_size = 5 / len(cubex_files)
    for path_id, path in enumerate(cubex_files):
        pbar.update(progress_step_size)
        folder_name = path.parent.name
        logging.debug(f"Cube file: {path} Folder: {folder_name}")

        # create the parameters
        par_start = folder_name.find(".") + 1
        par_end = folder_name.find(".r")
        par_end = None if par_end == -1 else par_end
        parameters = folder_name[par_start:par_end]
        # parameters = folder_name.split(".")

        # set scaling flag for experiment
        if path_id == 0:
            if scaling_type == "weak" or scaling_type == "strong":
                experiment.scaling = scaling_type

        param_list = re.split('([0-9.,]+)', parameters)
        param_list.remove("")

        parameter_names = [n for i, n in enumerate(param_list) if i % 2 == 0]
        parameter_value = [float(n.replace(',', '.').rstrip('.')) for i, n in enumerate(param_list) if i % 2 == 1]

        # check if parameter already exists
        if path_id == 0:
            parameter_names_initial = parameter_names
        elif parameter_names != parameter_names_initial:
            raise FileFormatError(
                f"Parameters must be the same and in the same order: {parameter_names} is not {parameter_names_initial}.")

        for n, v in zip(parameter_names, parameter_value):
            parameter_dict[n].add(v)
        parameter_values.append(parameter_value)

    # determine non-constant parameters and add them to experiment
    parameter_selection_mask = []
    for i, p in enumerate(parameter_names):
        if len(parameter_dict[p]) > 1:
            experiment.add_parameter(Parameter(p))
            parameter_selection_mask.append(i)

    # check number of parameters, if > 1 use weak scaling instead
    # since sum values for strong scaling does not work for more than 1 parameter
    if scaling_type == 'strong' and len(experiment.parameters) > 1:
        warnings.warn("Strong scaling only works for one parameter. Using weak scaling instead.")
        scaling_type = 'weak'
        experiment.scaling = scaling_type

    pbar.step("Reading cube files")

    show_warning_skipped_metrics = set()
    aggregated_values = defaultdict(list)

    # import data from cube files
    # optimize import memory usage by reordering files and grouping by coordinate
    num_points = 0
    reordered_files = sorted(zip(cubex_files, parameter_values), key=itemgetter(1))
    for parameter_value, point_group in groupby(reordered_files, key=itemgetter(1)):
        num_points += 1
        # create coordinate
        coordinate = Coordinate(parameter_value[i] for i in parameter_selection_mask)
        experiment.add_coordinate(coordinate)

        aggregated_values.clear()
        for path, _ in point_group:
            pbar.update()
            with CubexParser(str(path)) as parsed:
                callpaths = make_callpath_mapping(parsed.get_root_cnodes())
                # iterate over all metrics
                for cube_metric in parsed.get_metrics():
                    pbar.update(0)
                    # NOTE: here we could choose which metrics to extract
                    if selected_metrics and cube_metric.name not in selected_metrics:
                        continue
                    try:
                        metric_values = parsed.get_metric_values(metric=cube_metric, cache=False)
                        # create the metrics
                        metric = Metric(cube_metric.name)

                        for cnode_id in metric_values.cnode_indices:
                            pbar.update(0)
                            cnode = parsed.get_cnode(cnode_id)
                            callpath = callpaths[cnode_id]
                            # NOTE: here we can use clustering algorithm to select only certain node level values
                            # create the measurements
                            cnode_values = metric_values.cnode_values(cnode, convert_to_exclusive=True)

                            # in case of weak scaling calculate mean and median over all mpi process values
                            if scaling_type == "weak":
                                # do NOT use generator it is slower
                                aggregated_values[(callpath, metric)].extend(map(float, cnode_values))

                                # in case of strong scaling calculate the sum over all mpi process values
                            elif scaling_type == "strong":
                                aggregated_values[(callpath, metric)].append(float(sum(cnode_values)))

                    # Take care of missing metrics
                    except MissingMetricError as e:  # @UnusedVariable
                        show_warning_skipped_metrics.add(e.metric.name)
                        logging.info(
                            f'The cubex file {Path(*path.parts[-2:])} does not contain data for the metric "{e.metric.name}"')

        # add measurements to experiment
        for (callpath, metric), values in aggregated_values.items():
            pbar.update(0)
            experiment.add_measurement(Measurement(coordinate, callpath, metric, values))

    pbar.step("Unify calltrees")
    callpaths_to_merge = []
    # determine common callpaths for common calltree
    # add common callpaths and metrics to experiment
    for key, value in pbar(experiment.measurements.items(), len(experiment.measurements), scale=0.1):
        if len(value) < num_points:
            callpaths_to_merge.append(key)
            pbar.total += 0.1
        else:
            (callpath, metric) = key
            experiment.add_callpath(callpath)
            experiment.add_metric(metric)
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
            measurements: Dict[Coordinate, Measurement] = {m.coordinate: m for m in experiment.measurements[new_key]}
            for m in experiment.measurements[key]:
                new_m = measurements.get(m.coordinate)
                if new_m:
                    new_m.merge(m)
                else:
                    m.callpath = experiment.measurements[new_key][0].callpath
                    experiment.measurements[new_key].append(m)
        else:
            warnings.warn("Some call paths could not be integrated into the common call tree.")
        pbar.update(0.1)
        # delete current measurements
        del experiment.measurements[key]

    # determine calltree
    call_tree = io_helper.create_call_tree(experiment.callpaths, pbar, progress_scale=0.1)
    experiment.call_tree = call_tree

    if show_warning_skipped_metrics:
        warnings.warn("The following metrics were skipped because they contained no data: "
                      f"{', '.join(show_warning_skipped_metrics)}. For more details see log.")

    io_helper.validate_experiment(experiment, pbar)
    pbar.update()
    return experiment
