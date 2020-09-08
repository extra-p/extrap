import logging
import re
import warnings
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree, Node
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio import io_helper
from extrap.util.exceptions import FileFormatError
from extrap.util.progress_bar import DUMMY_PROGRESS
from pycubexr import CubexParser  # @UnresolvedImport
from pycubexr.utils.exceptions import MissingMetricError


def make_call_tree(cnodes):
    callpaths = []
    root = CallTree()

    def walk_tree(parent_cnode, parent):
        for cnode in parent_cnode.get_children():
            name = cnode.region.name
            path_name = '->'.join((parent.path.name, name))
            callpath = Callpath(path_name)
            callpaths.append(callpath)
            node = Node(name, callpath)
            parent.add_child_node(node)
            walk_tree(cnode, node)

    for root_cnode in cnodes:
        name = root_cnode.region.name
        callpath = Callpath(name)
        callpaths.append(callpath)
        node = Node(name, callpath)
        root.add_node(node)
        walk_tree(root_cnode, node)

    return root, callpaths


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
    cubex_files = list(path.glob('*/*.cubex'))
    if not cubex_files:
        raise FileFormatError(f'No cube files were found in: {dir_name}')
    pbar.total += 2 * len(cubex_files)
    # iterate over all folders and read the cube profiles in them
    experiment = Experiment()

    pbar.step("Reading cube files")
    parameter_names_initial = []
    parameter_names = []
    parameter_values = []
    parameter_dict = defaultdict(set)
    for path_id, path in enumerate(cubex_files):
        pbar.update()
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
            if scaling_type == "weak":
                experiment.set_scaling("weak")
            elif scaling_type == "strong":
                experiment.set_scaling("strong")

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

    pbar.step("Reading cube files")
    show_warning_no_strong_scaling = False
    show_warning_skipped_metrics = False
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

        aggregated_values = defaultdict(list)
        for path, _ in point_group:
            pbar.update()
            with CubexParser(str(path)) as parsed:
                callpaths = make_callpath_mapping(parsed.get_root_cnodes())
                # iterate over all metrics
                for cube_metric in parsed.get_metrics():
                    # NOTE: here we could choose which metrics to extract
                    if selected_metrics and cube_metric.name not in selected_metrics:
                        continue
                    try:
                        metric_values = parsed.get_metric_values(metric=cube_metric)
                        # create the metrics
                        metric = Metric(cube_metric.name)

                        for cnode_id in metric_values.cnode_indices:
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
                                # check number of parameters, if > 1 use weak scaling instead
                                # since sum values for strong scaling does not work for more than 1 parameter
                                if len(experiment.get_parameters()) > 1:
                                    aggregated_values[(callpath, metric)].extend(map(float, cnode_values))
                                    show_warning_no_strong_scaling = True
                                else:
                                    aggregated_values[(callpath, metric)].append(float(sum(cnode_values)))

                    # Take care of missing metrics
                    except MissingMetricError as e:  # @UnusedVariable
                        show_warning_skipped_metrics = True
                        logging.info(f'The cubex file does not contain data for the metric "{e.metric.name}"')

        # add measurements to experiment
        for (callpath, metric), values in aggregated_values.items():
            experiment.add_measurement(Measurement(coordinate, callpath, metric, values))

    to_delete = []
    # determine common callpaths for common calltree
    # add common callpaths and metrics to experiment
    for key, value in experiment.measurements.items():
        if len(value) < num_points:
            to_delete.append(key)
        else:
            (callpath, metric) = key
            experiment.add_callpath(callpath)
            experiment.add_metric(metric)
    for key in to_delete:
        del experiment.measurements[key]

    # determine calltree
    call_tree = io_helper.create_call_tree(experiment.callpaths, pbar)
    experiment.add_call_tree(call_tree)

    if show_warning_no_strong_scaling:
        warnings.warn("Strong scaling only works for one parameter. Using weak scaling instead.")
    if show_warning_skipped_metrics:
        warnings.warn("Some metrics were skipped because they contained no data. For details see log.")

    # io_helper.validate_experiment(experiment, pbar)
    return experiment
