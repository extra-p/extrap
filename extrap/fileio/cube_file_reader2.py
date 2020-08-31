import logging
import re
import warnings
from collections import defaultdict
from pathlib import Path

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree, Node
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio import io_helper
from extrap.util.exceptions import FileFormatError
from extrap.util.progress_bar import DUMMY_PROGRESS
from pycubexr import CubexParser  # @UnresolvedImport
from pycubexr.utils.exceptions import MissingMetricError  # @UnresolvedImport


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


def read_cube_file(dir_name, scaling_type, pbar=DUMMY_PROGRESS):
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

    parameter_selection_mask = []
    for i, p in enumerate(parameter_names):
        if len(parameter_dict[p]) > 1:
            experiment.add_parameter(Parameter(p))
            parameter_selection_mask.append(i)

    pbar.step("Reading cube files")
    show_warning_no_strong_scaling = False
    show_warning_skipped_metrics = False
    complete_data = {}
    # create a progress bar for reading the cube files
    for path_id, (path, parameter_value) in enumerate(zip(cubex_files, parameter_values)):
        pbar.update()
        with CubexParser(str(path)) as parsed:

            # create coordinate
            coordinate = Coordinate(parameter_value[i] for i in parameter_selection_mask)

            # check if the coordinate already exists
            if not experiment.coordinate_exists(coordinate):
                experiment.add_coordinate(coordinate)

            # get call tree
            if path_id == 0:
                call_tree, callpaths = make_call_tree(parsed.get_root_cnodes())
                # create the callpaths
                for c in callpaths:
                    experiment.add_callpath(c)

                # create the call tree and add it to the experiment

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    call_tree.print_tree()
                experiment.add_call_tree(call_tree)

            # make list with region ids
            # for metric in parsed.get_metrics():
            #     if metric.name == "time":
            #         metric_values = parsed.get_metric_values(metric=metric)
            #         for cnode_id in metric_values.cnode_indices:
            #             cnode = parsed.get_cnode(cnode_id)
            #             region = parsed.get_region(cnode)
            #             print(region)
            #         break

            # NOTE: here we could choose which metrics to extract
            # iterate over all metrics
            for cube_metric in parsed.get_metrics():
                try:
                    metric_values = parsed.get_metric_values(metric=cube_metric)
                    # create the metrics
                    metric = Metric(cube_metric.name)
                    experiment.add_metric(metric)

                    for cnode_id in metric_values.cnode_indices:
                        cnode = parsed.get_cnode(cnode_id)
                        callpath = callpaths[cnode_id]

                        # NOTE: here we can use clustering algorithm to select only certain node level values
                        # create the measurements
                        cnode_values = metric_values.cnode_values(cnode, convert_to_exclusive=True)

                        # in case of weak scaling calculate mean and median over all mpi process values
                        if scaling_type == "weak":
                            values = [float(v) for v in cnode_values]

                            # in case of strong scaling calculate the sum over all mpi process values
                        elif scaling_type == "strong":
                            # check number of parameters, if > 1 use weak scaling instead
                            # since sum values for strong scaling does not work for more than 1 parameter
                            if len(experiment.get_parameters()) > 1:
                                values = [float(v) for v in cnode_values]
                                show_warning_no_strong_scaling = True
                            else:
                                values = float(sum(cnode_values))

                        io_helper.append_to_repetition_dict(complete_data, (callpath, metric), coordinate, values)

                # Take care of missing metrics
                except MissingMetricError as e:  # @UnusedVariable
                    show_warning_skipped_metrics = True
                    logging.info(f'The cubex file does not contain data for the metric "{e.metric.name}"')

    io_helper.repetition_dict_to_experiment(complete_data, experiment, pbar)

    if show_warning_no_strong_scaling:
        warnings.warn("Strong scaling only works for one parameter. Using weak scaling instead.")
    if show_warning_skipped_metrics:
        warnings.warn("Some metrics were skipped because they contained no data. For details see log.")
    # take care of the repetitions of the measurements
    # experiment = compute_repetitions(experiment)
    io_helper.validate_experiment(experiment, pbar)
    return experiment
