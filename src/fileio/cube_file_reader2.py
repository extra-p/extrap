import logging
import re
import warnings
from collections import defaultdict
from pathlib import Path

from entities.callpath import Callpath
from entities.coordinate import Coordinate
from entities.experiment import Experiment
from entities.metric import Metric
from entities.parameter import Parameter
from fileio import io_helper
from fileio.io_helper import create_call_tree
from pycubexr import CubexParser  # @UnresolvedImport
from pycubexr.utils.exceptions import MissingMetricError  # @UnresolvedImport
from util.exceptions import FileFormatError
from util.progress_bar import DUMMY_PROGRESS


def construct_parent(calltree_elements, occurances, calltree_element_id):
    occurances_parent = occurances - 1
    calltree_element_id2 = calltree_element_id - 1
    calltree_element_id3 = -1
    while calltree_element_id2 >= 0:
        calltree_element = calltree_elements[calltree_element_id2]
        occurances_new = calltree_element.count("-")
        if occurances_parent == occurances_new:
            calltree_element_id3 = calltree_element_id2
            break
        calltree_element_id2 -= 1
    calltree_element_new = calltree_elements[calltree_element_id3]
    if calltree_element_new.count("-") == 0:
        return calltree_element_new
    else:
        occurances_new = calltree_element_new.count("-")
        calltree_element_parent = construct_parent(calltree_elements, occurances_new, calltree_element_id)
        calltree_element_new = calltree_element_new.replace("-", "")
        calltree_element_final = calltree_element_parent + "->" + calltree_element_new
        return calltree_element_final


def fix_call_tree(calltree):
    calltree_elements = calltree.split("\n")
    calltree_elements.remove("")
    calltree_elements_new = []

    for calltree_element_id in range(len(calltree_elements)):
        calltree_element = calltree_elements[calltree_element_id]
        if calltree_element.count("-") == 0:
            calltree_elements_new.append(calltree_element)
        elif calltree_element.count("-") > 0:
            occurances = calltree_element.count("-")
            calltree_element_new = construct_parent(calltree_elements, occurances, calltree_element_id)
            calltree_element_new = calltree_element_new + "->"
            calltree_element = calltree_element.replace("-", "")
            calltree_element_new = calltree_element_new + calltree_element
            calltree_elements_new.append(calltree_element_new)

    return calltree_elements_new


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
                call_tree = parsed.get_calltree()
                call_tree = fix_call_tree(call_tree)
                callpaths = [Callpath(node) for node in call_tree]
                # create the callpaths
                for c in callpaths:
                    experiment.add_callpath(c)

                # create the call tree and add it to the experiment
                call_tree = create_call_tree(callpaths)
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
    io_helper.validate_experiment(experiment)
    return experiment
