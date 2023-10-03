# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree
from extrap.entities.calltree import Node
from extrap.entities.measurement import Measurement
from extrap.util.exceptions import InvalidExperimentError
from extrap.util.progress_bar import DUMMY_PROGRESS

if TYPE_CHECKING:
    from extrap.entities.experiment import Experiment


def format_callpaths(experiment):
    """
    This method formats the output so that only the callpaths are shown.
    """
    callpaths = experiment.callpaths
    text = ""
    for callpath_id in range(len(callpaths)):
        callpath = callpaths[callpath_id]
        callpath_string = callpath.name
        text += callpath_string + "\n"
    return text


def format_metrics(experiment):
    """
    This method formats the output so that only the metrics are shown.
    """
    metrics = experiment.metrics
    text = ""
    for metric_id in range(len(metrics)):
        metric = metrics[metric_id]
        metric_string = metric.name
        text += metric_string + "\n"
    return text


def format_parameters(experiment):
    """
    This method formats the output so that only the parameters are shown.
    """
    parameters = experiment.parameters
    text = ""
    for parameters_id in range(len(parameters)):
        parameter = parameters[parameters_id]
        parameter_string = parameter.name
        text += parameter_string + "\n"
    return text


def format_functions(experiment):
    """
    This method formats the output so that only the functions are shown.
    """
    modeler = experiment.modelers[0]
    models = modeler.models
    text = ""
    for model in models.values():
        hypothesis = model.hypothesis
        function = hypothesis.function
        function_string = function.to_string(*experiment.parameters)
        text += function_string + "\n"
    return text


def format_latex_functions(experiment):
    """
    This method formats the output so that only the functions are shown.
    """
    modeler = experiment.modelers[0]
    models = modeler.models
    text = ""
    for model in models.values():
        hypothesis = model.hypothesis
        function = hypothesis.function
        function_string = function.to_latex_string(*experiment.parameters)
        text += function_string + "\n"
    return text


def format_all(experiment):
    """
    This method formats the output so that all information is shown.
    """
    coordinates = experiment.coordinates
    callpaths = experiment.callpaths
    metrics = experiment.metrics
    modeler = experiment.modelers[0]
    text = ""
    for callpath_id in range(len(callpaths)):
        callpath = callpaths[callpath_id]
        callpath_string = callpath.name
        text += "Callpath: " + callpath_string + "\n"
        for metric_id in range(len(metrics)):
            metric = metrics[metric_id]
            metric_string = metric.name
            text += "\tMetric: " + metric_string + "\n"
            for coordinate_id in range(len(coordinates)):
                coordinate = coordinates[coordinate_id]
                dimensions = coordinate.dimensions
                coordinate_text = "Measurement point: ("
                for dimension in range(dimensions):
                    value = coordinate[dimension]
                    value_string = "{:.2E}".format(value)
                    coordinate_text += value_string + ","
                coordinate_text = coordinate_text[:-1]
                coordinate_text += ")"
                measurement = experiment.get_measurement(coordinate_id, callpath_id, metric_id)
                if measurement == None:
                    value_mean = 0
                    value_median = 0
                else:
                    value_mean = measurement.mean
                    value_median = measurement.median
                text += f"\t\t{coordinate_text} Mean: {value_mean:.2E} Median: {value_median:.2E}\n"
            try:
                model = modeler.models[callpath, metric]
            except KeyError as e:
                model = None
            if model != None:
                hypothesis = model.hypothesis
                function = hypothesis.function
                rss = hypothesis.RSS
                ar2 = hypothesis.AR2
                function_string = function.to_string(*experiment.parameters)
            else:
                rss = 0
                ar2 = 0
                function_string = "None"
            text += "\t\tModel: " + function_string + "\n"
            text += "\t\tRSS: {:.2E}\n".format(rss)
            text += "\t\tAdjusted R^2: {:.2E}\n".format(ar2)
    return text


def format_output(experiment, printtype):
    """
    This method formats the output of the modeler to a string that can be printed in the console
    or to a file. Depending on the given options only parts of the modelers output get printed.
    """
    if printtype == "ALL":
        text = format_all(experiment)
    elif printtype == "CALLPATHS":
        text = format_callpaths(experiment)
    elif printtype == "METRICS":
        text = format_metrics(experiment)
    elif printtype == "PARAMETERS":
        text = format_parameters(experiment)
    elif printtype == "FUNCTIONS":
        text = format_functions(experiment)
    elif printtype == "LATEX-FUNCTIONS":
        text = format_latex_functions(experiment)
    else:
        raise ValueError('printtype does not exist')
    return text


def save_output(text, path):
    """
    This method saves the output of the modeler, i.e. it's results to a text file at the given path.
    """
    with open(path, "w+") as out:
        out.write(text)


def append_to_repetition_dict(complete_data, key, coordinate, value, progress_bar=DUMMY_PROGRESS):
    if isinstance(value, list):
        if key in complete_data:
            if coordinate in complete_data[key]:
                complete_data[key][coordinate].extend(value)
            else:
                complete_data[key][coordinate] = value
                progress_bar.total += 1
        else:
            complete_data[key] = {
                coordinate: value
            }
            progress_bar.total += 1
    else:
        if key in complete_data:
            if coordinate in complete_data[key]:
                complete_data[key][coordinate].append(value)
            else:
                complete_data[key][coordinate] = [value]
                progress_bar.total += 1
        else:
            complete_data[key] = {
                coordinate: [value]
            }
            progress_bar.total += 1


def repetition_dict_to_experiment(complete_data, experiment, progress_bar=DUMMY_PROGRESS):
    progress_bar.step('Creating experiment')
    for mi, key in enumerate(complete_data):
        progress_bar.update()
        callpath, metric = key
        measurementset = complete_data[key]
        experiment.add_callpath(callpath)
        experiment.add_metric(metric)
        for coordinate in measurementset:
            values = measurementset[coordinate]
            experiment.add_coordinate(coordinate)
            experiment.add_measurement(Measurement(coordinate, callpath, metric, values))


def create_call_tree(callpaths: List[Callpath], progress_bar=DUMMY_PROGRESS, progress_total_added=False,
                     progress_scale=1):
    """
    This method creates the call tree object from the callpaths read.
    It builds a structure with a root node and child nodes.
    It can be used to display the callpaths in a tree structure.
    However, this method only works if the read callpaths are in
    the correct order, as they would appear in the real program.
    """
    tree = CallTree()
    progress_bar.step('Creating calltree')
    # create a two dimensional array of the callpath elements as strings
    callpaths2 = []
    max_length = 0

    if not progress_total_added:
        progress_bar.total += len(callpaths) * progress_scale

    for splitted_callpath in callpaths:
        callpath_string = splitted_callpath.name
        elems = callpath_string.split("->")
        callpaths2.append(elems)
        progress_bar.total += len(elems) * progress_scale
        progress_bar.update(progress_scale)
        if len(elems) > max_length:
            max_length = len(elems)

    # iterate over the elements of one call path
    for i in range(max_length):
        # iterate over all callpaths
        for callpath, splitted_callpath in zip(callpaths, callpaths2):
            # check that we do not try to access an element that does not exist
            if i >= len(splitted_callpath):
                continue
            # if the element does exist
            progress_bar.update(progress_scale)
            callpath_string = splitted_callpath[i]

            # when at root level
            if i == 0:
                root_node = tree
            # when not at root level, the previous nodes of the elements have to be checked
            else:
                # find the root node of the element that we want to add currently
                root_node = find_root_node(splitted_callpath, tree, i)

            # check if that child node is already existing
            child_node = root_node.find_child(callpath_string)
            is_leaf = i == len(splitted_callpath) - 1
            if child_node:
                if is_leaf:
                    if child_node.path == Callpath.EMPTY:
                        child_node.path = callpath
                    else:
                        warnings.warn("Duplicate callpath encountered, only first occurence is retained.")

            else:
                # add a new child node to the root node
                if is_leaf:
                    child_node = Node(callpath_string, callpath)
                else:
                    child_node = Node(callpath_string, Callpath.EMPTY)
                root_node.add_child_node(child_node)

    return tree


def find_root_node(callpath_elements, tree, loop_id):
    """
    This method finds the root node of a element in the callpath tree.
    Therefore, it searches iterativels through the tree.
    """
    level = 0
    root_element_string = callpath_elements[level]
    root_node = tree.get_node(root_element_string)

    # root node already found
    if loop_id == level + 1:
        return root_node

    # need to search deeper in the tree for the root node
    else:
        return find_child_node(root_node, level, callpath_elements, loop_id)


def find_child_node(root_node, level, callpath_elements, loop_id):
    """
    This method searches for a child node in the tree. Searches iteratively
    into the three and each nodes child nodes. Returns the root node of the
    child.
    """
    level = level + 1
    root_element_string = callpath_elements[level]
    childs = root_node.childs

    for i in range(len(childs)):
        child_name = childs[i].name

        if child_name == root_element_string:
            new_root_node = childs[i]

            # root node already found
            if loop_id == level + 1:
                return new_root_node

            # need to search deeper in the tree for the root node
            else:
                return find_child_node(new_root_node, level, callpath_elements, loop_id)


def validate_experiment(experiment: Experiment, progress_bar=DUMMY_PROGRESS):
    def require(cond, message):
        if not cond:
            raise InvalidExperimentError(message)

    progress_bar.step('Validating experiment')

    length_parameters = len(experiment.parameters)
    require(length_parameters > 0, "Parameters are missing.")
    length_coordinates = len(experiment.coordinates)
    require(length_coordinates > 0, "Coordinates are missing.")
    require(len(experiment.metrics) > 0, "Metrics are missing.")
    require(len(experiment.callpaths) > 0, "Callpaths are missing.")
    require(len(experiment.call_tree.childs) > 0, "Calltree is missing.")
    for c in experiment.coordinates:
        require(len(c) == length_parameters,
                f'The number of coordinate units of {c} does not match the number of '
                f'parameters ({length_parameters}).')

    for k, m in progress_bar(experiment.measurements.items(), len(experiment.measurements)):
        require(len(m) == length_coordinates,
                f'The number of measurements ({len(m)}) for {k} does not match the number of coordinates '
                f'({length_coordinates}).')
