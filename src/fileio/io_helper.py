"""
This file is part of the Extra-P software (https://github.com/extra-p/extrap)

Copyright (c) 2020 Technical University of Darmstadt, Darmstadt, Germany

All rights reserved.
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""


from entities.measurement import Measurement
from entities.calltree import CallTree
from entities.calltree import Node
import logging
from util.deprecation import deprecated
from entities.callpath import Callpath


def format_callpaths(experiment):
    """
    This method formats the ouput so that only the callpaths are shown.
    """
    callpaths = experiment.get_callpaths()
    text = ""
    for callpath_id in range(len(callpaths)):
        callpath = callpaths[callpath_id]
        callpath_string = callpath.get_name()
        text += callpath_string + "\n"
    return text


def format_metrics(experiment):
    """
    This method formats the ouput so that only the metrics are shown.
    """
    metrics = experiment.get_metrics()
    text = ""
    for metric_id in range(len(metrics)):
        metric = metrics[metric_id]
        metric_string = metric.get_name()
        text += metric_string + "\n"
    return text


def format_parameters(experiment):
    """
    This method formats the ouput so that only the parameters are shown.
    """
    parameters = experiment.get_parameters()
    text = ""
    for parameters_id in range(len(parameters)):
        parameter = parameters[parameters_id]
        parameter_string = parameter.get_name()
        text += parameter_string + "\n"
    return text


def format_functions(experiment):
    """
    This method formats the ouput so that only the functions are shown.
    """
    modeler = experiment.get_modeler(0)
    models = modeler.get_models()
    text = ""
    for model_id in range(len(models)):
        model = models[model_id]
        hypothesis = model.get_hypothesis()
        function = hypothesis.get_function()
        if len(experiment.get_parameters()) == 1:
            # set exact = True to get exact function printout
            function_string = function.to_string(experiment.get_parameter(0), True)
        else:
            # set exact = True to get exact function printout
            function_string = function.to_string(True)
        text += function_string + "\n"
    return text


def format_all(experiment):
    """
    This method formats the ouput so that all information is shown.
    """
    coordinates = experiment.get_coordinates()
    callpaths = experiment.get_callpaths()
    metrics = experiment.get_metrics()
    modeler = experiment.get_modeler(0)
    text = ""
    for callpath_id in range(len(callpaths)):
        callpath = callpaths[callpath_id]
        callpath_string = callpath.get_name()
        text += "Callpath: " + callpath_string + "\n"
        for metric_id in range(len(metrics)):
            metric = metrics[metric_id]
            metric_string = metric.get_name()
            text += "\tMetric: " + metric_string + "\n"
            for coordinate_id in range(len(coordinates)):
                coordinate = coordinates[coordinate_id]
                coordinate_text = "Measurement point: "
                coordinate_text += str(coordinate)
                measurement = experiment.get_measurement(coordinate_id, callpath_id, metric_id)
                value_mean = measurement.get_value_mean()
                value_mean_string = "{:.2E}".format(value_mean)
                value_median = measurement.get_value_median()
                value_median_string = "{:.2E}".format(value_median)
                text += "\t\t" + coordinate_text + " Mean: " + value_mean_string + " Median: " + value_median_string + "\n"
            model = modeler.get_model(callpath_id, metric_id)
            hypothesis = model.get_hypothesis()
            function = hypothesis.get_function()
            rss = hypothesis.get_RSS()
            ar2 = hypothesis.get_AR2()
            if len(experiment.get_parameters()) == 1:
                function_string = function.to_string(experiment.get_parameter(0))
            else:
                function_string = function.to_string()
            text += "\t\tModel: " + function_string + "\n"
            text += "\t\tRSS: {:.2E}\n".format(rss)
            text += "\t\tAdjusted R^2: {:.2E}\n".format(ar2)
    return text


def format_output(experiment, printtype):
    """
    This method formats the ouput of the modeler to a string that can be printed in the console
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
    return text


def save_output(text, path):
    """
    This method saves the output of the modeler, i.e. it's results to a text file at the given path.
    """
    with open(path, "w+") as out:
        out.write(text)


@deprecated("Handle accumulation as part of file reader")
def compute_repetitions(experiment, progress_event=lambda x: ()):
    """
    This method takes an experiment and computes the mean and median values of the measurements
    over all repetitions per coordinate. The return is the experiment without any measurement
    repetitions, just one mean and one median value per coordinate.  
    """

    logging.info("Computing measurement repetitions...")
    # TODO: should be only active when using the command line tool
    # create a progress bar for computing the repetitions
    progress_event(0)

    # create a container for computing the mean or median values of the measurements for each coordinate
    computed_measurements = []
    for coordinate_id in range(experiment.get_len_coordinates()):
        for metric_id in range(experiment.get_len_metrics()):
            for callpath_id in range(experiment.get_len_callpaths()):
                measurement = Measurement(coordinate_id, callpath_id, metric_id, [], None)
                computed_measurements.append(measurement)

    # update progress bar
    progress_event(20)

    # iterate over all previously read measurements
    measurements = experiment.get_measurements()
    for measurement_id in range(len(measurements)):
        measurement = measurements[measurement_id]
        coordinate_id = measurement.get_coordinate_id()
        callpath_id = measurement.get_callpath_id()
        metric_id = measurement.get_metric_id()
        value = measurement.get_value_mean()
        computed_measurement_id = -1

        # search the coordinate, metric, callpath that fits to the measurement and remember the id
        for computed_measurements_list_id in range(len(computed_measurements)):
            computed_measurement = computed_measurements[computed_measurements_list_id]
            computed_coordinate_id = computed_measurement.get_coordinate_id()
            computed_callpath_id = computed_measurement.get_callpath_id()
            computed_metric_id = computed_measurement.get_metric_id()
            if computed_coordinate_id == coordinate_id and computed_callpath_id == callpath_id and computed_metric_id == metric_id:
                computed_measurement_id = computed_measurements_list_id
                break

        # add the value of the measurement to the container object and the list inside (one list per coordinate*callpath*metric)
        # the field value_mean serves as a temporary storage for the real measured value, before the median and mean of the repetitions are computed
        # after theses value have been computed, they are written to the measurement object and the original measured value is overwritten
        computed_measurements[computed_measurement_id].value_mean.append(value)

    # update progress bar
    progress_event(20)

    # calculate median values of measurements
    for measurement_id in range(len(computed_measurements)):
        computed_measurement = computed_measurements[measurement_id]
        values = computed_measurement.get_value_mean()

        # if there exists at least one measurement for this coordinate, metric, callpath calculate the value
        if len(values) != 0:
            sorted_values = sorted(values)

            # even number of elements
            if len(sorted_values) % 2 == 0:
                middle_id_1 = int(len(sorted_values) / 2) - 1
                middle_id_2 = middle_id_1 + 1
                median_value = (sorted_values[middle_id_1] + sorted_values[middle_id_2]) / 2

            # uneven number of elements
            else:
                middle_id = int((len(sorted_values) + 1) / 2) - 1
                median_value = sorted_values[middle_id]

        # if not set value to empty value
        else:
            median_value = None

        computed_measurement.set_value_median(median_value)
        computed_measurements[measurement_id] = computed_measurement

    # update progress bar
    progress_event(20)

    # calculate mean values of measurements
    for measurement_id in range(len(computed_measurements)):
        computed_measurement = computed_measurements[measurement_id]
        values = computed_measurement.get_value_mean()

        # if there exists at least one measurement for this coordinate, metric, callpath calculate the value
        if len(values) != 0:
            mean_value = sum(values) / len(values)

        # if not set value to empty value
        else:
            mean_value = None

        computed_measurement.set_value_mean(mean_value)
        computed_measurements[measurement_id] = computed_measurement

    # remove the old measurement objects from the experiment
    experiment.clear_measurements()

    # update progress bar
    progress_event(20)

    # add the new measurement objects to the experiment with the computed mean and median values
    for measurement_id in range(len(computed_measurements)):
        measurement = computed_measurements[measurement_id]

        # ignore a coordinate, metric, callpath if no measurement are available for it
        if measurement.get_value_mean() != None and measurement.get_value_median() != None:
            experiment.add_measurement(measurement)
            metric_id = measurement.get_metric_id()
            callpath_id = measurement.get_callpath_id()
            coordinate_id = measurement.get_coordinate_id()
            value_mean = measurement.get_value_mean()
            value_median = measurement.get_value_median()
            logging.debug("Measurement: "+experiment.get_metric(metric_id).get_name()+", "+experiment.get_callpath(callpath_id).get_name()+", " +
                          experiment.get_coordinate(coordinate_id).get_as_string()+": "+str(value_mean)+" (mean), "+str(value_median)+" (median)")

    # update progress bar
    progress_event(20)
    progress_event(None)

    return experiment


def create_call_tree(callpaths):
    """
    This method creates the call tree object from the callpaths read.
    It builds a structure with a root node and child nodes.
    It can be used to display the callpaths in a tree structure.
    However, this method only works if the read callpaths are in
    the correct order, as they would appear in the real program.
    """
    tree = CallTree()

    # create a two dimensional array of the callpath elements as strings
    callpaths2 = []
    max_length = 0

    for i in range(len(callpaths)):
        callpath = callpaths[i]
        callpath_string = callpath.get_name()
        elems = callpath_string.split("->")
        callpaths2.append(elems)
        if len(elems) > max_length:
            max_length = len(elems)

    # iterate over the elements of one call path
    for i in range(max_length):

        # iterate over all callpaths
        for j in range(len(callpaths2)):

            # check that we do not try to access an element that does not exist
            length = len(callpaths2[j])
            length = length-1

            if i > length:
                pass
            # if the element does exist
            else:
                callpath_string = callpaths2[j][i]

                # when at root level
                if i == 0:

                    # check if that node is already existing
                    # if no
                    if tree.node_exist(callpath_string) == False:

                        # add a new rootles node to the tree
                        node = Node(callpath_string, Callpath(callpath_string))

                        tree.add_node(node)

                    # if yes
                    else:
                        # do nothing
                        pass

                # when not at root level the root node of the elements have to be checked
                else:

                    # find the root node of the element that we want to add currently
                    root_node = find_root_node(callpaths2[j], tree, i)

                    # check if that child node is already existing
                    # if no
                    if root_node.child_exists(callpath_string) == False:

                        # add a new child node to the root node
                        child_node = Node(callpath_string, callpaths[j])

                        root_node.add_child_node(child_node)

                    # if yes
                    else:
                        # do nothing
                        pass

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
    if loop_id == level+1:
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
    level = level+1
    root_element_string = callpath_elements[level]
    childs = root_node.childs

    for i in range(len(childs)):
        child_name = childs[i].name

        if child_name == root_element_string:
            new_root_node = childs[i]

            # root node already found
            if loop_id == level+1:
                return new_root_node

            # need to search deeper in the tree for the root node
            else:
                return find_child_node(new_root_node, level, callpath_elements, loop_id)
