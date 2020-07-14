"""
This file is part of the Extra-P software (https://github.com/extra-p/extrap)

Copyright (c) 2020 Technical University of Darmstadt, Darmstadt, Germany

All rights reserved.
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""


from entities.parameter import Parameter
from entities.measurement import Measurement
from entities.coordinate import Coordinate
from entities.callpath import Callpath
from entities.metric import Metric
from entities.experiment import Experiment
from fileio.io_helper import create_call_tree
import logging


def read_text_file(path):

    # read text file into list
    lines = []
    with open(path) as file:
        for line in file:
            if not line.isspace():
                lines.append(line)

    # remove empty lines
    ids = []
    for i in range(len(lines)):
        if lines[i] == "\n":
            ids.append(i)
    for i in range(len(ids)):
        lines.pop(ids[i])

    # remove line breaks
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")

    # create an experiment object to save the date loaded from the text file
    experiment = Experiment()

    # variables for parsing
    number_parameters = 0
    last_metric = ""
    last_callpath = ""
    coordinate_id = 0

    # parse text to extrap objects
    for i in range(len(lines)):
        # get the current line
        line = lines[i]

        # get field name
        field_seperator_idx = line.find(" ")
        field_name = line[:field_seperator_idx]
        field_value = line[field_seperator_idx + 1:].strip()

        if field_name == "METRIC":
            # create a new metric if not already exists
            metric_name = field_value
            if experiment.metric_exists(metric_name) == False:
                metric = Metric(metric_name)
                experiment.add_metric(metric)
                last_metric = metric_name
            else:
                last_metric = metric_name
            # reset the coordinate id, since moving to a new region
            coordinate_id = 0

        elif field_name == "REGION":
            # create a new region if not already exists
            callpath_name = field_value
            if experiment.callpath_exists(callpath_name) == False:
                callpath = Callpath(callpath_name)
                experiment.add_callpath(callpath)
                last_callpath = callpath_name
            else:
                last_callpath = callpath_name
            # reset the coordinate id, since moving to a new region
            coordinate_id = 0

        elif field_name == "DATA":
            # create a new data set
            data_string = field_value
            data_string = data_string.strip()
            data_list = data_string.split(" ")
            for i in range(len(data_list)):
                data_list[i] = float(data_list[i])
            if number_parameters >= 1 and number_parameters <= 4:
                # create one measurement per repetition
                value_mean = 0
                value_median = 0

                # calculate mean value
                value_mean = sum(data_list)/len(data_list)

                # calculate median value
                sorted_data_list = sorted(data_list)
                # even number of elements
                if len(sorted_data_list) % 2 == 0:
                    middle_id_1 = int(len(sorted_data_list) / 2) - 1
                    middle_id_2 = middle_id_1 + 1
                    value_median = (
                        sorted_data_list[middle_id_1] + sorted_data_list[middle_id_2]) / 2
                # uneven number of elements
                else:
                    middle_id = int((len(sorted_data_list) + 1) / 2) - 1
                    value_median = sorted_data_list[middle_id]

                if not experiment.metric_exists(last_metric):
                    metric = Metric(last_metric)
                    experiment.add_metric(metric)
                callpath_id = experiment.get_callpath_id(last_callpath)
                metric_id = experiment.get_metric_id(last_metric)
                measurement = Measurement(
                    coordinate_id, callpath_id, metric_id, value_mean, value_median)
                experiment.add_measurement(measurement)

                coordinate_id += 1
            else:
                logging.warning(
                    "This input format supports a maximum of 4 parameters.")

        elif field_name == "PARAMETER":
            # create a new parameter
            parameters = field_value.split(' ')
            for p in parameters:
                experiment.add_parameter(Parameter(p))
                number_parameters += 1

        elif field_name == "POINTS":
            coordinate_string = field_value
            if '(' in coordinate_string:
                coordinate_string = coordinate_string.replace(") (", ")(")
                coordinate_string = coordinate_string[1:-1]
                coordinate_strings = coordinate_string.split(')(')
            else:
                coordinate_strings = coordinate_string.split(' ')
            # create a new point
            if number_parameters == 1:
                for value_string in coordinate_strings:
                    value = float(value_string)
                    parameter = experiment.get_parameter(0)
                    coordinate = Coordinate([(parameter, value)])
                    experiment.add_coordinate(coordinate)
            elif number_parameters > 1 and number_parameters < 5:
                for coordinate_string in coordinate_strings:
                    coordinate_string = coordinate_string.strip()
                    values = coordinate_string.split(" ")
                    coordinate = Coordinate(
                        [(experiment.get_parameter(j), float(value))
                         for j, value in enumerate(values)])
                    experiment.add_coordinate(coordinate)
            else:
                logging.warning(
                    "This input format supports a maximum of 4 parameters.")

    # create the call tree and add it to the experiment
    callpaths = experiment.get_callpaths()
    call_tree = create_call_tree(callpaths)
    experiment.add_call_tree(call_tree)

    return experiment
