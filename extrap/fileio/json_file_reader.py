# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import json
import logging
from json.decoder import JSONDecodeError

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio import io_helper
from extrap.fileio.io_helper import create_call_tree
from extrap.fileio.jsonlines_file_reader import read_jsonlines_file
from extrap.util.exceptions import FileFormatError
from extrap.util.progress_bar import DUMMY_PROGRESS

SCHEMA_URI = ""


def read_json_file(path, progress_bar=DUMMY_PROGRESS):
    # read lines from json file
    with open(path, "r") as inputfile:
        try:
            json_data = json.load(inputfile)
        except JSONDecodeError as error:
            inputfile.seek(0)
            is_jsonlines = any(line.strip().startswith('{') for line in inputfile) and \
                           all(line.strip().startswith('{') or line.strip() == "" for line in inputfile)
            if is_jsonlines:
                return read_jsonlines_file(path, progress_bar=DUMMY_PROGRESS)
            else:
                raise FileFormatError(str(error)) from error

    # create an experiment object to save the date loaded from the text file
    experiment = Experiment()

    if "callpaths" not in json_data:
        try:
            _read_new_json_file(experiment, json_data, progress_bar)
        except KeyError as err:
            raise FileFormatError(str(err)) from err
    else:
        try:
            _read_legacy_json_file(experiment, json_data, progress_bar)
        except KeyError as err:
            raise FileFormatError(str(err)) from err

    call_tree = create_call_tree(experiment.callpaths, progress_bar)
    experiment.call_tree = call_tree

    io_helper.validate_experiment(experiment, progress_bar)

    return experiment


def _read_new_json_file(experiment, json_data, progress_bar):
    parameter_data = json_data["parameters"]
    for p in parameter_data:
        parameter = Parameter(p)
        experiment.add_parameter(parameter)

    measurements_data = json_data["measurements"]
    for callpath_name, data in progress_bar(measurements_data.items()):
        for metric_name, measurements in data.items():
            for measurement in measurements:
                coordinate = Coordinate(measurement['point'])
                experiment.add_coordinate(coordinate)
                callpath = Callpath(callpath_name)
                experiment.add_callpath(callpath)
                metric = Metric(metric_name)
                experiment.add_metric(metric)
                measurement = Measurement(coordinate, callpath, metric, measurement['values'])
                experiment.add_measurement(measurement)


def _read_legacy_json_file(experiment, json_data, progress_bar):
    # read parameters
    parameter_data = json_data["parameters"]
    parameter_data = sorted(parameter_data, key=lambda x: x["id"])
    logging.debug("Number of parameters: " + str(len(parameter_data)))
    for i, p_data in enumerate(progress_bar(parameter_data)):
        parameter_name = p_data["name"]
        parameter = Parameter(parameter_name)
        experiment.add_parameter(parameter)
        logging.debug("Parameter " + str(i + 1) + ": " + parameter_name)
    # read callpaths
    callpath_data = json_data["callpaths"]
    logging.debug("Number of callpaths: " + str(len(callpath_data)))
    for i, c_data in enumerate(progress_bar(callpath_data)):
        callpath_name = c_data["name"]
        callpath = Callpath(callpath_name)
        experiment.add_callpath(callpath)
        logging.debug("Callpath " + str(i + 1) + ": " + callpath_name)
    # read metrics
    metric_data = json_data["metrics"]
    logging.debug("Number of metrics: " + str(len(metric_data)))
    for i, m_data in enumerate(progress_bar(metric_data)):
        metric_name = m_data["name"]
        metric = Metric(metric_name)
        experiment.add_metric(metric)
        logging.debug("Metric " + str(i + 1) + ": " + metric_name)
    # read coordinates
    coordinate_data = json_data["coordinates"]
    logging.debug("Number of coordinates: " + str(len(coordinate_data)))
    for i, c_data in enumerate(progress_bar(coordinate_data)):
        parameter_value_pairs = c_data["parameter_value_pairs"]
        parameter_value_pairs = sorted(parameter_value_pairs, key=lambda x: x["parameter_id"])
        coordinate = Coordinate()
        for j in range(len(parameter_value_pairs)):
            parameter_value_pair = parameter_value_pairs[j]
            parameter_id = int(parameter_value_pair["parameter_id"]) - 1
            parameter_value = float(parameter_value_pair["parameter_value"])
            parameter = experiment.get_parameter(parameter_id)
            coordinate.add_parameter_value(parameter, parameter_value)
        experiment.add_coordinate(coordinate)
        logging.debug("Coordinate " + str(i + 1) + ": " + coordinate.get_as_string())
    aggregate_data = {}
    # read measurements
    measurements_data = json_data["measurements"]
    logging.debug("Number of measurements: " + str(len(measurements_data)))
    for i, m_data in enumerate(progress_bar(measurements_data)):
        coordinate_id = int(m_data["coordinate_id"]) - 1
        callpath_id = int(m_data["callpath_id"]) - 1
        metric_id = int(m_data["metric_id"]) - 1
        value = float(m_data["value"])
        key = coordinate_id, callpath_id, metric_id
        if key in aggregate_data:
            aggregate_data[key].append(value)
        else:
            aggregate_data[key] = [value]
    for key in progress_bar(aggregate_data):
        coordinate_id, callpath_id, metric_id = key
        coordinate = experiment.get_coordinate(coordinate_id)
        callpath = experiment.get_callpath(callpath_id)
        metric = experiment.get_metric(metric_id)
        values = aggregate_data[key]
        measurement = Measurement(coordinate, callpath, metric, values)
        experiment.add_measurement(measurement)
