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
from fileio.io_helper import compute_repetitions
from fileio.io_helper import create_call_tree
import logging
import json

def read_json_file(path):
    
    # read lines from json file
    with open(path, "r") as inputfile:
        json_data = json.load(inputfile)
        
    # create an experiment object to save the date loaded from the text file
    experiment = Experiment()
    
    # read parameters
    parameter_data = json_data["parameters"]
    logging.debug("Number of parameters: "+str(len(parameter_data)))
    for i in range(len(parameter_data)):
        parameter_name = parameter_data[i]["name"]
        parameter = Parameter(parameter_name)
        experiment.add_parameter(parameter)
        logging.debug("Parameter "+str(i+1)+": "+parameter_name)
    
    # read callpaths
    callpath_data = json_data["callpaths"]
    logging.debug("Number of callpaths: "+str(len(callpath_data)))
    for i in range(len(callpath_data)):
        callpath_name = callpath_data[i]["name"]
        callpath = Callpath(callpath_name)
        experiment.add_callpath(callpath)
        logging.debug("Callpath "+str(i+1)+": "+callpath_name)
        
    # create the call tree and add it to the experiment
    callpaths = experiment.get_callpaths()
    call_tree = create_call_tree(callpaths)
    experiment.add_call_tree(call_tree)

    # read metrics
    metric_data = json_data["metrics"]
    logging.debug("Number of metrics: "+str(len(metric_data)))
    for i in range(len(metric_data)):
        metric_name = metric_data[i]["name"]
        metric = Metric(metric_name)
        experiment.add_metric(metric)
        logging.debug("Metric "+str(i+1)+": "+metric_name)

    # read coordinates
    coordinate_data = json_data["coordinates"]
    logging.debug("Number of coordinates: "+str(len(coordinate_data)))
    for i in range(len(coordinate_data)):
        parameter_value_pairs = coordinate_data[i]["parameter_value_pairs"]
        coordinate = Coordinate()
        for j in range(len(parameter_value_pairs)):
            parameter_value_pair = parameter_value_pairs[j]
            parameter_id = int(parameter_value_pair["parameter_id"]) - 1
            parameter_value = float(parameter_value_pair["parameter_value"])
            parameter = experiment.get_parameter(parameter_id)
            coordinate.add_parameter_value(parameter, parameter_value)
        experiment.add_coordinate(coordinate)
        logging.debug("Coordinate "+str(i+1)+": "+coordinate.get_as_string())
    
    # read measurements
    measurements_data = json_data["measurements"]
    logging.debug("Number of measurements: "+str(len(measurements_data)))
    for i in range(len(measurements_data)):
        coordinate_id = int(measurements_data[i]["coordinate_id"]) - 1
        callpath_id = int(measurements_data[i]["callpath_id"]) - 1
        metric_id = int(measurements_data[i]["metric_id"]) - 1
        value = float(measurements_data[i]["value"])
        measurement = Measurement(coordinate_id, callpath_id, metric_id, value, None)
        experiment.add_measurement(measurement)
        
    # compute the mean or median of the repetitions for each coordinate
    experiment = compute_repetitions(experiment)
    
    return experiment

