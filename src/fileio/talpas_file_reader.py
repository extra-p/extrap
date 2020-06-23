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

def read_talpas_file(path):
    
    # read talpas file into list
    lines = []
    with open(path) as file:
        for line in file:
            if line != "\n":
                lines.append(line)
    
    # remove line breaks
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
        
    # create an experiment object to save the date loaded from the text file
    experiment = Experiment()
    
    for i in range(len(lines)):
        line = lines[i]
        
        # read the parameter names and values
        pos = line.find("\"parameters\"")
        pos = pos + len("\"parameters\"") + len(":")
        parameters_string = line[pos:]
        pos = parameters_string.find(";")
        parameters_string = parameters_string[:pos]
        parameters_string = parameters_string.replace("{", "")
        parameters_string = parameters_string.replace("}", "")
        parameter_list = parameters_string.split(",")
        
        coordinate = Coordinate()
        for j in range(len(parameter_list)):
            parameter_element = parameter_list[j]
            parameter_element = parameter_element.replace("\"", "")
            pos = parameter_element.find(":")
            parameter_name = parameter_element[:pos]
            
            # if the parameter does not exist create new object and add it to the experiment
            if experiment.parameter_exists(parameter_name) == False:
                parameter = Parameter(parameter_name)
                experiment.add_parameter(parameter)
                logging.debug("Parameter: "+parameter_name)
                
            # create the coordinate object
            parameter_id = experiment.get_parameter_id(parameter_name)
            parameter = experiment.get_parameter(parameter_id)
            parameter_value = parameter_element[pos+1:]
            value = float(parameter_value)
            coordinate.add_parameter_value(parameter, value)
        
        # check if the coordinate exists
        if experiment.coordinate_exists(coordinate) == False:
            experiment.add_coordinate(coordinate)
            logging.debug("Coordinate: "+coordinate.get_as_string())
        
        # read the metric
        pos = line.find("\"metric\"")
        pos = pos + len("\"metric\"") + len(":")
        metric_name = line[pos:]
        pos = metric_name.find(";")
        metric_name = metric_name[:pos]
        metric_name = metric_name.replace("\"", "")
        
        # if metric does not exist create new object and add it to the experiment
        if experiment.metric_exists(metric_name) == False:
            metric = Metric(metric_name)
            experiment.add_metric(metric)
            logging.debug("Metric: "+metric_name)
        
        # read the callpath
        pos = line.find("\"callpath\"")
        pos = pos + len("\"callpath\"") + len(":")
        callpath_name = line[pos:]
        pos = callpath_name.find(";")
        callpath_name = callpath_name[:pos]
        callpath_name = callpath_name.replace("\"", "")
        
        # if callpath does not exist create new object and add it to the experiment
        if experiment.callpath_exists(callpath_name) == False:
            callpath = Callpath(callpath_name)
            experiment.add_callpath(callpath)
            logging.debug("Callpath: "+callpath_name)
        
        # read the value
        pos = line.find("\"value\"")
        pos = pos + len("\"value\"") + len(":")
        value_string = line[pos:]
        pos = value_string.find("}")
        value_string = value_string[:pos]
        value = float(value_string)
        
        # create measurement object and add it to experiment
        callpath_id = experiment.get_callpath_id(callpath_name)
        metric_id = experiment.get_metric_id(metric_name)
        coordinate_id = experiment.get_coordinate_id(coordinate)
        measurement = Measurement(coordinate_id, callpath_id, metric_id, value, None)
        experiment.add_measurement(measurement)
        
    # compute the mean or median of the repetitions for each coordinate
    experiment = compute_repetitions(experiment)
    
    # create the call tree and add it to the experiment
    callpaths = experiment.get_callpaths()
    call_tree = create_call_tree(callpaths)
    experiment.add_call_tree(call_tree)
    
    return experiment

