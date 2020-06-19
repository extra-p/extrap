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

import os
import re
import numpy as np
import copy
#import logging

from pycube import CubexParser  # @UnresolvedImport
from pycube.utils.exceptions import MissingMetricError  # @UnresolvedImport


#TODO: refactor code to make it understandable
def construct_parent(x, l, i):
    l2 = l - 1
    id1 = i-1
    id2 = -1
    while id1 >= 0:
        y2 = x[id1]
        l3 = y2.count("-")
        if l2 == l3:
            id2 = id1
            break
        id1 -= 1
    y3 = x[id2]
    if y3.count("-") == 0:
        #y3 = y3 + "->"
        return y3
    else:
        l3 = y3.count("-")
        #print("count:",l3)
        y4 = construct_parent(x, l3, i)
        y3 = y3.replace("-","")
        y5 = y4 + "->" + y3
        return y5

#TODO: refactor code to make it understandable        
def fix_call_tree(calltree):
    #print(calltree)
    x = calltree.split("\n")
    x.remove("")
    #print(x)
    
    z = []
    
    for i in range(len(x)):
        y = x[i]
        if y.count("-") == 0:
            z.append(y)
        elif y.count("-") > 0:
            l = y.count("-")
            
            y2 = construct_parent(x, l, i)
            y2 = y2 + "->"
            #print("boobs:",y2)
            
            y = y.replace("-","")
            y3 = y2 + y
            z.append(y3)
            
    return z

#TODO: check what the scaling type did in the code and c++ code...
def read_cube_file(dir_name, scaling_type):
    
    # read the paths of the cube files in the given directory with dir_name
    paths = []
    folders = os.listdir(dir_name)
    for folder_id in range(len(folders)):
        folder_name = folders[folder_id]
        path = dir_name + folder_name
        paths.append(path)
    
    # iterate over all folders and read the cube profiles in them
    filename = "profile.cubex"
    experiment = Experiment()
    
    for path_id in range(len(paths)):
        path = paths[path_id]
        folder_name = folders[path_id]
                
        # create the parameters
        pos = folder_name.find(".")
        folder_name = folder_name[pos+1:]
        pos = folder_name.find(".r")
        folder_name = folder_name[:pos]
        parameters = folder_name.split(".")
        
        # when there is only one parameter
        if len(parameters) == 1:
            parameter = parameters[0]
            param_list = re.split("(\d+)", parameter)
            param_list.remove("")
            parameter_name = param_list[0]
            if param_list[1].find(","):
                param_list[1] = param_list[1].replace(",",".")
            parameter_value = float(param_list[1])
            
            # check if parameter already exists
            if path_id == 0:
                if experiment.parameter_exists(parameter_name) == False:
                    parameter = Parameter(parameter_name)
                    experiment.add_parameter(parameter)
            
            # create coordinate
            coordinate = Coordinate()
            parameter_id = experiment.get_parameter_id(parameter_name)
            parameter = experiment.get_parameter(parameter_id)
            coordinate.add_parameter_value(parameter, parameter_value)
            
            # check if the coordinate already exists
            if experiment.coordinate_exists(coordinate) == False:
                experiment.add_coordinate(coordinate)
                
            # get the coordinate id
            coordinate_id = experiment.get_coordinate_id(coordinate)
        
        # when there a several parameters
        elif len(parameters) > 1:
            coordinate = Coordinate()
            
            for parameter_id in range(len(parameters)):
                parameter = parameters[parameter_id]
                param_list = re.split("(\d+)", parameter)
                param_list.remove("")
                parameter_name = param_list[0]
                if param_list[1].find(","):
                    param_list[1] = param_list[1].replace(",",".")
                parameter_value = float(param_list[1])
                
                # check if parameter already exists
                if path_id == 0:
                    if experiment.parameter_exists(parameter_name) == False:
                        parameter = Parameter(parameter_name)
                        experiment.add_parameter(parameter)
                        
                # create coordinate
                parameter_id = experiment.get_parameter_id(parameter_name)
                parameter = experiment.get_parameter(parameter_id)
                coordinate.add_parameter_value(parameter, parameter_value)
                
            # check if the coordinate already exists
            if experiment.coordinate_exists(coordinate) == False:
                experiment.add_coordinate(coordinate)
                
            # get the coordinate id
            coordinate_id = experiment.get_coordinate_id(coordinate)

        
        #TODO: for windows systems only, add something for linux as well!
        cubefile_path = path + "\\" + filename
        print("path:",cubefile_path)
        
        with CubexParser(cubefile_path) as parsed:
            
            # get call tree
            if path_id == 0:
                call_tree = parsed.get_calltree()
                call_tree = fix_call_tree(call_tree)
                
                # create the callpaths
                for i in range(len(call_tree)):
                    callpath = Callpath(call_tree[i])
                    if experiment.callpath_exists(call_tree[i]) == False:
                        experiment.add_callpath(callpath)
                
                # create the call tree and add it to the experiment
                callpaths = experiment.get_callpaths()
                call_tree = create_call_tree(callpaths)
                experiment.add_call_tree(call_tree)

            #NOTE: here we could choose which metrics to extract
            # iterate over all metrics
            counter = 0
            for metric in parsed.get_metrics():
                
                # create the metrics
                if path_id == 0:
                    if experiment.metric_exists(metric.name) == False:
                        experiment.add_metric(Metric(metric.name))
                        
                # get the metric id
                metric_id = experiment.get_metric_id(metric.name)
                        
                try:
                    metric_values = parsed.get_metric_values(metric=metric)
                    
                    # iterate over all callpaths
                    for callpath_id in range(len(metric_values.cnode_indices)):
                        cnode = parsed.get_cnode(metric_values.cnode_indices[callpath_id])
                        
                        #NOTE: here we can use clustering algorithm to select only certain node level values
                        # create the measurements
                        cnode_values = metric_values.cnode_values(cnode.id)    
                        value_mean = np.mean(cnode_values)
                        value_median = np.median(cnode_values)
                        measurement = Measurement(coordinate_id, callpath_id, metric_id, value_mean, value_median)
                        experiment.add_measurement(measurement)
                                    
                except MissingMetricError as e:  # @UnusedVariable
                    # Ignore missing metrics
                    #TODO: check what happens here...!
                    print(e)
                    pass
            
                counter += 1
                
        break
    
    #TODO: need to handle repetitions in experiment of measurements...
    
    return experiment
    
    
    """

    # set configuration for loading the cube files
    prefix = configure_prefix(dir_name)
    num_params = configure_nr_parameters(dir_name)
    postfix = ""
    filename = "profile.cubex"
    displayed_names = configure_displayed_names(dir_name)
    names = configure_names(dir_name)
    repetitions = configure_repetitions(dir_name)
    parameter_values = configure_parameter_values(dir_name, num_params)

    if scaling_type == 0:
        logging.debug("scaling type: weak")
    else:
        logging.debug("scaling type: strong")
        
    logging.debug("dir name: "+str(dir_name))
    logging.debug("prefix: "+str(prefix))
    logging.debug("post fix: "+str(postfix))
    logging.debug("filename: "+str(filename))
    logging.debug("repetitions: "+str(repetitions))
    logging.debug("num params: "+str(num_params))
    logging.debug("displayed names: "+str(displayed_names))
    logging.debug("names: "+str(names))
    logging.debug("parameter values: "+str(parameter_values))

    cube_interface = load_cube_interface()

    # encode string so they can be read by the c code as char*
    b_dir_name = dir_name.encode('utf-8')
    b_prefix = prefix.encode('utf-8')
    b_postfix = postfix.encode('utf-8')
    b_filename = filename.encode('utf-8')
    b_displayed_names = displayed_names.encode('utf-8')
    b_names = names.encode('utf-8')
    b_parameter_values = parameter_values.encode('utf-8')

    # pointer object for c++ data structure
    data_pointer = POINTER(Data)
    exposed_function = cube_interface.exposed_function
    exposed_function.restype = data_pointer

    # number of parameters
    getNumParameters = cube_interface.getNumParameters
    getNumParameters.restype = c_int

    # number of chars for one paramater
    getNumCharsParameters = cube_interface.getNumCharsParameters
    getNumCharsParameters.restype = c_int

    # parameters char
    getParameterChar = cube_interface.getParameterChar
    getParameterChar.restype = c_char

    # number of coordinates
    getNumCoordinates = cube_interface.getNumCoordinates
    getNumCoordinates.restype = c_int

    # number of chars for one coordinate
    getNumCharsCoordinates = cube_interface.getNumCharsCoordinates
    getNumCharsCoordinates.restype = c_int

    # coordinate char
    getCoordinateChar = cube_interface.getCoordinateChar
    getCoordinateChar.restype = c_char

    # callpaths char
    getCallpathChar = cube_interface.getCallpathChar
    getCallpathChar.restype = c_char
    
    # number of callpaths
    getNumCallpaths = cube_interface.getNumCallpaths
    getNumCallpaths.restype = c_int

    # number of chars for one callpath
    getNumCharsCallpath = cube_interface.getNumCharsCallpath
    getNumCharsCallpath.restype = c_int

    # number of metrics
    getNumMetrics = cube_interface.getNumMetrics
    getNumMetrics.restype = c_int

    # number of chars for one metric
    getNumCharsMetrics = cube_interface.getNumCharsMetrics
    getNumCharsMetrics.restype = c_int

    # metrics char
    getMetricChar = cube_interface.getMetricChar
    getMetricChar.restype = c_char

    # data point values
    getDataPointValue = cube_interface.getDataPointValue
    getDataPointValue.restype = c_double

    # get pointer to c++ data object for mean values
    dp = data_pointer()  # @UnusedVariable
    dp = exposed_function(scaling_type, b_dir_name, b_prefix, b_postfix, b_filename, repetitions, num_params, b_displayed_names, b_names, b_parameter_values, 1)
    
    # get pointer to c++ data object for median values
    dp2 = data_pointer()  # @UnusedVariable
    dp2 = exposed_function(scaling_type, b_dir_name, b_prefix, b_postfix, b_filename, repetitions, num_params, b_displayed_names, b_names, b_parameter_values, 0)
    
    # create an experiment object to save the date loaded from the cube file
    experiment = Experiment()

    number_parameters = getNumParameters(dp)
    
    if number_parameters >=1 and number_parameters <= 3:
        
        # get the parameters
        for element_id in range(number_parameters):
            num_chars = getNumCharsParameters(dp, element_id)
            parameter_string = ""
            for char_id in range(num_chars):
                byte_parameter = getParameterChar(dp, element_id, char_id)
                parameter_string += byte_parameter.decode('utf-8')
            logging.debug("Parameter "+str(element_id+1)+": "+parameter_string)
            # save the parameter in the experiment object
            parameter = Parameter(parameter_string)
            experiment.add_parameter(parameter)
    
        # get the coordinates
        number_coordinates = getNumCoordinates(dp)
        for element_id in range(number_coordinates):
            num_chars = getNumCharsCoordinates(dp, element_id)
            coordinate_string = ""
            for char_id in range(num_chars):
                byte_coordinate = getCoordinateChar(dp, element_id, char_id)
                coordinate_string += byte_coordinate.decode('utf-8')
            logging.debug("Coordinate "+str(element_id+1)+": "+coordinate_string)
            # save the coordinate in the experiment object
            coordinate = Coordinate()
            
            # if there is only a single parameter
            if number_parameters == 1:
                coordinate_string = coordinate_string[1:]
                coordinate_string = coordinate_string[:-1]
                pos = coordinate_string.find(",")
                parameter_name = coordinate_string[:pos]
                parameter_value = coordinate_string[pos+1:]
                parameter_value = float(parameter_value)
                parameter_id = experiment.get_parameter_id(parameter_name)
                parameter = experiment.get_parameter(parameter_id)
                coordinate.add_parameter_value(parameter, parameter_value)
            
            # when there are several parameters
            else:
                coordinate_string = coordinate_string[1:]
                coordinate_string = coordinate_string[:-1]
                coordinate_string = coordinate_string.replace(")(", ";")
                elements = coordinate_string.split(";")
                for element_id in range(len(elements)):
                    element = elements[element_id]
                    parts = element.split(",")
                    parameter_name = parts[0]
                    parameter_value = parts[1]
                    parameter_value = float(parameter_value)
                    parameter_id = experiment.get_parameter_id(parameter_name)
                    parameter = experiment.get_parameter(parameter_id)
                    coordinate.add_parameter_value(parameter, parameter_value)
                    
            experiment.add_coordinate(coordinate)
    
        # get the callpaths
        number_callpaths = getNumCallpaths(dp)
        for element_id in range(number_callpaths):
            num_chars = getNumCharsCallpath(dp, element_id)
            callpath_string = ""
            for char_id in range(num_chars):
                byte_callpath = getCallpathChar(dp, element_id, char_id)
                callpath_string += byte_callpath.decode('utf-8')
            logging.debug("Callpath "+str(element_id+1)+": "+callpath_string)
            # save the callpath in the experiment object
            callpath = Callpath(callpath_string)
            experiment.add_callpath(callpath)
            
        # create the call tree and add it to the experiment
        callpaths = experiment.get_callpaths()
        call_tree = create_call_tree(callpaths)
        experiment.add_call_tree(call_tree)
        
        # get the metrics
        number_metrics = getNumMetrics(dp)
        for element_id in range(number_metrics):
            num_chars = getNumCharsMetrics(dp, element_id)
            metric_string = ""
            for char_id in range(num_chars):
                byte_metric = getMetricChar(dp, element_id, char_id)
                metric_string += byte_metric.decode('utf-8')
            logging.debug("Metric "+str(element_id+1)+": "+metric_string)
            # save the metric in the experiment object
            metric = Metric(metric_string)
            experiment.add_metric(metric)
    
        # get the measurements per metric, callpath, coordinate (no repetitions, value is mean or median computed cube)  
        for metric_id in range(number_metrics):
            for callpath_id in range(number_callpaths):
                for coordinate_id in range(number_coordinates):
                    value_mean = getDataPointValue(dp, metric_id, callpath_id, coordinate_id)
                    value_mean = float(value_mean)
                    value_median = getDataPointValue(dp2, metric_id, callpath_id, coordinate_id)
                    value_median = float(value_median)
                    # save the measurement in the experiment object
                    measurement = Measurement(coordinate_id, callpath_id, metric_id, value_mean, value_median)
                    experiment.add_measurement(measurement)
                    logging.debug("Measurement: "+experiment.get_metric(metric_id).get_name()+", "+experiment.get_callpath(callpath_id).get_name()+", "+experiment.get_coordinate(coordinate_id).get_as_string()+": "+str(value_mean)+" (mean), "+str(value_median)+" (median)")
      
    else:
        logging.critical("This input format supports a maximum of 3 parameters.")
    
    return experiment
    
    """

