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

#from util.shared_library_interface import load_cube_interface
#from fileio.io_helper import create_call_tree
#from ctypes import *  # @UnusedWildImport

import os
import re
#import logging

from pycube import CubexParser  # @UnresolvedImport
from pycube.utils.exceptions import MissingMetricError  # @UnresolvedImport

"""



def configure_repetitions(dir_name):
    paths = os.listdir(dir_name)
    parameter_value_list = []
    for i in range(len(paths)):
        coordinate_string = paths[i]
        pos = coordinate_string.find(".")
        coordinate_string = coordinate_string[pos+1:]
        pos = coordinate_string.find(".r")
        coordinate_string = coordinate_string[:pos]
        if (parameter_value_list)==0:
            parameter_value_list.append(coordinate_string)
        else:
            in_list = False
            for j in range(len(parameter_value_list)):
                parameter_value_element = parameter_value_list[j]
                if coordinate_string == parameter_value_element:
                    in_list = True
                    break
            if in_list == False:
                parameter_value_list.append(coordinate_string)
    rep_map = {}
    for i in range(len(parameter_value_list)):
        item = parameter_value_list[i]
        rep_map[item] = 0
    for i in range(len(paths)):
        coordinate_string = paths[i]
        pos = coordinate_string.find(".")
        coordinate_string = coordinate_string[pos+1:]
        pos = coordinate_string.find(".r")
        coordinate_string = coordinate_string[:pos]
        rep_map[coordinate_string] += 1
    repetitions = rep_map[parameter_value_list[0]]
    return repetitions


def configure_parameter_values(dir_name, num_params):
    paths = os.listdir(dir_name)
    parameter_value_list = []
    coordinate_list = []
    for _ in range(num_params):
        coordinate_list.append([])
    for i in range(len(paths)):
        coordinate_string = paths[i]
        pos = coordinate_string.find(".")
        coordinate_string = coordinate_string[pos+1:]
        pos = coordinate_string.find(".r")
        coordinate_string = coordinate_string[:pos]
        parameter_value_list = coordinate_string.split(".")
        for i in range(num_params):
            if len(coordinate_list[i]) == 0:
                coordinate_list[i].append(parameter_value_list[i])
            else:
                in_list = False
                for j in range(len(coordinate_list[i])):
                    if parameter_value_list[i] == coordinate_list[i][j]:
                        in_list = True
                        break
                if in_list == False:
                    coordinate_list[i].append(parameter_value_list[i])
    coordinate_object_list = []
    for j in range(num_params):
        param_value_list = []
        for i in range(len(coordinate_list[j])):
            item = coordinate_list[j][i]
            parameter_name = "".join([i for i in item if not i.isdigit()])
            parameter_value = item.replace(",", ".")
            parameter_value = "".join([i for i in parameter_value if i.isdigit() or i == "."])
            parameter_value = float(parameter_value)
            param_value = ParameterValue(parameter_name, parameter_value)
            param_value_list.append(param_value)
        param_value_list.sort(key=lambda ParameterValue: ParameterValue.value, reverse=False)
        coordinate_object_list.append(param_value_list)
    parameter_value_string = ""
    for j in range(num_params):
        for i in range(len(coordinate_object_list[j])):
            item = coordinate_object_list[j][i]
            parameter_value_string += str(item.value) + ","
        parameter_value_string = parameter_value_string[:-1]
        parameter_value_string += ";"
    parameter_value_string = parameter_value_string[:-1]
    return parameter_value_string

"""
    
"""  
    # create the call tree and add it to the experiment
    callpaths = experiment.get_callpaths()
    call_tree = create_call_tree(callpaths)
    experiment.add_call_tree(call_tree)
"""


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
        
        # create the metrics
        if path_id == 0:
            cubefile_path = path + "\\" + filename
            with CubexParser(cubefile_path) as parsed:
                for metric in parsed.get_metrics():
                    if experiment.metric_exists(metric.display_name) == False:
                        experiment.add_metric(Metric(metric.display_name))
        
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
            parsed.print_calltree()
            
            # extracting all the metrics
            # should extract only the ones we can actually model
            # or offer the user to choose them...
            for metric in parsed.get_metrics():
                
                
                
                try:
                    metric_values = parsed.get_metric_values(metric=metric)
                    
                    # get number of callpaths
                    callpath_id = 0
                    while callpath_id < len(metric_values.cnode_indices):
                        #num_callpaths = len(metric_values.cnode_indices)
                    
                        # with the cnode_indices I can manipulate the region that is chosen
                        # should put for here to extract all of them
                        cnode = parsed.get_cnode(metric_values.cnode_indices[callpath_id])
                        #debug
                        #print(metric_values.cnode_indices)
                        #print("node values:",metric_values.cnode_values(cnode.id))
                        #print("number node values:",len(metric_values.cnode_values(cnode.id)))
    
                        # the more processes the more values per node, like this shows only 5 of them
                        # here we will do some magic with selecting only certain values based on that clustering algorithm
                        #TODO: measurements are read correctly and sorted to metrics, callpaths and coordinates
                        #TODO: calc mean and median over all node values
                        cnode_values = metric_values.cnode_values(cnode.id)[:5]
                        
                        
                        #TODO: works, make screenshot of calltree in old application and callpaths
                        # create callpath
                        region = parsed.get_region(cnode)
                        callpath_string = region.name
                        callpath = Callpath(callpath_string)
                        if experiment.callpath_exists(callpath_string) == False:
                            experiment.add_callpath(callpath)
                            
                        # remember id of callpath for later
                        
                        
                        print('\t' + '-' * 100)
                        print(f'\tRegion: {region.name}\n\tMetric: {metric.name}\n\tMetricValues: {cnode_values})')
                
                        callpath_id += 1
                
                except MissingMetricError as e:
                    # Ignore missing metrics
                    pass
            
     
    
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

