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
#import logging

# pycube package imports
from pycube import CubexParser  # @UnresolvedImport
from pycube.utils.exceptions import MissingMetricError  # @UnresolvedImport


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
        calltree_element_new = calltree_element_new.replace("-","")
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
            calltree_element = calltree_element.replace("-","")
            calltree_element_new = calltree_element_new + calltree_element
            calltree_elements_new.append(calltree_element_new)
            
    return calltree_elements_new


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
                #todo change i to something else
                for i in range(len(call_tree)):
                    callpath = Callpath(call_tree[i])
                    if experiment.callpath_exists(call_tree[i]) == False:
                        experiment.add_callpath(callpath)
                
                # create the call tree and add it to the experiment
                callpaths = experiment.get_callpaths()
                call_tree = create_call_tree(callpaths)
                experiment.add_call_tree(call_tree)
                
            # make list with region ids
            callpaths_ids = []
            metric_time_id = -1
            for i in range(len(parsed.get_metrics())):
                metric_name = parsed.get_metrics()[i].name
                if metric_name == "time":
                    metric_time_id = i
                    break
                
            metric_time = parsed.get_metrics()[metric_time_id]
            metric_values = parsed.get_metric_values(metric=metric_time)
            for callpath_id in range(len(metric_values.cnode_indices)):
                cnode = parsed.get_cnode(metric_values.cnode_indices[callpath_id])
                region = parsed.get_region(cnode)
                region_id = int(region.id)
                callpaths_ids.append(region_id)

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
                    
                    # standard case, all callpaths have values
                    if len(metric_values.cnode_indices) == len(callpaths):
                        for callpath_id in range(len(callpaths)):
                            cnode = parsed.get_cnode(metric_values.cnode_indices[callpath_id])
                            cnode = parsed.get_cnode(metric_values.cnode_indices[callpath_id])
                            #NOTE: here we can use clustering algorithm to select only certain node level values
                            # create the measurements
                            cnode_values = metric_values.cnode_values(cnode.id)    
                            value_mean = np.mean(cnode_values)
                            value_median = np.median(cnode_values)
                            measurement = Measurement(coordinate_id, callpath_id, metric_id, value_mean, value_median)
                            experiment.add_measurement(measurement)
                    
                    #TODO: handle missing values for specific callpaths
                    else:
                        done = False
                        counter = 0
                        cnode_counter = 0
                        while done == False:
                            cnode = parsed.get_cnode(metric_values.cnode_indices[cnode_counter])
                            region = parsed.get_region(cnode)
                            region_id = int(region.id)
                            callpath_region_id = callpaths_ids[counter]
                            
                            # if ids dont match value is missing for this callpath
                            if region_id != callpath_region_id:
                                value_mean = 0.0
                                value_median = 0.0
                                measurement = Measurement(coordinate_id, counter, metric_id, value_mean, value_median)
                                experiment.add_measurement(measurement)
                                counter += 1
                            
                            # value exists
                            else:
                                #NOTE: here we can use clustering algorithm to select only certain node level values
                                # create the measurements
                                cnode_values = metric_values.cnode_values(cnode.id)    
                                value_mean = np.mean(cnode_values)
                                value_median = np.median(cnode_values)
                                measurement = Measurement(coordinate_id, counter, metric_id, value_mean, value_median)
                                experiment.add_measurement(measurement)
                                counter += 1
                                if cnode_counter < len(metric_values.cnode_indices)-1:
                                    cnode_counter += 1
                                
                            if len(metric_values.cnode_indices)-1 == cnode_counter and counter == len(callpaths):
                                done = True
          
                except MissingMetricError as e:  # @UnusedVariable
                    # Take care of missing metrics
                    
                    # get the metric id
                    metric_id = experiment.get_metric_id(metric.name)
                    
                    # iterate over all callpaths
                    for callpath_id in range(len(callpaths)):
                        
                        # create measurement with 0.0 as value for all missing fields in cube file
                        value_mean = 0.0
                        value_median = 0.0
                        measurement = Measurement(coordinate_id, callpath_id, metric_id, value_mean, value_median)
                        experiment.add_measurement(measurement)
            
                counter += 1
                
        break
    
    #TODO: need to handle repetitions in experiment of measurements...
    # should be able to use the method from iohelper class that auto. takes care of the repetitions...
    
    return experiment
    
    


