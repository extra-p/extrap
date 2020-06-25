from entities.parameter import Parameter
from entities.measurement import Measurement
from entities.coordinate import Coordinate
from entities.callpath import Callpath
from entities.metric import Metric
from entities.experiment import Experiment
from fileio.io_helper import create_call_tree
from fileio.io_helper import compute_repetitions
import os
import re
import numpy
import logging
from tqdm import tqdm
from pathlib import Path
from pycubexr import CubexParser  # @UnresolvedImport
from pycubexr.utils.exceptions import MissingMetricError  # @UnresolvedImport


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
    
    logging.info("Reading cube files...")
    
    #TODO: the progress bar should be only active when using the command line tool, since gui dont use it.
    # create a progress bar for reading the cube files
    show_message = False
    with tqdm(total=len(paths)) as pbar:
    
        for path_id in range(len(paths)):
            
            path = paths[path_id]
            folder_name = folders[path_id]
            
            #TODO debug
            print("File:",folder_name)
                    
            # create the parameters
            pos = folder_name.find(".")
            folder_name = folder_name[pos+1:]
            pos = folder_name.find(".r")
            folder_name = folder_name[:pos]
            parameters = folder_name.split(".")
            
            # when there is only one parameter
            if len(parameters) == 1:
                
                # set scaling flag for experiment
                if path_id == 0:
                    if scaling_type == "weak":
                        experiment.set_scaling("weak")
                    elif scaling_type == "strong":
                        experiment.set_scaling("strong")
                
                parameter = parameters[0]
                param_list = re.split("(\d+)", parameter)
                param_list.remove("")
                
                # if parameter is float value
                if "," in param_list:
                    parameter_name = param_list[0]
                    parameter_value = ""
                    counter = 1
                    while counter < len(param_list):
                        parameter_value += param_list[counter]
                        counter += 1
                    parameter_value = parameter_value.replace(",", ".")
                    parameter_value = float(parameter_value)
                
                # if parameter is integer value
                else:
                    parameter_name = param_list[0]
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
                
                # set scaling flag for experiment
                if path_id == 0:
                    if scaling_type == "weak":
                        experiment.set_scaling("weak")
                    elif scaling_type == "strong":
                        experiment.set_scaling("weak")
                        show_message = True
                
                coordinate = Coordinate()
                
                for parameter_id in range(len(parameters)):
                    parameter = parameters[parameter_id]
                    param_list = re.split("(\d+)", parameter)
                    param_list.remove("")
                    
                    # if parameter is float value
                    if "," in param_list:
                        parameter_name = param_list[0]
                        parameter_value = ""
                        counter = 1
                        while counter < len(param_list):
                            parameter_value += param_list[counter]
                            counter += 1
                        parameter_value = parameter_value.replace(",", ".")
                        parameter_value = float(parameter_value)
                    
                    # if parameter is integer value
                    else:
                        parameter_name = param_list[0]
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
    
            cubefile_path = Path("")
            cubefile_path = cubefile_path / path / filename
            
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
                    #TODO debug
                    call_tree.print_tree()
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
                    #TODO debug
                    print(region)
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
                                
                                #NOTE: here we can use clustering algorithm to select only certain node level values
                                # create the measurements
                                cnode_values = metric_values.cnode_values(cnode.id)
                                
                                # in case of weak scaling calculate mean and median over all mpi process values
                                if scaling_type == "weak":
                                    value_mean = float(numpy.mean(cnode_values))
                                    value_median = float(numpy.median(cnode_values))
                                    
                                # in case of strong scaling calculate the sum over all mpi process values
                                elif scaling_type == "strong":
                                    # check number of parameters, if > 1 use weak scaling instead
                                    # since sum values for strong scaling does not work for more than 1 parameter
                                    if len(experiment.get_parameters()) > 1:
                                        value_mean = float(numpy.mean(cnode_values))
                                        value_median = float(numpy.median(cnode_values))
                                    else:
                                        value = float(numpy.sum(cnode_values))
                                        value_mean = value
                                        value_median = value
                                    
                                measurement = Measurement(coordinate_id, callpath_id, metric_id, value_mean, value_median)
                                experiment.add_measurement(measurement)
                        
                        # handle missing values for specific callpaths
                        else:
                            done = False
                            counter = 0
                            cnode_counter = 0
                            #debug
                            print("len callpaths_ids:",len(callpaths_ids))
                            #print("callpaths number:",len(callpaths))
                            while done == False:
                                cnode = parsed.get_cnode(metric_values.cnode_indices[cnode_counter])
                                region = parsed.get_region(cnode)
                                region_id = int(region.id)
                                #debug
                                #print("counter:",counter)
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
                                    
                                    # in case of weak scaling calculate mean and median over all mpi process values
                                    if scaling_type == "weak":
                                        value_mean = float(numpy.mean(cnode_values))
                                        value_median = float(numpy.median(cnode_values))
                                        
                                    # in case of strong scaling calculate the sum over all mpi process values
                                    elif scaling_type == "strong":
                                        # check number of parameters, if > 1 use weak scaling instead
                                        # since sum values for strong scaling does not work for more than 1 parameter
                                        if len(experiment.get_parameters()) > 1:
                                            value_mean = float(numpy.mean(cnode_values))
                                            value_median = float(numpy.median(cnode_values))
                                        else:  
                                            value = float(numpy.sum(cnode_values))
                                            value_mean = value
                                            value_median = value
                                    
                                    measurement = Measurement(coordinate_id, counter, metric_id, value_mean, value_median)
                                    experiment.add_measurement(measurement)
                                    
                                    counter += 1
                                    if cnode_counter < len(metric_values.cnode_indices)-1:
                                        cnode_counter += 1
                                    
                                if len(metric_values.cnode_indices)-1 == cnode_counter and counter == len(callpaths):
                                    done = True
                    
                    # Take care of missing metrics
                    except MissingMetricError as e:  # @UnusedVariable
                        
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
            
            # update progress bar
            pbar.update(1)
            
            #TODO: debug
            #break
        
        if show_message == True:
            logging.warning("Strong scaling only works for one parameter. Using weak scaling instead.")
                        
    # take care of the repetitions of the measurements
    experiment = compute_repetitions(experiment)
    
    return experiment
    