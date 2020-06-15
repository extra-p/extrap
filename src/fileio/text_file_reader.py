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
            if line != "\n":
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
        field_name = line[:line.find(" ")]
        
        if field_name == "METRIC":
            # create a new metric if not already exists
            metric_name = line[line.find(" ")+1:]
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
            callpath_name = line[line.find(" ")+1:]
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
            data_string = line[line.find(" ")+1:]
            data_list = data_string.split(" ")
            for i in range(len(data_list)):
                data_list[i] = float(data_list[i])
            if number_parameters >= 1 and number_parameters <=4:
                # create one measurement per repetition
                value_mean = 0
                value_median = 0
                
                # calculate mean value
                value_mean = sum(data_list)/len(data_list) 
                
                # calculate median value
                sorted_data_list = sorted(data_list)
                # even number of elements
                if len(sorted_data_list) % 2 == 0:
                    middle_id_1 = int(len(sorted_data_list) / 2) -1
                    middle_id_2 = middle_id_1 + 1
                    value_median = (sorted_data_list[middle_id_1] + sorted_data_list[middle_id_2]) / 2
                # uneven number of elements
                else:
                    middle_id = int((len(sorted_data_list) + 1) / 2) - 1
                    value_median = sorted_data_list[middle_id]

                callpath_id = experiment.get_callpath_id(last_callpath)
                metric_id = experiment.get_metric_id(last_metric)
                measurement = Measurement(coordinate_id, callpath_id, metric_id, value_mean, value_median)
                experiment.add_measurement(measurement)
                
                coordinate_id += 1
            else:
                logging.warning("This input format supports a maximum of 4 parameters.")
        
        elif field_name == "PARAMETER":
            # create a new parameter
            parameter_string = line[line.find(" ")+1:]
            parameter = Parameter(parameter_string)
            experiment.add_parameter(parameter)
            number_parameters += 1
            
        elif field_name == "POINTS":
            # create a new point
            if number_parameters == 1:
                coordinate_string = line[line.find(" ")+1:]
                coordinate_string = coordinate_string.replace(" ", "")
                coordinate_string = coordinate_string.replace("(", "")
                coordinate_string = coordinate_string.replace(")", "")
                parameter_value = float(coordinate_string)
                parameter = experiment.get_parameter(0)
                coordinate = Coordinate()
                coordinate.add_parameter_value(parameter, parameter_value)
                experiment.add_coordinate(coordinate)
            elif number_parameters > 1 and number_parameters < 5:
                coordinate_string = line[line.find(" ")+1:]
                coordinate_string = coordinate_string.replace(" )", ";")
                coordinate_string = coordinate_string.replace("( ", "")
                coordinate_string = coordinate_string.replace(" ", ",")
                if coordinate_string[len(coordinate_string)-1] == ",":
                    coordinate_string = coordinate_string[:-1]
                coordinate_strings = coordinate_string.split(";")
                coordinate_strings.remove("")
                for i in range(len(coordinate_strings)):
                    coordinate_string = coordinate_strings[i]
                    if coordinate_string[0] == ",":
                        coordinate_string = coordinate_string[1:]
                    coordinate_strings[i] = coordinate_string
                for i in range(len(coordinate_strings)):
                    coordinate_string = coordinate_strings[i]
                    parameter_values = coordinate_string.split(",")
                    for j in range(len(parameter_values)):
                        parameter_values[j] = float(parameter_values[j])
                    coordinate = Coordinate()
                    for j in range(len(parameter_values)):
                        parameter = experiment.get_parameter(j)
                        coordinate.add_parameter_value(parameter, parameter_values[j])
                    experiment.add_coordinate(coordinate)
            else:
                logging.warning("This input format supports a maximum of 4 parameters.")
    
    # create the call tree and add it to the experiment
    callpaths = experiment.get_callpaths()
    call_tree = create_call_tree(callpaths)
    experiment.add_call_tree(call_tree)
        
    return experiment

