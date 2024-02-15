# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from multiprocessing import Manager
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import copy
from extrap.gpr.util import identify_selection_mode, build_parameter_value_series, identify_step_factor, extend_parameter_value_series, build_search_space, identify_possible_points, suggest_points_base_mode, suggest_points_add_mode, suggest_points_gpr_mode
from extrap.entities.coordinate import Coordinate
from collections import Counter


class MeasurementPointAdvisor():

    def __init__(self, budget, processes, callpaths, metric, experiment, current_cost, manual_pms_selection, manual_parameter_value_series, calculate_cost_manual, number_processes) -> None:
        self.budget = budget
        print("budget:",budget)

        self.processes = processes
        print("processes:",processes)

        self.experiment = experiment

        self.normalization = True

        self.current_cost = current_cost

        self.manual_pms_selection = manual_pms_selection

        self.manual_parameter_value_series = manual_parameter_value_series

        self.calculate_cost_manual = calculate_cost_manual

        self.number_processes = number_processes

        self.parameters = []
        for i in range(len(self.experiment.parameters)):
            self.parameters.append(str(self.experiment.parameters[i]))
        print("parameters:",self.parameters)

        self.metric = metric
        print("metric:",self.metric)

        # these are tree nodes, need to convert them to actual callpaths manually
        self.selected_callpath_ids = []
        self.selected_callpaths = []
        for i in range(len(callpaths)):
            self.selected_callpaths.append(callpaths[i].path)
            for j in range(len(self.experiment.callpaths)):
                if str(callpaths[i].path) == str(self.experiment.callpaths[j]):
                    self.selected_callpath_ids.append(j)
                    break
        print("selected callpaths:",self.selected_callpaths)
        print("selected callpath ids:",self.selected_callpath_ids)

        # set the minimum number of points required for modeling with the sparse modeler
        min_points = 0
        if len(self.experiment.parameters) == 1:
            min_points = 5
        elif len(self.experiment.parameters) == 2:
            min_points = 9
        elif len(self.experiment.parameters) == 3:
            min_points = 13
        elif len(self.experiment.parameters) == 4:
            min_points = 17
        else:
            min_points = 5

        # identify the state of the selection process
        # possible states are:
        # 1. not enough points for modeling 
        #   -> continue row of parameter values for all parameters until modeling requirements are reached
        # 2. enough points for modeling, without an additional point not part of the lines
        #   -> suggest an extra point not part of the lines for each parameter
        # 3. enough points for modeling with an additional point not part of the lines
        #   -> suggest additional points using the gpr method
        
        # can be: gpr, add, base
        # gpr -> suggest additional measurement points with the gpr method
        # add -> suggest an additional point that is not part of the lines for each parameter
        # base -> suggest points to complete the lines of points for each parameter
        self.selection_mode = identify_selection_mode(self.experiment, min_points)

        print("DEBUG selection_mode:",self.selection_mode)


    def suggest_points(self):

        if self.manual_pms_selection:
            # 1.2.3. build the parameter series from manual entries in GUI
            try:
                parameter_value_series = []
                for i in range(len(self.experiment.parameters)):
                    value_series = self.manual_parameter_value_series[i].split(",")
                    y = []
                    for x in value_series:
                        y.append(float(x))
                    parameter_value_series.append(y)
            except ValueError as e:
                print(e)
                return []

        else:
            # 1. build a value series for each parameter
            parameter_value_series = build_parameter_value_series(self.experiment)

            # 2. identify the step factor size for each parameter
            mean_step_size_factors = identify_step_factor(parameter_value_series)

            # 3. continue and complete these series for each parameter
            parameter_value_series = extend_parameter_value_series(parameter_value_series, mean_step_size_factors)

        # 4. create search space (1D, 2D, 3D, ND) from the series values of each parameters
        search_space_coordinates = build_search_space(self.experiment, parameter_value_series)

        # 5. remove existing points from search space to obtain only new possible points
        possible_points = identify_possible_points(search_space_coordinates, self.experiment)
                
        # 6. suggest points using selected mode
        # a. base mode
        # a.1 choose the smallest of the values for each parameter
        # a.2 combine these values of each parameter to a coordinate
        # a.3 repeat until enough suggestions for cords to complete a line of 5 points for each parameter
        if self.selection_mode == "base":
            suggested_cords = suggest_points_base_mode(self.experiment, parameter_value_series)
            return suggested_cords
        
        # 6. suggest points using add mode
        # b. add mode
        # b.1 predict the runtime of the possible_points using the existing performance models
        # b.2 calculate the cost of these points using the runtime (same calculation as for the current cost in the GUI)
        # b.3 choose the point from the seach space with the lowest cost
        # b.4 check if that point fits into the available budget
        # b.41 create a coordinate from it and suggest it if fits into budget
        # b.42 if not fit then need to show message instead that available budget is not sufficient and needs to be increased...
        elif self.selection_mode == "add":
            suggested_cords = suggest_points_add_mode(self.experiment, possible_points, self.selected_callpaths, self.metric, self.calculate_cost_manual, self.processes, self.number_processes, self.budget, self.current_cost)
            return suggested_cords

        
        # 6. suggest points using gpr mode
        # c. gpr mode
        # c.1 predict the runtime of these points using the existing performance models (only possible if already enough points existing for modeling)
        # c.2 calculate the cost of these points using the runtime (same calculation as for the current cost in the GUI)
        #NOTE: the search space points should have a dict like for the costs of the remaining points for my case study analysis...
        # c.3 all of the data is used as input to the GPR method
        # c.4 get the top x points suggested by the GPR method that do fit into the available budget
        # c.5 create coordinates and suggest them
        elif self.selection_mode == "gpr":
            suggested_cords = suggest_points_gpr_mode(self.experiment, parameter_value_series)
            return suggested_cords
        

        
        

        return [(0,0),(1,1)]


        """#TODO: if modeling requirements are not satisfied 
        # suggest points to complete the rows of parameter values
        else:
            if len(self.experiment.parameters) == 1:
                pass
            elif len(self.experiment.parameters) == 2:
                pass
            elif len(self.experiment.parameters) == 3:
                pass
            elif len(self.experiment.parameters) == 4:
                pass"""

        




        # 2. enough points for modeling
        # ...



        cost_container = {}
        total_costs_container = {}

        modeler = self.experiment.modelers[0]

        # calculate the overall runtime of the application and the cost of each kernel per measurement point
        for callpath_id in range(len(experiment.callpaths)):
            callpath = experiment.callpaths[callpath_id]
            callpath_string = callpath.name

            cost = {}
            total_cost = 0
            
            try:
                model = modeler.models[callpath, metric]
            except KeyError:
                model = None
            if model != None:
                hypothesis = model.hypothesis
                function = hypothesis.function
                
                # get the extrap function as a string
                #function_string = function.to_string(*experiment.parameters)
                #function_string = get_eval_string(function_string)

                overall_runtime = 0
                for i in range(len(experiment.coordinates)):
                    if experiment.coordinates[i] not in cost:
                        cost[experiment.coordinates[i]] = []
                    values = experiment.coordinates[i].as_tuple()
                    nr_processes = values[processes]
                    coordinate_id = -1
                    for k in range(len(experiment.coordinates)):
                        if experiment.coordinates[i] == experiment.coordinates[k]:
                            coordinate_id = k
                    measurement_temp = experiment.get_measurement(coordinate_id, callpath_id, self.metric_id)
                    coordinate_cost = 0
                    if measurement_temp != None:
                        for k in range(len(measurement_temp.values)):
                            runtime = np.mean(measurement_temp.values[k])
                            core_hours = runtime * nr_processes
                            cost[experiment.coordinates[i]].append(core_hours)
                            coordinate_cost += core_hours
                            overall_runtime += runtime
                    total_cost += coordinate_cost

            else:
                #function_string = "None"
                total_cost = 0
                overall_runtime = 0

            cost_container[callpath_string] = cost
            total_costs_container[callpath_string] = total_cost

            #runtime_sums[callpath_string] = overall_runtime


        manager = Manager()
        shared_dict = manager.dict()
        cpu_count = mp.cpu_count()
        cpu_count -= 2
        if len(self.selected_callpaths) < cpu_count:
            cpu_count = len(self.selected_callpaths)

        inputs = []
        for i in range(len(self.selected_callpath_ids)):
            inputs.append([self.selected_callpath_ids[i], shared_dict, cost, 
                           self.selected_callpaths[i], cost_container, total_costs_container, 
                           self.experiment.measurements, len(self.experiment.parameters),
                           self.experiment.coordinates, self.metric,
                           self.metric_id, self.parameters,
                           self.budget,
                           self.normalization,
                           min_points])
            
        #with Pool(cpu_count) as pool:
        #    _ = list(tqdm(pool.imap(self.analyze_callpath, inputs), total=len(self.selected_callpath_ids), disable=False))

        #result_dict = copy.deepcopy(shared_dict)
        
        #print("DEBUG:",result_dict)


    def analyze_callpath(self, inputs):
        
        # get the values from the parallel input dict
        callpath_id = inputs[0]
        shared_dict = inputs[1]
        cost = inputs[2]
        callpath = inputs[3]
        cost_container = inputs[4]
        total_costs_container = inputs[5]
        #grid_search = inputs[6]
        experiment_measurements = inputs[6]
        nr_parameters = inputs[7]
        experiment_coordinates = inputs[8]
        metric = inputs[9]
        #base_values = inputs[11]
        metric_id = inputs[10]
        #nr_repetitions = inputs[13]
        parameters = inputs[11]
        #args = inputs[15]
        budget = inputs[12]
        #eval_point = inputs[17]
        #all_points_functions_strings = inputs[18]
        #coordinate_evaluation = inputs[19]
        #measurement_evaluation = inputs[20]
        normalization = inputs[13]
        min_points = inputs[14]
        #hybrid_switch = inputs[23]
        result_container = {}

        callpath_string = callpath.name

        # get the cost values for this particular callpath
        cost = cost_container[callpath_string]
        total_cost = total_costs_container[callpath_string]

        # create copy of the cost dict
        remaining_points = copy.deepcopy(cost)

        ##########################
        ## Base point selection ##
        ##########################

        # create copy of the cost dict
        remaining_points = copy.deepcopy(cost)
        
        # create copy of the cost dict for the minimum experiment with gpr and hybrid strategies
        remaining_points_min = copy.deepcopy(cost)

        measurements_gpr = copy.deepcopy(experiment_measurements)
        measurements_hybrid = copy.deepcopy(experiment_measurements)



        shared_dict[callpath_id] = result_container




    #NOTE: base_values, always use all available reps of a point!

    #NOTE: the number of available repetitions needs to be calculated for each
    # individual measurement point

    """# identify the number of repetitions per measurement point
    nr_repetitions = 1
    measurements = self.experiment.measurements
    try:
        nr_repetitions = len(measurements[(selected_callpath[0].path, runtime_metric)].values)
    except TypeError:
        pass
    except KeyError:
        pass"""
