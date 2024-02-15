# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
from extrap.gpr.util import identify_selection_mode, build_parameter_value_series, identify_step_factor, extend_parameter_value_series, build_search_space, identify_possible_points
from extrap.gpr.base_selection_strategy import suggest_points_base_mode
from extrap.gpr.add_selection_strategy import suggest_points_add_mode
from extrap.gpr.gpr_selection_strategy import suggest_points_gpr_mode


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




