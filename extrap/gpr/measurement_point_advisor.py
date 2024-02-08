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
from extrap.gpr.util import identify_selection_mode


class MeasurementPointAdvisor():


    def __init__(self, budget, processes, callpaths, metric, experiment, current_cost) -> None:
        self.budget = budget
        print("budget:",budget)

        self.processes = processes
        print("processes:",processes)

        self.experiment = experiment

        self.normalization = True

        self.current_cost = current_cost

        self.parameters = []
        for i in range(len(self.experiment.parameters)):
            self.parameters.append(str(self.experiment.parameters[i]))
        print("parameters:",self.parameters)

        self.metric_str = metric
        self.metric = None
        self.metric_id = -1
        for i in range(len(self.experiment.metrics)):
            if str(self.experiment.metrics[i]) == self.metric_str:
                self.metric = self.experiment.metrics[i]
                self.metric_id = i
                break
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
