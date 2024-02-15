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


#TODO: finish this code!
def suggest_points_gpr_mode(experiment, parameter_value_series):
    pass
    # c.1 predict the runtime of these points using the existing performance models (only possible if already enough points existing for modeling)
    # c.2 calculate the cost of these points using the runtime (same calculation as for the current cost in the GUI)
    #NOTE: the search space points should have a dict like for the costs of the remaining points for my case study analysis...
    # c.3 all of the data is used as input to the GPR method
    # c.4 get the top x points suggested by the GPR method that do fit into the available budget
    # c.5 create coordinates and suggest them

    suggested_points = []
    return suggested_points

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


