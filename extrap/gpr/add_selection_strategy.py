# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

def suggest_points_add_mode(experiment, possible_points, selected_callpaths, metric, calculate_cost_manual, process_parameter_id, number_processes, budget, current_cost):
    
    # sum up the values from the selected callpaths
    if len(selected_callpaths) > 1:
        runtimes = {}
        for i in range(len(selected_callpaths)):
            callpath = selected_callpaths[i]
            modeler = experiment.modelers[0]
            model = modeler.models[callpath, metric]
            hypothesis = model.hypothesis
            function = hypothesis.function

            for j in range(len(possible_points)):
                point = possible_points[j]
                # b.1 predict the runtime of the possible_points using the existing performance models
                runtime = function.evaluate(point.as_tuple())
                if i == 0:
                    runtimes[point] = runtime
                else:
                    runtimes[point] += runtime

        costs = {}
        for i in range(len(possible_points)):
            point = possible_points[i]
            runtime = runtimes[point]
            # b.2 calculate the cost of these points using the runtime (same calculation as for the current cost in the GUI)
            if calculate_cost_manual:
                nr_processes = number_processes
            else:
                nr_processes = point[process_parameter_id]
            cost = runtime * nr_processes
            costs[point] = cost

        # b.3 choose n points from the seach space with the lowest cost and check if they fit in the available budget
        costs_sorted = dict(sorted(costs.items(), key=lambda item: item[1]))
        available_budget = budget-current_cost
        suggested_points = []

        # suggest the coordinate if it fits into budget
        for key, value in costs_sorted.items():
            if value <= available_budget:
                suggested_points.append(key)
                available_budget -= value
            else:
                break

    # use the values from the selected callpath
    elif len(selected_callpaths) == 1:
        callpath = selected_callpaths[0]

        modeler = experiment.modelers[0]
        model = modeler.models[callpath, metric]
        hypothesis = model.hypothesis
        function = hypothesis.function
        runtimes = {}
        costs = {}
        
        for j in range(len(possible_points)):
            point = possible_points[j]
            # b.1 predict the runtime of the possible_points using the existing performance models
            runtime = function.evaluate(point.as_tuple())
            runtimes[point] = runtime
            # b.2 calculate the cost of these points using the runtime (same calculation as for the current cost in the GUI)
            if calculate_cost_manual:
                nr_processes = number_processes
            else:
                nr_processes = point[process_parameter_id]
            cost = runtime * nr_processes
            costs[point] = cost
            
        # b.3 choose n points from the seach space with the lowest cost and check if they fit in the available budget
        costs_sorted = dict(sorted(costs.items(), key=lambda item: item[1]))
        available_budget = budget-current_cost
        suggested_points = []

        # suggest the coordinate if it fits into budget
        for key, value in costs_sorted.items():
            if value <= available_budget:
                suggested_points.append(key)
                available_budget -= value
            else:
                break

    # sum all callpaths runtime
    elif len(selected_callpaths) == 0:
        runtimes = {}
        for i in range(len(experiment.callpaths)):
            callpath = experiment.callpaths[i]
            modeler = experiment.modelers[0]
            model = modeler.models[callpath, metric]
            hypothesis = model.hypothesis
            function = hypothesis.function

            for j in range(len(possible_points)):
                point = possible_points[j]
                # b.1 predict the runtime of the possible_points using the existing performance models
                runtime = function.evaluate(point.as_tuple())
                if i == 0:
                    runtimes[point] = runtime
                else:
                    runtimes[point] += runtime

        costs = {}
        for i in range(len(possible_points)):
            point = possible_points[i]
            runtime = runtimes[point]
            # b.2 calculate the cost of these points using the runtime (same calculation as for the current cost in the GUI)
            if calculate_cost_manual:
                nr_processes = number_processes
            else:
                nr_processes = point[process_parameter_id]
            cost = runtime * nr_processes
            costs[point] = cost

        # b.3 choose n points from the seach space with the lowest cost and check if they fit in the available budget
        costs_sorted = dict(sorted(costs.items(), key=lambda item: item[1]))
        available_budget = budget-current_cost
        suggested_points = []

        # suggest the coordinate if it fits into budget
        for key, value in costs_sorted.items():
            if value <= available_budget:
                suggested_points.append(key)
                available_budget -= value
            else:
                break

    return suggested_points