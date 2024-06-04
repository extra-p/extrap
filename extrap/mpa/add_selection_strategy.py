# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
from __future__ import annotations

import numbers
from collections.abc import Callable


def suggest_points_add_mode(experiment, possible_points, selected_callpaths, metric,
                            calculate_cost: Callable[[tuple, numbers.Real], numbers.Real],
                            budget, current_cost, model_generator):
    """
    Suggest points using add mode

    1. Predict the runtime of the possible_points using the existing performance models
    2. Calculate the cost of these points using the runtime (same calculation as for the current cost in the GUI)
    3. Choose the point from the search space with the lowest cost
    4. Check if that point fits into the available budget
    4.1. Create a coordinate from it and suggest it if fits into budget
    4.2. If not fit then need to show message instead that available budget is not sufficient and needs to be increased...
    """
    runtimes = {}
    # sum up the values from the selected callpaths
    for callpath in selected_callpaths:

        model = model_generator.models[callpath, metric]
        function = model.hypothesis.function

        for j, point in enumerate(possible_points):
            # b.1 predict the runtime of the possible_points using the existing performance models
            runtime = function.evaluate(point.as_tuple())
            if point not in runtimes:
                runtimes[point] = runtime
            else:
                runtimes[point] += runtime

    costs = {}
    for point in possible_points:
        runtime = runtimes[point]
        # b.2 calculate the cost of these points using the runtime (same calculation as for the current cost in the GUI)
        cost = calculate_cost(point, runtime)
        costs[point] = cost

    # b.3 choose n points from the seach space with the lowest cost and check if they fit in the available budget
    costs_sorted = sorted(costs.items(), key=lambda item: item[1])
    available_budget = budget - current_cost
    suggested_points = []

    # suggest the coordinate if it fits into budget
    for key, value in costs_sorted:
        if value <= available_budget:
            suggested_points.append(key)
            available_budget -= value
        else:
            break

    return suggested_points
