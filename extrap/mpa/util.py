# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
from __future__ import annotations

import itertools
import logging
import numbers
from collections import Counter
from collections.abc import Sequence, Iterable, Collection

import numpy as np

from extrap.entities.coordinate import Coordinate


def check_model_requirements(experiment, min_points):
    modeling_requirements_satisfied = False
    # one model parameter
    if len(experiment.parameters) == 1:
        if len(experiment.coordinates) >= min_points:
            modeling_requirements_satisfied = True

    # two model parameters
    elif len(experiment.parameters) == 2:
        x_requirement = False
        y_requirement = False

        # find the cheapest line of 5 points for y
        y_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            y = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x:
                    y.append(cord_values2[1])
            if len(y) == 5:
                if x not in y_lines:
                    y_lines[x] = y
        if len(y_lines) >= 1:
            y_requirement = True
        # print("y_lines:",y_lines)

        # find the cheapest line of 5 points for x
        x_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            y = cord_values[1]
            x = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == y:
                    x.append(cord_values2[0])
            if len(x) == 5:
                if y not in x_lines:
                    x_lines[y] = x
        if len(x_lines) >= 1:
            x_requirement = True
        # print("x_lines:",x_lines)

        if x_requirement and y_requirement:
            modeling_requirements_satisfied = True

    # three model parameters
    elif len(experiment.parameters) == 3:
        x_requirement = False
        y_requirement = False
        z_requirement = False

        # find the cheapest line of 5 points for y
        y_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            y = []
            z = cord_values[2]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x and cord_values2[2] == z:
                    y.append(cord_values2[1])
            if len(y) >= 5:
                if (x, z) not in y_lines:
                    y_lines[(x, z)] = y
        if len(y_lines) >= 1:
            y_requirement = True
        # print("y_lines:",y_lines)

        # find the cheapest line of 5 points for x
        x_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            y = cord_values[1]
            x = []
            z = cord_values[2]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == y and cord_values2[2] == z:
                    x.append(cord_values2[0])
            if len(x) >= 5:
                if (y, z) not in x_lines:
                    x_lines[(y, z)] = x
        if len(x_lines) >= 1:
            x_requirement = True
        # print("x_lines:",x_lines)

        # find the cheapest line of 5 points for z
        z_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            z = []
            y = cord_values[1]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x and cord_values2[1] == y:
                    z.append(cord_values2[2])
            if len(z) >= 5:
                if (x, y) not in z_lines:
                    z_lines[(x, y)] = z
        if len(z_lines) >= 1:
            z_requirement = True
        # print("z_lines:",z_lines)

        if x_requirement and y_requirement and z_requirement:
            modeling_requirements_satisfied = True

    # four model parameters
    elif len(experiment.parameters) == 4:
        x1_requirement = False
        x2_requirement = False
        x3_requirement = False
        x4_requirement = False

        # find the cheapest line of 5 points for x1
        x1_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = []
            x2 = cord_values[1]
            x3 = cord_values[2]
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == x2 and cord_values2[2] == x3 and cord_values2[3] == x4:
                    x1.append(cord_values2[0])
            if len(x1) >= 5:
                if (x2, x3, x4) not in x1_lines:
                    x1_lines[(x2, x3, x4)] = x1
        if len(x1_lines) >= 1:
            x1_requirement = True
        # print("x1_lines:",x1_lines)

        # find the cheapest line of 5 points for x2
        x2_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = []
            x3 = cord_values[2]
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[2] == x3 and cord_values2[3] == x4:
                    x2.append(cord_values2[1])
            if len(x2) >= 5:
                if (x1, x3, x4) not in x2_lines:
                    x2_lines[(x1, x3, x4)] = x2
        if len(x2_lines) >= 1:
            x2_requirement = True
        # print("x2_lines:",x2_lines)

        # find the cheapest line of 5 points for x3
        x3_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = cord_values[1]
            x3 = []
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[1] == x2 and cord_values2[3] == x4:
                    x3.append(cord_values2[2])
            if len(x3) >= 5:
                if (x1, x2, x4) not in x3_lines:
                    x3_lines[(x1, x2, x4)] = x3
        if len(x3_lines) >= 1:
            x3_requirement = True
        # print("x3_lines:",x3_lines)

        # find the cheapest line of 5 points for x4
        x4_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = cord_values[1]
            x3 = cord_values[2]
            x4 = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[1] == x2 and cord_values2[2] == x3:
                    x4.append(cord_values2[3])
            if len(x4) >= 5:
                if (x1, x2, x3) not in x4_lines:
                    x4_lines[(x1, x2, x3)] = x4
        if len(x4_lines) >= 1:
            x4_requirement = True
        # print("x4_lines:",x4_lines)

        if x1_requirement and x2_requirement and x3_requirement and x4_requirement:
            modeling_requirements_satisfied = True

    return modeling_requirements_satisfied


def identify_selection_mode(experiment, min_points):
    selection_mode = None

    # check if there are enough measurement points for Extra-P to create a model
    modeling_requirements_satisfied = check_model_requirements(experiment, min_points)

    # if modeling requirements are satisfied check if an additional point that is not part of the lines is existing
    # already (only for 2+ parameters)
    if modeling_requirements_satisfied:
        if len(experiment.parameters) > 1:
            additional_point_exists = check_additional_point(experiment)
            # print("DEBUG additional_point_exists:",additional_point_exists)
            # if additional point is available suggest more points using gpr method
            if additional_point_exists:
                selection_mode = "gpr"
            # if not available an additional point suggest points using add method
            else:
                selection_mode = "add"
        else:
            # TODO: or maybe later a special option if that does not work with GPR, since it was never testes before
            selection_mode = "gpr"
    # if modeling requirements are not satisfied suggest points using base method
    else:
        selection_mode = "base"

    return selection_mode


def check_additional_point(experiment):
    # two model parameters
    if len(experiment.parameters) == 2:
        # find the possible y lines
        y_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            y = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x:
                    y.append(cord_values2[1])
            if len(y) == 5:
                if x not in y_lines:
                    y_lines[x] = y
        base_cords = []
        for key, value in y_lines.items():
            for val in value:
                base_cords.append(Coordinate((key, val)))
        # find the possible x lines
        x_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            y = cord_values[1]
            x = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == y:
                    x.append(cord_values2[0])
            if len(x) == 5:
                if y not in x_lines:
                    x_lines[y] = x
        for key, value in x_lines.items():
            for val in value:
                if Coordinate((val, key)) not in base_cords:
                    base_cords.append(Coordinate((val, key)))
        additional_cord_found = False
        for i in range(len(experiment.coordinates)):
            if experiment.coordinates[i] not in base_cords:
                additional_cord_found = True
                break

        if len(x_lines) > 1 or len(y_lines) > 1:
            additional_cord_found = True

    # three model parameters
    elif len(experiment.parameters) == 3:
        # find the cheapest line of 5 points for y
        y_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            y = []
            z = cord_values[2]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x and cord_values2[2] == z:
                    y.append(cord_values2[1])
            if len(y) >= 5:
                if (x, z) not in y_lines:
                    y_lines[(x, z)] = y
        base_cords = []
        for key, value in y_lines.items():
            for val in value:
                base_cords.append(Coordinate((key[0], val, key[1])))

        # find the cheapest line of 5 points for x
        x_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            y = cord_values[1]
            x = []
            z = cord_values[2]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == y and cord_values2[2] == z:
                    x.append(cord_values2[0])
            if len(x) >= 5:
                if (y, z) not in x_lines:
                    x_lines[(y, z)] = x
        for key, value in x_lines.items():
            for val in value:
                if Coordinate((val, key[0], key[1])) not in base_cords:
                    base_cords.append(Coordinate((val, key[0], key[1])))

        # find the cheapest line of 5 points for z
        z_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            z = []
            y = cord_values[1]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x and cord_values2[1] == y:
                    z.append(cord_values2[2])
            if len(z) >= 5:
                if (x, y) not in z_lines:
                    z_lines[(x, y)] = z
        for key, value in z_lines.items():
            for val in value:
                if Coordinate((key[0], key[1], val)) not in base_cords:
                    base_cords.append(Coordinate((key[0], key[1], val)))

        additional_cord_found = False
        for i in range(len(experiment.coordinates)):
            if experiment.coordinates[i] not in base_cords:
                additional_cord_found = True
                break
        if len(y_lines) > 1 or len(x_lines) > 1 or len(z_lines) > 1:
            additional_cord_found = True

    # four model parameters
    elif len(experiment.parameters) == 4:
        # find the cheapest line of 5 points for x1
        x1_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = []
            x2 = cord_values[1]
            x3 = cord_values[2]
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == x2 and cord_values2[2] == x3 and cord_values2[3] == x4:
                    x1.append(cord_values2[0])
            if len(x1) >= 5:
                if (x2, x3, x4) not in x1_lines:
                    x1_lines[(x2, x3, x4)] = x1
        base_cords = []
        for key, value in x1_lines.items():
            for val in value:
                base_cords.append(Coordinate((val, key[0], key[1], key[2])))

        # find the cheapest line of 5 points for x2
        x2_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = []
            x3 = cord_values[2]
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[2] == x3 and cord_values2[3] == x4:
                    x2.append(cord_values2[1])
            if len(x2) >= 5:
                if (x1, x3, x4) not in x2_lines:
                    x2_lines[(x1, x3, x4)] = x2
        for key, value in x2_lines.items():
            for val in value:
                if Coordinate((key[0], val, key[1], key[2])) not in base_cords:
                    base_cords.append(Coordinate((key[0], val, key[1], key[2])))

        # find the cheapest line of 5 points for x3
        x3_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = cord_values[1]
            x3 = []
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[1] == x2 and cord_values2[3] == x4:
                    x3.append(cord_values2[2])
            if len(x3) >= 5:
                if (x1, x2, x4) not in x3_lines:
                    x3_lines[(x1, x2, x4)] = x3
        for key, value in x3_lines.items():
            for val in value:
                if Coordinate((key[0], key[1], val, key[2])) not in base_cords:
                    base_cords.append(Coordinate((key[0], key[1], val, key[2])))

        # find the cheapest line of 5 points for x4
        x4_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = cord_values[1]
            x3 = cord_values[2]
            x4 = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[1] == x2 and cord_values2[2] == x3:
                    x4.append(cord_values2[3])
            if len(x4) >= 5:
                if (x1, x2, x3) not in x4_lines:
                    x4_lines[(x1, x2, x3)] = x4
        for key, value in x4_lines.items():
            for val in value:
                if Coordinate((key[0], key[1], key[2], val)) not in base_cords:
                    base_cords.append(Coordinate((key[0], key[1], key[2], val)))

        additional_cord_found = False
        for i in range(len(experiment.coordinates)):
            if experiment.coordinates[i] not in base_cords:
                additional_cord_found = True
                break

        if len(x1_lines) > 1 or len(x2_lines) > 1 or len(x3_lines) > 1 or len(x4_lines) > 1:
            additional_cord_found = True

    return additional_cord_found


def build_parameter_value_series(coordinates: Sequence[Coordinate]) -> list[list[numbers.Real]]:
    """Get the parameter value series of each parameter that exist from the existing measurement points.

    :return: A list containing one series per parameter. The series are ordered in ascending order.
    """
    if not coordinates:
        return []
    parameter_value_series = [[] for _ in coordinates[0]]
    for coordinate in coordinates:
        parameter_values = coordinate.as_tuple()
        for j, value in enumerate(parameter_values):
            if value not in parameter_value_series[j]:
                parameter_value_series[j].append(value)
    for p in parameter_value_series:
        p.sort()  # ensure order
        # print("DEBUG parameter_values:",parameter_values)
    # print("DEBUG parameter_value_series:",parameter_value_series)
    return parameter_value_series


def identify_step_factor(parameter_value_series: Sequence[Sequence[numbers.Real]]) \
        -> list[tuple[str, numbers.Real]]:
    """Get the step value factor for each parameter value series"""
    median_step_size_factors = []
    for series in parameter_value_series:
        if len(series) == 0:
            median_step_size_factors.append(("+", 1))
            continue
        if len(series) == 1:
            median_step_size_factors.append(("*", 2))
            continue

        factors = []
        steps = []
        for j in range(len(series) - 1):
            factors.append(series[j + 1] / series[j])
            steps.append(series[j + 1] - series[j])

        counted_factors = Counter(factors)
        counted_steps = Counter(steps)
        factor_max = counted_factors.most_common(1)[0][1]
        steps_max = counted_steps.most_common(1)[0][1]
        logging.debug("steps_dict: %s step_maximum: %s", counted_steps, steps_max)
        logging.debug("factor_dict: %s factor_maximum: %s", counted_steps, steps_max)

        if factor_max > steps_max:
            median_step_size_factors.append(("*", np.median(factors)))
        elif steps_max > factor_max:
            median_step_size_factors.append(("+", np.median(steps)))
        else:
            all_steps_same = counted_steps[steps[0]] == len(steps)
            if all_steps_same:
                median_step_size_factors.append(("+", np.median(steps)))
            else:
                facts = []
                for i in range(len(factors) - 1):
                    if factors[i + 1] % factors[0] == 0:
                        facts.append(factors[0])
                    else:
                        facts.append(factors[i + 1])
                all_facts_same = True
                for i in range(len(facts) - 1):
                    if facts[0] != facts[i + 1]:
                        all_facts_same = False
                        break
                if all_facts_same:
                    median_step_size_factors.append(("*", np.median(facts)))
                else:
                    median_step_size_factors.append(("+", np.median(steps)))
    logging.debug("median_step_size_factor: %s", median_step_size_factors)
    return median_step_size_factors


def extend_parameter_value_series(parameter_value_series, mean_step_size_factors, additional_values=5):
    """
    Continue and complete the parameter value series for each parameter

    NOTE: This search space with 5 additional values is large enough especially for 4 model parameters this results
    in thousands of possible points as soon as additional points are measured and loaded into extra-p, the search
    space will be extended using the new values as a baseline anyway.
    """
    for series, (operator, step) in zip(parameter_value_series, mean_step_size_factors):
        if not series:
            continue
        added_values = 0
        for j in range(len(series)):
            if operator == "*":
                new_value = series[j] * step
            elif operator == "+":
                new_value = series[j] + step
            else:
                raise ValueError('Operator must be either "+" or "*"')
            if new_value not in series:
                series.append(new_value)
                added_values += 1
        if added_values < additional_values:
            for j in range(additional_values - added_values):
                if operator == "*":
                    new_value = series[len(series) - 1] * step
                elif operator == "+":
                    new_value = series[len(series) - 1] + step
                else:
                    raise ValueError('Operator must be either "+" or "*"')
                if new_value not in series:
                    series.append(new_value)
                    added_values += 1
        series.sort()
    # print("DEBUG parameter_value_serieses:",parameter_value_serieses)
    return parameter_value_series


def get_search_space_generator(parameter_value_series):
    """Create search space (1D, 2D, 3D, ND) from the series values of each parameter

    :return: Returns a generator for the search space
    """

    return (Coordinate(*c) for c in itertools.product(*parameter_value_series))


def identify_possible_points(search_space_coordinates: Iterable[Coordinate], coordinates: Collection[Coordinate]):
    possible_points = [c for c in search_space_coordinates if c not in coordinates]
    # print("DEBUG len() search_space_coordinates:",len(search_space_coordinates))
    # print("DEBUG len() possible_points:",len(possible_points))
    # print("DEBUG len() experiment.coordinates:",len(experiment.coordinates))
    # print("possible_points:",possible_points)
    return possible_points
