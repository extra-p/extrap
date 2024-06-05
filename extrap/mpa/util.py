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


def find_lines(coordinates, p):
    lines = {}
    for coordinate in coordinates:
        other_values = coordinate.as_partial_tuple(p)
        param_values = []
        for coordinate2 in coordinates:
            if coordinate.is_mostly_equal(coordinate2, p):
                param_values.append(coordinate2[p])
        if other_values not in lines:
            lines[other_values] = param_values
    return lines


def check_model_requirements(num_parameters, coordinates, min_points):
    modeling_requirements_satisfied = False
    # one model parameter
    if num_parameters == 1:
        if len(coordinates) >= min_points:
            modeling_requirements_satisfied = True
    else:
        line_length_requirement = [False] * num_parameters
        for p in range(num_parameters):
            lines = find_lines(coordinates, p)
            if any(len(line) >= min_points for line in lines.values()):
                line_length_requirement[p] = True

        modeling_requirements_satisfied = all(line_length_requirement)

    return modeling_requirements_satisfied


def identify_selection_mode(experiment, min_points):
    selection_mode = None

    coordinates = sorted(experiment.coordinates)

    # check if there are enough measurement points for Extra-P to create a model
    modeling_requirements_satisfied = check_model_requirements(len(experiment.parameters), coordinates, min_points)

    # if modeling requirements are satisfied check if an additional point that is not part of the lines is existing
    # already (only for 2+ parameters)
    if modeling_requirements_satisfied:
        if len(experiment.parameters) > 1:
            additional_point_exists = check_additional_point(len(experiment.parameters), coordinates, min_points)
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


def check_additional_point(num_parameters, coordinates, min_points):
    base_cords = []
    for p in range(num_parameters):
        lines = find_lines(coordinates, p)

        for key, value in lines.items():
            if len(value) != min_points:
                continue
            for val in value:
                base_cords.append(Coordinate(*key[:p], val, *key[p:]))

    additional_cord_found = False
    for coordinate in coordinates:
        if coordinate not in base_cords:
            additional_cord_found = True
            break

    # if len(x1_lines) > 1 or len(x2_lines) > 1 or len(x3_lines) > 1 or len(x4_lines) > 1:
    #     additional_cord_found = True

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
