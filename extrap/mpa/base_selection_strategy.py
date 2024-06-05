# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from extrap.entities.coordinate import Coordinate
from extrap.mpa.util import find_lines


def suggest_points_base_mode(experiment, parameter_value_series, total_num_points_needed=5) -> list[Coordinate]:
    """
    Suggest points using the base mode

    1. Chooses the smallest of the values for each parameter
    2. Combines these values of each parameter to a coordinate
    3. Repeats until enough suggestions for cords to complete a line of 5 points for each parameter
    """

    coordinates = sorted(experiment.coordinates)
    suggested_cords = []
    for p, _ in enumerate(experiment.parameters):
        lines = find_lines(coordinates, p)

        max_value = 0
        best_line_key = None
        for key, value in lines.items():
            line_length = len(value)
            if line_length > max_value:
                best_line_key = key
                max_value = line_length
        best_line = lines[best_line_key]
        points_needed = total_num_points_needed - max_value

        potential_values = [p_value for p_value in parameter_value_series[p] if p_value not in best_line]
        potential_values.sort()

        for i in range(min(points_needed, len(potential_values))):
            suggested_cords.append(Coordinate(*best_line_key[:p], potential_values[i], *best_line_key[p:]))

    return suggested_cords
