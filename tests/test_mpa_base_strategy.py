# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

from extrap.entities.coordinate import Coordinate
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.mpa.base_selection_strategy import suggest_points_base_mode


class TestMPABaseStrategy(unittest.TestCase):
    def test_one_parameter(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_1.txt")

        parameter_value_series = [[20, 30, 40, 50, 60]]
        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([], result)

        experiment.coordinates.remove(Coordinate(60))

        parameter_value_series = [[20, 30, 40, 50, 60]]
        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([Coordinate(60)], result)

        parameter_value_series = [[20, 30, 40, 50]]
        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([], result)

        experiment.coordinates = [experiment.coordinates[0]]

        parameter_value_series = [[20, 30, 40, 50, 60]]
        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([Coordinate(30), Coordinate(40), Coordinate(50), Coordinate(60)], result)

        parameter_value_series = [[20, 30, 40, 50]]
        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([Coordinate(30), Coordinate(40), Coordinate(50)], result)

    def test_two_parameter(self):
        experiment = TextFileReader().read_experiment("data/text/two_parameter_1.txt")

        parameter_value_series = [[20, 30, 40, 50, 60], [1, 2, 3, 4, 5]]
        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([], result)

        experiment.coordinates = experiment.coordinates[0:5]

        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([Coordinate(30, 1), Coordinate(40, 1), Coordinate(50, 1), Coordinate(60, 1)], result)

        experiment.coordinates = experiment.coordinates[0:2]

        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([Coordinate(30, 1), Coordinate(40, 1), Coordinate(50, 1), Coordinate(60, 1),
                          Coordinate(20, 3), Coordinate(20, 4), Coordinate(20, 5)], result)

        experiment.coordinates = experiment.coordinates[0:1]

        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([Coordinate(30, 1), Coordinate(40, 1), Coordinate(50, 1), Coordinate(60, 1),
                          Coordinate(20, 2), Coordinate(20, 3), Coordinate(20, 4), Coordinate(20, 5)], result)

    def test_three_parameter(self):
        experiment = TextFileReader().read_experiment("data/text/three_parameter_1.txt")

        parameter_value_series = [[20, 30, 40, 50, 60], [1, 2, 3, 4, 5], [100, 200, 300, 400, 500]]
        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([], result)

        experiment.coordinates = experiment.coordinates[0:5]

        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([
            Coordinate(30, 1.0, 100.0),
            Coordinate(40, 1.0, 100.0),
            Coordinate(50, 1.0, 100.0),
            Coordinate(60, 1.0, 100.0),
            Coordinate(20.0, 1.0, 200),
            Coordinate(20.0, 1.0, 300),
            Coordinate(20.0, 1.0, 400),
            Coordinate(20.0, 1.0, 500)], result)

        experiment.coordinates = experiment.coordinates[0:2]

        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([
            Coordinate(30, 1.0, 100.0),
            Coordinate(40, 1.0, 100.0),
            Coordinate(50, 1.0, 100.0),
            Coordinate(60, 1.0, 100.0),
            Coordinate(20.0, 3.0, 100),
            Coordinate(20.0, 4.0, 100),
            Coordinate(20.0, 5.0, 100),
            Coordinate(20.0, 1.0, 200),
            Coordinate(20.0, 1.0, 300),
            Coordinate(20.0, 1.0, 400),
            Coordinate(20.0, 1.0, 500)], result)

        experiment.coordinates = experiment.coordinates[0:1]

        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([
            Coordinate(30, 1.0, 100.0),
            Coordinate(40, 1.0, 100.0),
            Coordinate(50, 1.0, 100.0),
            Coordinate(60, 1.0, 100.0),
            Coordinate(20.0, 2.0, 100),
            Coordinate(20.0, 3.0, 100),
            Coordinate(20.0, 4.0, 100),
            Coordinate(20.0, 5.0, 100),
            Coordinate(20.0, 1.0, 200),
            Coordinate(20.0, 1.0, 300),
            Coordinate(20.0, 1.0, 400),
            Coordinate(20.0, 1.0, 500)], result)

    def test_two_parameter_unordered(self):
        experiment = TextFileReader().read_experiment("data/text/two_parameter_1.txt")

        co = experiment.coordinates
        experiment.coordinates = [co[3], co[4], co[2], co[1], co[0]] + co[5:]

        parameter_value_series = [[20, 30, 40, 50, 60], [1, 2, 3, 4, 5]]
        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([], result)

        experiment.coordinates = experiment.coordinates[0:5]

        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([Coordinate(30, 1), Coordinate(40, 1), Coordinate(50, 1), Coordinate(60, 1)], result)

        experiment.coordinates = experiment.coordinates[0:2]

        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([Coordinate(30, 4), Coordinate(40, 4), Coordinate(50, 4), Coordinate(60, 4),
                          Coordinate(20, 1), Coordinate(20, 2), Coordinate(20, 3)], result)

        experiment.coordinates = experiment.coordinates[0:1]

        result = suggest_points_base_mode(experiment, parameter_value_series)
        self.assertEqual([Coordinate(30, 4), Coordinate(40, 4), Coordinate(50, 4), Coordinate(60, 4),
                          Coordinate(20, 1), Coordinate(20, 2), Coordinate(20, 3), Coordinate(20, 5)], result)
