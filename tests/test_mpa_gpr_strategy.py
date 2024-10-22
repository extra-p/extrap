# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import math
import unittest

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.metric import Metric
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.modelers.model_generator import ModelGenerator
from extrap.mpa.gpr_selection_strategy import suggest_points_gpr_mode, get_cost_of_measurements, MeanRepPair


class TestMpaGprStrategy(unittest.TestCase):
    def test_suggest_points_gpr_mode(self):
        file_reader = TextFileReader()
        file_reader.keep_values = True
        experiment = file_reader.read_experiment("data/text/one_parameter_1.txt")
        model_generator = ModelGenerator(experiment)
        model_generator.model_all()

        selected_callpath = [Callpath('compute')]
        metric = Metric('time')

        def calculate_cost(p, t):
            return t

        budget = 10 ** 5
        current_cost = 10864.16

        existing_measurements_dict = {}
        for measurement in experiment.measurements[(Callpath('compute'), metric)]:
            # prepare dictionary of existing measurements
            if measurement.coordinate not in existing_measurements_dict:
                existing_measurements_dict[measurement.coordinate] = MeanRepPair(measurement.mean,
                                                                                 measurement.repetitions)
            else:
                existing_measurements_dict[measurement.coordinate] += MeanRepPair(measurement.mean,
                                                                                  measurement.repetitions)

        cost = get_cost_of_measurements(calculate_cost, existing_measurements_dict)
        self.assertAlmostEqual(current_cost, cost, places=2)

        possible_points = [Coordinate(70)]
        suggested_points, rep_numbers = suggest_points_gpr_mode(experiment, possible_points, selected_callpath, metric,
                                                                calculate_cost, budget,
                                                                current_cost, model_generator, random_state=0)

        self.assertListEqual([Coordinate(70)] * 5, suggested_points)

        possible_points = [Coordinate(70), Coordinate(80)]
        suggested_points, rep_numbers = suggest_points_gpr_mode(experiment, possible_points, selected_callpath, metric,
                                                                calculate_cost, budget,
                                                                current_cost, model_generator, random_state=0)

        self.assertListEqual([Coordinate(80), Coordinate(70)], suggested_points[:2])

    def test_suggest_points_gpr_mode_without_values(self):
        file_reader = TextFileReader()
        experiment = file_reader.read_experiment("data/text/one_parameter_1.txt")
        model_generator = ModelGenerator(experiment)
        model_generator.model_all()

        selected_callpath = [Callpath('compute')]
        metric = Metric('time')

        def calculate_cost(p, t):
            return t

        budget = 10 ** 5
        current_cost = 10864.16

        existing_measurements_dict = {}
        for measurement in experiment.measurements[(Callpath('compute'), metric)]:
            # prepare dictionary of existing measurements
            if measurement.coordinate not in existing_measurements_dict:
                existing_measurements_dict[measurement.coordinate] = MeanRepPair(measurement.mean,
                                                                                 measurement.repetitions)
            else:
                existing_measurements_dict[measurement.coordinate] += MeanRepPair(measurement.mean,
                                                                                  measurement.repetitions)

        cost = get_cost_of_measurements(calculate_cost, existing_measurements_dict)
        self.assertAlmostEqual(current_cost, cost, places=2)

        possible_points = [Coordinate(70)]
        suggested_points, rep_numbers = suggest_points_gpr_mode(experiment, possible_points, selected_callpath, metric,
                                                                calculate_cost, budget,
                                                                current_cost, model_generator, random_state=0)

        self.assertListEqual([Coordinate(70)] * 5, suggested_points)

        possible_points = [Coordinate(70), Coordinate(80)]
        suggested_points, rep_numbers = suggest_points_gpr_mode(experiment, possible_points, selected_callpath, metric,
                                                                calculate_cost, budget,
                                                                current_cost, model_generator, random_state=0)

        self.assertListEqual([Coordinate(80), Coordinate(70)], suggested_points[:2])

        possible_points = [Coordinate(70), Coordinate(80), Coordinate(110)]
        suggested_points, rep_numbers = suggest_points_gpr_mode(experiment, possible_points, selected_callpath, metric,
                                                                calculate_cost, budget,
                                                                current_cost, model_generator, random_state=0)

        self.assertListEqual([Coordinate(110), Coordinate(70), Coordinate(80)], suggested_points[:3])

    def test_one_param_multiple_callpath(self):
        file_reader = TextFileReader()
        file_reader.keep_values = True
        experiment = file_reader.read_experiment("data/text/one_parameter_5.txt")
        model_generator = ModelGenerator(experiment)
        model_generator.model_all()

        selected_callpath = [Callpath('merge'), Callpath('sort')]
        metric = Metric('time')

        def calculate_cost(p, t):
            return t

        budget = 5 * 10 ** 5
        current_cost = 18099.36

        existing_measurements_dict = {}
        for callpath in selected_callpath:
            for measurement in experiment.measurements[(callpath, metric)]:
                # prepare dictionary of existing measurements
                if measurement.coordinate not in existing_measurements_dict:
                    existing_measurements_dict[measurement.coordinate] = MeanRepPair(measurement.mean,
                                                                                     measurement.repetitions)
                else:
                    existing_measurements_dict[measurement.coordinate] += MeanRepPair(measurement.mean,
                                                                                      measurement.repetitions)

        cost = get_cost_of_measurements(calculate_cost, existing_measurements_dict)
        self.assertAlmostEqual(current_cost, cost, places=2)

        possible_points = [Coordinate(70)]
        suggested_points, rep_numbers = suggest_points_gpr_mode(experiment, possible_points, selected_callpath, metric,
                                                                calculate_cost, budget,
                                                                current_cost, model_generator, random_state=0)

        self.assertListEqual([Coordinate(70)] * 5, suggested_points)

        possible_points = [Coordinate(70), Coordinate(80)]
        suggested_points, rep_numbers = suggest_points_gpr_mode(experiment, possible_points, selected_callpath, metric,
                                                                calculate_cost, budget,
                                                                current_cost, model_generator, random_state=0)

        self.assertListEqual([Coordinate(70), Coordinate(80)], suggested_points[:2])

    def test_suggest_points_gpr_mode_multi_parameter_without_values(self):
        file_reader = TextFileReader()
        experiment = file_reader.read_experiment("data/text/two_parameter_1.txt")
        model_generator = ModelGenerator(experiment)
        model_generator.model_all()

        selected_callpath = [Callpath('reg')]
        metric = Metric('metr')

        def calculate_cost(p, t):
            return t

        budget = math.inf
        current_cost = 62299.14

        existing_measurements_dict = {}
        for measurement in experiment.measurements[(selected_callpath[0], metric)]:
            # prepare dictionary of existing measurements
            if measurement.coordinate not in existing_measurements_dict:
                existing_measurements_dict[measurement.coordinate] = MeanRepPair(measurement.mean,
                                                                                 measurement.repetitions)
            else:
                existing_measurements_dict[measurement.coordinate] += MeanRepPair(measurement.mean,
                                                                                  measurement.repetitions)

        cost = get_cost_of_measurements(calculate_cost, existing_measurements_dict)
        self.assertAlmostEqual(current_cost, cost, places=2)

        possible_points = [Coordinate(70, 6)]
        suggested_points, rep_numbers = suggest_points_gpr_mode(experiment, possible_points, selected_callpath, metric,
                                                                calculate_cost, budget,
                                                                current_cost, model_generator, random_state=0)

        self.assertListEqual([Coordinate(70, 6)] * 5, suggested_points)

        possible_points = [Coordinate(60, 6), Coordinate(70, 5), Coordinate(70, 6)]
        suggested_points, rep_numbers = suggest_points_gpr_mode(experiment, possible_points, selected_callpath, metric,
                                                                calculate_cost, budget,
                                                                current_cost, model_generator, random_state=0)

        self.assertListEqual([Coordinate(70, 6), Coordinate(60, 6)], suggested_points[:2])

    def test_suggest_points_gpr_mode_multi_parameter_without_values5(self):
        file_reader = TextFileReader()
        experiment = file_reader.read_experiment("data/text/two_parameter_5.txt")
        model_generator = ModelGenerator(experiment)
        model_generator.model_all()

        selected_callpath = [Callpath('main')]
        metric = Metric('')

        def calculate_cost(p, t):
            return t

        budget = math.inf
        current_cost = 2039

        existing_measurements_dict = {}
        for measurement in experiment.measurements[(selected_callpath[0], metric)]:
            # prepare dictionary of existing measurements
            if measurement.coordinate not in existing_measurements_dict:
                existing_measurements_dict[measurement.coordinate] = MeanRepPair(measurement.mean,
                                                                                 measurement.repetitions)
            else:
                existing_measurements_dict[measurement.coordinate] += MeanRepPair(measurement.mean,
                                                                                  measurement.repetitions)

        cost = get_cost_of_measurements(calculate_cost, existing_measurements_dict)
        self.assertAlmostEqual(current_cost, cost, places=2)

        possible_points = [Coordinate(64, 60)]
        suggested_points, rep_numbers = suggest_points_gpr_mode(experiment, possible_points, selected_callpath, metric,
                                                                calculate_cost, budget,
                                                                current_cost, model_generator, random_state=0)

        self.assertListEqual([Coordinate(64, 60)] * 5, suggested_points)

        possible_points = [Coordinate(64, 60), Coordinate(64, 70), Coordinate(128, 70)]
        suggested_points, rep_numbers = suggest_points_gpr_mode(experiment, possible_points, selected_callpath, metric,
                                                                calculate_cost, budget,
                                                                current_cost, model_generator, random_state=0)

        self.assertListEqual([Coordinate(128, 70), Coordinate(64, 60)], suggested_points[:2])


if __name__ == '__main__':
    unittest.main()
