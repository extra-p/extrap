# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import itertools
import unittest

from extrap.entities.coordinate import Coordinate
from extrap.mpa.util import identify_step_factor, build_parameter_value_series, extend_parameter_value_series, \
    get_search_space_generator


class TestBuildParameterSeries(unittest.TestCase):

    def test_empty_coordinates(self):
        self.assertEqual([], build_parameter_value_series([]))

    def test_one_parameter(self):
        coordinates = [Coordinate(20), Coordinate(30), Coordinate(40), Coordinate(50), Coordinate(60)]
        self.assertListEqual([[20, 30, 40, 50, 60]], build_parameter_value_series(coordinates))

    def test_one_parameter_unordered(self):
        coordinates = [Coordinate(30), Coordinate(40), Coordinate(20), Coordinate(50), Coordinate(60)]
        self.assertListEqual([[20, 30, 40, 50, 60]], build_parameter_value_series(coordinates))

    def test_two_parameter(self):
        result = [[20, 30, 40, 50, 60], [1, 2, 3, 4]]
        coordinates = []
        for c in itertools.product(*result):
            coordinates += [Coordinate(*c)]
        self.assertListEqual(result, build_parameter_value_series(coordinates))

    def test_three_parameter(self):
        result = [[20, 30, 40, 50, 60], [1, 2, 3, 4], [2, 4, 8, 16, 32]]
        coordinates = []
        for c in itertools.product(*result):
            coordinates += [Coordinate(*c)]
        self.assertListEqual(result, build_parameter_value_series(coordinates))


class TestIdentifyStepFactor(unittest.TestCase):
    def test_corner_cases(self):
        self.assertListEqual([], identify_step_factor([]))
        self.assertListEqual([("+", 1)], identify_step_factor([[]]))
        self.assertListEqual([("*", 2)], identify_step_factor([[1]]))
        self.assertListEqual([("*", 2)], identify_step_factor([[12345]]))

        self.assertListEqual([("+", 1), ("+", 1)], identify_step_factor([[], []]))

    def test_factor_cases(self):
        self.assertListEqual([("*", 2)], identify_step_factor([[1, 3, 4, 8, 16]]))
        self.assertListEqual([("*", 2)], identify_step_factor([[6, 12, 24]]))

        for k in [2, 3, 57, 700, 1000]:
            with self.subTest(faktor=k):
                points = [k ** i for i in range(1, 6)]
                self.assertListEqual([('*', k)], identify_step_factor([points]))
                points = [k ** i for i in range(1, 4)]
                self.assertListEqual([('*', k)], identify_step_factor([points]))
                points = [k ** i for i in range(1, 10)]
                self.assertListEqual([('*', k)], identify_step_factor([points]))
                points = [10 * k ** i for i in range(1, 6)]
                self.assertListEqual([('*', k)], identify_step_factor([points]))
                points = [10 * k ** i for i in range(1, 4)]
                self.assertListEqual([('*', k)], identify_step_factor([points]))
                points = [10 * k ** i for i in range(1, 10)]
                self.assertListEqual([('*', k)], identify_step_factor([points]))

    def test_step_cases(self):
        self.assertListEqual([("+", 2)], identify_step_factor([[2, 4, 6, 8, 10]]))

        for k in [2, 3, 57, 700, 1000]:
            with self.subTest(step_size=k):
                points = [k ** i for i in range(1, 3)]
                self.assertListEqual([('+', k ** 2 - k)], identify_step_factor([points]))
                points = [k * i for i in range(1, 3)]
                self.assertListEqual([('+', k)], identify_step_factor([points]))
                points = [k * i for i in range(1, 6)]
                self.assertListEqual([('+', k)], identify_step_factor([points]))
                points = [k * i for i in range(1, 4)]
                self.assertListEqual([('+', k)], identify_step_factor([points]))
                points = [k * i for i in range(1, 10)]
                self.assertListEqual([('+', k)], identify_step_factor([points]))

    def test_step_factor_cases(self):
        self.assertListEqual([("*", 2.25)], identify_step_factor([[2, 4, 9]]))
        self.assertListEqual([("+", 8)], identify_step_factor([[2, 4, 8, 16, 24, 32]]))
        self.assertListEqual([("+", 9)], identify_step_factor([[3, 9, 27, 36, 45]]))

    def test_multi_parameter(self):
        for k1 in [2, 3, 57, 700, 1000]:
            points1 = [k1 ** i for i in range(1, 4)]
            for k2 in [2, 3, 57, 700, 1000]:
                with self.subTest(faktor=(k1, k2)):
                    result = [('*', k1), ('*', k2)]
                    points2 = [k2 ** i for i in range(1, 4)]
                    self.assertListEqual(result, identify_step_factor([points1, points2]))
                    points2 = [k2 ** i for i in range(1, 10)]
                    self.assertListEqual(result, identify_step_factor([points1, points2]))
                    points2 = [10 * k2 ** i for i in range(1, 6)]
                    self.assertListEqual(result, identify_step_factor([points1, points2]))
                    points2 = [10 * k2 ** i for i in range(1, 10)]
                    self.assertListEqual(result, identify_step_factor([points1, points2]))

        for k1 in [2, 3, 57, 700, 1000]:
            points1 = [k1 * i for i in range(1, 4)]
            for k2 in [2, 3, 57, 700, 1000]:
                with self.subTest(faktor=(0, k2), step=(k1, 0)):
                    result = [('+', k1), ('*', k2)]
                    points2 = [k2 ** i for i in range(1, 4)]
                    self.assertListEqual(result, identify_step_factor([points1, points2]))
                    points2 = [k2 ** i for i in range(1, 10)]
                    self.assertListEqual(result, identify_step_factor([points1, points2]))
                    points2 = [10 * k2 ** i for i in range(1, 6)]
                    self.assertListEqual(result, identify_step_factor([points1, points2]))
                    points2 = [10 * k2 ** i for i in range(1, 10)]
                    self.assertListEqual(result, identify_step_factor([points1, points2]))

        for k1 in [2, 3, 57, 700, 1000]:
            points1 = [k1 ** i for i in range(1, 4)]
            for k2 in [2, 3, 57, 700, 1000]:
                with self.subTest(faktor=(k1, 0), step=(0, k2)):
                    result = [('*', k1), ('+', k2)]
                    points2 = [k2 * i for i in range(1, 4)]
                    self.assertListEqual(result, identify_step_factor([points1, points2]))
                    points2 = [k2 * i for i in range(1, 10)]
                    self.assertListEqual(result, identify_step_factor([points1, points2]))


class TestExtendParameterValueSeries(unittest.TestCase):

    def test_corner_cases(self):
        result = extend_parameter_value_series([[1, 2, 3, 4, 5]], [])
        self.assertEqual([[1, 2, 3, 4, 5]], result)

        result = extend_parameter_value_series([], [('+', 1)])
        self.assertEqual([], result)

        result = extend_parameter_value_series([], [])
        self.assertEqual([], result)

        result = extend_parameter_value_series([[]], [])
        self.assertEqual([[]], result)

        result = extend_parameter_value_series([[]], [('+', 1)])
        self.assertEqual([[]], result)

    def test_one_parameter(self):
        result = extend_parameter_value_series([[1, 2, 3, 4, 5]], [('+', 1)])
        self.assertEqual([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], result)

        result = extend_parameter_value_series([[1, 2, 4, 8, 16]], [('*', 2)])
        self.assertEqual([[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]], result)

        result = extend_parameter_value_series([[2, 4, 9]], [('*', 2.25)], 4)
        self.assertEqual([[2, 4, 4.5, 9, 20.25, 45.5625, 102.515625]], result)

    def test_two_parameter(self):
        result = extend_parameter_value_series([[1, 2, 3, 4, 5], [1, 2, 4, 8, 16]], [('+', 1), ('*', 2)])
        self.assertEqual([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]], result)


class TestSearchSpaceGenerator(unittest.TestCase):
    def test_1_parameter(self):
        parameter_value_serieses = [[1, 2, 3, 4, 5]]
        search_space_coordinates = []
        for i in range(len(parameter_value_serieses[0])):
            search_space_coordinates.append(Coordinate(parameter_value_serieses[0][i]))
        # print("DEBUG search_space_coordinates:",search_space_coordinates)
        self.assertListEqual(search_space_coordinates, list(get_search_space_generator(parameter_value_serieses)))

    def test_2_parameters(self):
        parameter_value_serieses = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]

        search_space_coordinates = []
        for i in range(len(parameter_value_serieses[0])):
            for j in range(len(parameter_value_serieses[1])):
                search_space_coordinates.append(
                    Coordinate(parameter_value_serieses[0][i], parameter_value_serieses[1][j]))
        # print("DEBUG search_space_coordinates:",search_space_coordinates)
        self.assertListEqual(search_space_coordinates, list(get_search_space_generator(parameter_value_serieses)))

    def test_3_parameters(self):
        parameter_value_serieses = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [11, 22, 33, 44, 55]]
        search_space_coordinates = []
        for i in range(len(parameter_value_serieses[0])):
            for j in range(len(parameter_value_serieses[1])):
                for g in range(len(parameter_value_serieses[2])):
                    search_space_coordinates.append(
                        Coordinate(parameter_value_serieses[0][i], parameter_value_serieses[1][j],
                                   parameter_value_serieses[2][g]))
        # print("DEBUG search_space_coordinates:",search_space_coordinates)
        self.assertListEqual(search_space_coordinates, list(get_search_space_generator(parameter_value_serieses)))

    def test_4_parameters(self):
        parameter_value_serieses = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [11, 22, 33, 44, 55],
                                    [100, 200, 300, 400, 500]]
        search_space_coordinates = []
        for i in range(len(parameter_value_serieses[0])):
            for j in range(len(parameter_value_serieses[1])):
                for g in range(len(parameter_value_serieses[2])):
                    for h in range(len(parameter_value_serieses[3])):
                        search_space_coordinates.append(
                            Coordinate(parameter_value_serieses[0][i], parameter_value_serieses[1][j],
                                       parameter_value_serieses[2][g], parameter_value_serieses[3][h]))
        # print("DEBUG search_space_coordinates:",search_space_coordinates)
        self.assertListEqual(search_space_coordinates, list(get_search_space_generator(parameter_value_serieses)))


if __name__ == '__main__':
    unittest.main()
