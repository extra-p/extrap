# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extrap.comparison.metric_conversion.cpu_gpu import FlopsDP
from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.functions import SingleParameterFunction, MultiParameterFunction, ConstantFunction
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.terms import CompoundTerm, MultiParameterTerm
from tests.modelling_testcase import TestCaseWithFunctionAssertions


class TestMetricConversion(TestCaseWithFunctionAssertions):

    def test_metric_conversion(self):
        converter = FlopsDP()
        # TODO implement

    def test_measurement_calculation(self):
        measurement1 = Measurement(Coordinate(1), Callpath('main'), Metric('time'), [1])
        self.assertEqual(measurement1, measurement1 + measurement1 - measurement1)
        measurement2 = measurement1 * 2
        self.assertEqual(2, measurement2.mean)
        self.assertEqual(2, measurement2.median)
        self.assertEqual(2, measurement2.minimum)
        self.assertEqual(2, measurement2.maximum)
        self.assertEqual(0, measurement2.std)
        self.assertEqual(measurement2, measurement1 + measurement1)

    def test_function_calculation(self):
        function1 = SingleParameterFunction(CompoundTerm.create(1, 2, 1))
        function1.constant_coefficient = 3
        function2 = SingleParameterFunction(CompoundTerm.create(1, 3, 2))
        function2.constant_coefficient = 2

        add_const_function = ComputationFunction(function1) + 4

        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) + 4, add_const_function.evaluate(i))

        mul_const_function = ComputationFunction(function1) * 4

        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) * 4, mul_const_function.evaluate(i))

        div_const_function = ComputationFunction(function1) / 4

        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) / 4, div_const_function.evaluate(i))

        mul_function = ComputationFunction(function1) * ComputationFunction(function2)

        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) * function2.evaluate(i), mul_function.evaluate(i))

        add_function = ComputationFunction(function1) + ComputationFunction(function2)

        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) + function2.evaluate(i), add_function.evaluate(i))

        sub_function = ComputationFunction(function1) - ComputationFunction(function2)

        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) - function2.evaluate(i), sub_function.evaluate(i))

        div_function = ComputationFunction(function1) / ComputationFunction(function2)

        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) / function2.evaluate(i), div_function.evaluate(i))

    def test_function_calculation_multi_parameter(self):
        mpterm1 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 1)), (1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 3
        function1 = MultiParameterFunction(mpterm1)
        function1.constant_coefficient = 3
        mpterm2 = MultiParameterTerm((0, CompoundTerm.create(1, 3, 2)), (1, CompoundTerm.create(3, 2, 0)))
        mpterm2.coefficient = 2
        function2 = MultiParameterFunction(mpterm2)
        function2.constant_coefficient = 2

        add_const_function = ComputationFunction(function1) + 4

        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(function1.evaluate((i, j)) + 4,
                                       add_const_function.evaluate((i, j)))

        mul_const_function = ComputationFunction(function1) * 4

        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(function1.evaluate((i, j)) * 4, mul_const_function.evaluate((i, j)))

        div_const_function = ComputationFunction(function1) / 4

        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(function1.evaluate((i, j)) / 4, div_const_function.evaluate((i, j)))

        mul_function = ComputationFunction(function1) * ComputationFunction(function2)

        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(function1.evaluate((i, j)) * function2.evaluate((i, j)),
                                       mul_function.evaluate((i, j)))

        add_function = ComputationFunction(function1) + ComputationFunction(function2)

        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(function1.evaluate((i, j)) + function2.evaluate((i, j)),
                                       add_function.evaluate((i, j)))

        sub_function = ComputationFunction(function1) - ComputationFunction(function2)

        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(function1.evaluate((i, j)) - function2.evaluate((i, j)),
                                       sub_function.evaluate((i, j)))

        div_function = ComputationFunction(function1) / ComputationFunction(function2)

        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(function1.evaluate((i, j)) / function2.evaluate((i, j)),
                                       div_function.evaluate((i, j)))

    def test_function_calculation_multi_parameter_combined(self):
        mpterm1 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 1)), (1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 2
        function1 = MultiParameterFunction(mpterm1)
        function1.constant_coefficient = 2
        mpterm2 = MultiParameterTerm((0, CompoundTerm.create(1, 3, 2)), (1, CompoundTerm.create(3, 2, 0)))
        mpterm2.coefficient = 3
        function2 = MultiParameterFunction(mpterm2)
        function2.constant_coefficient = 3
        mpterm3 = MultiParameterTerm((0, CompoundTerm.create(1, 4, 2)), (1, CompoundTerm.create(2, 1, 0)))
        mpterm3.coefficient = 4
        function3 = MultiParameterFunction(mpterm3)
        function3.constant_coefficient = 4

        test_function = ComputationFunction(function1) * ComputationFunction(function2) / ComputationFunction(
            function3)
        test_function.to_string()
        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(
                    function1.evaluate((i, j)) * function2.evaluate((i, j)) / function3.evaluate((i, j)),
                    test_function.evaluate((i, j)))

        test_function = ComputationFunction(function1) / ComputationFunction(function2) * ComputationFunction(function3)
        test_function.to_string()
        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(
                    function1.evaluate((i, j)) / function2.evaluate((i, j)) * function3.evaluate((i, j)),
                    test_function.evaluate((i, j)))

        test_function = ComputationFunction(function1) / ComputationFunction(function2) * ComputationFunction(
            function3) + 4
        test_function.to_string()
        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(
                    function1.evaluate((i, j)) / function2.evaluate((i, j)) * function3.evaluate((i, j)) + 4,
                    test_function.evaluate((i, j)))

        test_function = (ComputationFunction(function1) - 8) / (
                1 - ComputationFunction(function2)) * ComputationFunction(function3) / 4 + ComputationFunction(
            function1) * 2
        test_function.to_string()
        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(
                    (function1.evaluate((i, j)) - 8) / (1 - function2.evaluate((i, j))) * function3.evaluate(
                        (i, j)) / 4 + function1.evaluate((i, j)) * 2,
                    test_function.evaluate((i, j)))

        test_function = 3 / (-(4 / ComputationFunction(function2)) - (1 + 2 * ComputationFunction(function3))) - 1
        test_function.to_string()
        for i in range(1, 10):
            for j in range(1, 10):
                self.assertAlmostEqual(
                    3 / (-(4 / function2.evaluate((i, j))) - (1 + 2 * function3.evaluate((i, j)))) - 1,
                    test_function.evaluate((i, j)))

    def test_function_reversed_calculation(self):
        function1 = SingleParameterFunction(CompoundTerm.create(1, 2, 1))
        function1.constant_coefficient = 3

        test_function = (ConstantFunction(3) - ComputationFunction(function1))
        for i in range(1, 10):
            self.assertAlmostEqual(3 - function1.evaluate(i), test_function.evaluate(i))

        test_function = (function1 * ComputationFunction(function1))
        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) * function1.evaluate(i), test_function.evaluate(i))

        test_function = (function1 + ComputationFunction(function1))
        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) + function1.evaluate(i), test_function.evaluate(i))

        test_function = (ConstantFunction(3) / ComputationFunction(function1))
        for i in range(1, 10):
            self.assertAlmostEqual(3 / function1.evaluate(i), test_function.evaluate(i))

    def test_function_calculation_optimisation(self):
        function1 = SingleParameterFunction(CompoundTerm.create(1, 2, 1))
        function1.constant_coefficient = 3
        function2 = SingleParameterFunction(CompoundTerm.create(1, 3, 2))
        function2.constant_coefficient = 2
        constant = ConstantFunction()
        constant0 = ConstantFunction(0)

        add_function = ComputationFunction(constant) + ComputationFunction(function1)
        self.assertEqual(ComputationFunction, type(add_function))
        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) + 1, add_function.evaluate(i))

        add_function = ComputationFunction(constant) + ComputationFunction(function1) * ComputationFunction(function2)
        self.assertEqual(ComputationFunction, type(add_function))
        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) * function2.evaluate(i) + 1, add_function.evaluate(i))

        sub_function = ComputationFunction(function1) - ComputationFunction(constant)
        self.assertEqual(ComputationFunction, type(sub_function))
        for i in range(1, 10):
            self.assertAlmostEqual(function1.evaluate(i) - 1, sub_function.evaluate(i))

        mul_function = ComputationFunction(constant0) * ComputationFunction(function1)
        self.assertEqual(ComputationFunction, type(mul_function))
        for i in range(1, 10):
            self.assertAlmostEqual(0, mul_function.evaluate(i))

        mul_function = ComputationFunction(function1) * ComputationFunction(constant0)
        self.assertEqual(ComputationFunction, type(mul_function))
        for i in range(1, 10):
            self.assertAlmostEqual(0, mul_function.evaluate(i))

        mul_function = (ComputationFunction(function1) + ComputationFunction(function2)) * ComputationFunction(
            constant0)
        self.assertEqual(ComputationFunction, type(mul_function))
        for i in range(1, 10):
            self.assertAlmostEqual(0, mul_function.evaluate(i))

        mul_function = (ComputationFunction(function1) + ComputationFunction(function2)) * (ComputationFunction(
            constant0) + 2 * ComputationFunction(
            constant0))
        self.assertEqual(ComputationFunction, type(mul_function))
        for i in range(1, 10):
            self.assertAlmostEqual(0, mul_function.evaluate(i))
        print(mul_function.to_string())
