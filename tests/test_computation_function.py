# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

import numpy

from extrap.entities.calculation_element import divide_no0
from extrap.entities.function_computation import ComputationFunction, ComputationFunctionSchema, CFType
from extrap.entities.functions import MultiParameterFunction, SingleParameterFunction, ConstantFunction
from extrap.entities.terms import MultiParameterTerm, CompoundTerm


class TestComputationFunction(unittest.TestCase):
    def test_creation(self):
        function0 = SingleParameterFunction(CompoundTerm.create(1, 1, 0))
        function0.constant_coefficient = 1

        cfunction0 = ComputationFunction(function0)
        self.assertEqual(function0, cfunction0.original_function)
        self.assertEqual(len(cfunction0._params), 1)
        self.assertTrue(cfunction0._ftype)
        self.assertEqual(0, cfunction0.constant_coefficient)
        cfunction0a = ComputationFunction(cfunction0)
        self.assertEqual(cfunction0, cfunction0a.original_function)
        self.assertEqual(len(cfunction0a._params), 1)
        self.assertEqual(CFType.SINGLE_PARAMETER, cfunction0a._ftype)
        self.assertEqual(0, cfunction0a.constant_coefficient)
        self.assertEqual(cfunction0, cfunction0a)

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

        cfunction1 = ComputationFunction(function1)
        self.assertEqual(function1, cfunction1.original_function)
        self.assertEqual(len(cfunction1._params), 2)
        self.assertEqual(CFType.MULTI_PARAMETER, cfunction1._ftype)
        self.assertEqual(0, cfunction1.constant_coefficient)
        cfunction2 = ComputationFunction(function2)
        self.assertEqual(function2, cfunction2.original_function)
        self.assertEqual(len(cfunction2._params), 2)
        self.assertEqual(CFType.MULTI_PARAMETER, cfunction2._ftype)
        self.assertEqual(0, cfunction2.constant_coefficient)
        cfunction3 = ComputationFunction(function3)
        self.assertEqual(function3, cfunction3.original_function)
        self.assertEqual(len(cfunction3._params), 2)
        self.assertEqual(CFType.MULTI_PARAMETER, cfunction3._ftype)
        self.assertEqual(0, cfunction3.constant_coefficient)
        cfunction3a = ComputationFunction(cfunction3)
        self.assertEqual(cfunction3, cfunction3a.original_function)
        self.assertEqual(len(cfunction3a._params), 2)
        self.assertEqual(CFType.MULTI_PARAMETER, cfunction3a._ftype)
        self.assertEqual(0, cfunction3a.constant_coefficient)
        self.assertEqual(cfunction3, cfunction3a)

        function4 = ConstantFunction(5)

        cfunction4 = ComputationFunction(function4)
        self.assertEqual(function4, cfunction4.original_function)
        self.assertEqual(len(cfunction4._params), 0)
        self.assertEqual(CFType.SINGLE_MULTI_PARAMETER, cfunction4._ftype)
        self.assertEqual(0, cfunction4.constant_coefficient)

    def test_evaluation_sp(self):
        function0 = SingleParameterFunction(CompoundTerm.create(1, 1, 0))
        function0.constant_coefficient = 1

        cfunction0 = ComputationFunction(function0)
        self.assertEqual(function0.evaluate(5), cfunction0.evaluate(5))
        value = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        numpy.testing.assert_array_equal(cfunction0.evaluate(value), function0.evaluate(value))
        value = numpy.array([5])
        numpy.testing.assert_array_equal(cfunction0.evaluate(value), function0.evaluate(value))
        value = [5]
        numpy.testing.assert_array_equal(cfunction0.evaluate(value), function0.evaluate(value))
        value = {0: 5}
        numpy.testing.assert_array_equal(cfunction0.evaluate(value), function0.evaluate(value))
        value = {0: numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        numpy.testing.assert_array_equal(cfunction0.evaluate(value), function0.evaluate(value))

        self.assertRaises(TypeError, function0.evaluate, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertRaises(TypeError, cfunction0.evaluate, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_evaluation_mp(self):
        mpterm1 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 0)), (1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 2
        function1 = MultiParameterFunction(mpterm1)
        function1.constant_coefficient = 2

        cfunction1 = ComputationFunction(function1)
        value = [9, 5]
        self.assertEqual(function1.evaluate(value), cfunction1.evaluate(value))
        value = [numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))
        value = numpy.array([numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])])
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))
        value = {0: numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 1: numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))
        value = {0: 9, 1: 5}
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))

        mpterm1 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 0)))
        mpterm1.coefficient = 2
        function1 = MultiParameterFunction(mpterm1)
        function1.constant_coefficient = 2

        cfunction1 = ComputationFunction(function1)
        value = [9, 5]
        self.assertEqual(function1.evaluate(value), cfunction1.evaluate(value))
        value = [numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))
        value = numpy.array([numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])])
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))
        value = {0: numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 1: numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))
        value = {0: 9, 1: 5}
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))

        mpterm1 = MultiParameterTerm((1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 2
        function1 = MultiParameterFunction(mpterm1)
        function1.constant_coefficient = 2

        cfunction1 = ComputationFunction(function1)
        value = [9, 5]
        self.assertEqual(function1.evaluate(value), cfunction1.evaluate(value))
        value = [numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))
        value = numpy.array([numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])])
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))
        value = {0: numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 1: numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))
        value = {0: 9, 1: 5}
        numpy.testing.assert_array_equal(cfunction1.evaluate(value), function1.evaluate(value))

    def test_to_string_sp(self):
        function = SingleParameterFunction(CompoundTerm.create(1, 1, 0))
        function.constant_coefficient = 1
        cfunction = ComputationFunction(function)
        self.assertEqual('1 + p', cfunction.to_string())

        function = SingleParameterFunction(CompoundTerm.create(1, 2, 0))
        function.constant_coefficient = 1
        cfunction = ComputationFunction(function)
        self.assertEqual(function.to_string(), cfunction.to_string())

        function = SingleParameterFunction(CompoundTerm.create(1, 1, 0), CompoundTerm.create(2, 1, 0),
                                           CompoundTerm.create(3, 1, 0))
        function.constant_coefficient = 1
        cfunction = ComputationFunction(function)
        self.assertEqual('1 + p + p^(2) + p^(3)', cfunction.to_string())

        function = SingleParameterFunction(CompoundTerm.create(1, 1, 2), CompoundTerm.create(2, 1, 0),
                                           CompoundTerm.create(3, 1, 0))
        function.constant_coefficient = 1
        cfunction = ComputationFunction(function)
        self.assertEqual('1 + p * log2(p)^(2) + p^(2) + p^(3)', cfunction.to_string())

        function = SingleParameterFunction(CompoundTerm.create(1, 1, 1), CompoundTerm.create(2, 1, 0),
                                           CompoundTerm.create(3, 1, 0))
        function.constant_coefficient = 1
        cfunction = ComputationFunction(function)
        self.assertEqual('1 + p * log2(p) + p^(2) + p^(3)', cfunction.to_string())

    def test_to_string_mp(self):
        mpterm1 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 0)), (1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 2
        function = MultiParameterFunction(mpterm1)
        function.constant_coefficient = 2
        cfunction = ComputationFunction(function)
        self.assertEqual('2 + 2 * p^(1/2) * q', cfunction.to_string())

        mpterm1 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 3)), (1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 2
        function = MultiParameterFunction(mpterm1)
        function.constant_coefficient = 2
        cfunction = ComputationFunction(function)
        self.assertEqual('2 + 2 * p^(1/2) * q * log2(p)^(3)', cfunction.to_string())

        mpterm2 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 3)))
        mpterm2.coefficient = 3
        function = MultiParameterFunction(mpterm1, mpterm2)
        function.constant_coefficient = 2
        cfunction = ComputationFunction(function)
        self.assertEqual('2 + 3 * p^(1/2) * log2(p)^(3) + 2 * p^(1/2) * q * log2(p)^(3)', cfunction.to_string())

    def test_serialization(self):
        schema = ComputationFunctionSchema()

        function = SingleParameterFunction(CompoundTerm.create(1, 5, 3))
        function.constant_coefficient = 1
        cfunction = ComputationFunction(function)
        ser = schema.dump(cfunction)
        rfunction = schema.load(ser)
        self.assertEqual(cfunction.to_string(), rfunction.to_string())
        self.assertEqual(cfunction.evaluate(5), rfunction.evaluate(5))
        self.assertEqual(cfunction, rfunction)

        function = SingleParameterFunction(CompoundTerm.create(1, 5, 3))
        function.constant_coefficient = 1.6547616587417541
        cfunction = ComputationFunction(function)
        ser = schema.dump(cfunction)
        rfunction = schema.load(ser)
        self.assertEqual(cfunction.to_string(), rfunction.to_string())
        self.assertEqual(cfunction.evaluate(5), rfunction.evaluate(5))
        self.assertEqual(cfunction, rfunction)

        mpterm1 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 0)), (1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 2
        function = MultiParameterFunction(mpterm1)
        function.constant_coefficient = 2
        cfunction = ComputationFunction(function)
        ser = schema.dump(cfunction)
        rfunction = schema.load(ser)
        self.assertEqual(cfunction.to_string(), rfunction.to_string())
        self.assertEqual(cfunction.evaluate([9, 5]), rfunction.evaluate([9, 5]))
        self.assertEqual(cfunction, rfunction)

        mpterm1 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 3)), (1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 2
        function = MultiParameterFunction(mpterm1)
        function.constant_coefficient = 2
        cfunction = ComputationFunction(function)
        ser = schema.dump(cfunction)
        rfunction = schema.load(ser)
        self.assertEqual(cfunction.to_string(), rfunction.to_string())
        self.assertEqual(cfunction.evaluate([9, 5]), rfunction.evaluate([9, 5]))
        self.assertEqual(cfunction, rfunction)

        mpterm2 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 3)))
        mpterm2.coefficient = 3
        function = MultiParameterFunction(mpterm1, mpterm2)
        function.constant_coefficient = 2
        cfunction = ComputationFunction(function)
        ser = schema.dump(cfunction)
        rfunction = schema.load(ser)
        self.assertEqual(cfunction.to_string(), rfunction.to_string())
        self.assertEqual(cfunction.evaluate([9, 5]), rfunction.evaluate([9, 5]))
        self.assertEqual(cfunction, rfunction)

        mpterm1 = MultiParameterTerm((1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 2
        function = MultiParameterFunction(mpterm1)
        function.constant_coefficient = 2
        cfunction = ComputationFunction(function)
        ser = schema.dump(cfunction)
        rfunction = schema.load(ser)
        self.assertEqual(cfunction.to_string(), rfunction.to_string())
        self.assertEqual(cfunction.evaluate([9, 5]), rfunction.evaluate([9, 5]))
        self.assertEqual(cfunction, rfunction)

    def test_addition(self):
        function2 = MultiParameterFunction(MultiParameterTerm((2, CompoundTerm.create(3, 1, 0))))
        mpterm1 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 0)), (1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 2
        function = MultiParameterFunction(mpterm1)
        function.constant_coefficient = 2
        cfunction = ComputationFunction(function)
        self.assertEqual('2 + 2 * p^(1/2) * q', cfunction.to_string())
        self.assertEqual(32, cfunction.evaluate([9, 5]))
        cfunction1 = cfunction + cfunction
        self.assertEqual('4 + 4 * p^(1/2) * q', cfunction1.to_string())
        self.assertEqual(64, cfunction1.evaluate([9, 5]))
        cfunction1 = cfunction + function2
        self.assertEqual('2 + r^(3) + 2 * p^(1/2) * q', cfunction1.to_string())
        self.assertEqual(40, cfunction1.evaluate([9, 5, 2]))
        cfunction1 = cfunction + 3
        self.assertEqual('5 + 2 * p^(1/2) * q', cfunction1.to_string())
        self.assertEqual(35, cfunction1.evaluate([9, 5]))
        cfunction1 = function2 + cfunction
        self.assertEqual('2 + r^(3) + 2 * p^(1/2) * q', cfunction1.to_string())
        self.assertEqual(40, cfunction1.evaluate([9, 5, 2]))
        cfunction1 = 3 + cfunction
        self.assertEqual('5 + 2 * p^(1/2) * q', cfunction1.to_string())
        self.assertEqual(35, cfunction1.evaluate([9, 5]))

        function = SingleParameterFunction(CompoundTerm.create(1, 2, 0))
        function.constant_coefficient = 2
        function2 = SingleParameterFunction(CompoundTerm.create(3, 1, 0))

        cfunction_sp = ComputationFunction(function)
        self.assertEqual('2 + p^(1/2)', cfunction_sp.to_string())
        self.assertEqual(5, cfunction_sp.evaluate(9))
        cfunction1 = cfunction_sp + cfunction_sp
        self.assertEqual('4 + 2 * p^(1/2)', cfunction1.to_string())
        self.assertEqual(10, cfunction1.evaluate(9))
        cfunction1 = cfunction_sp + function2
        self.assertEqual('2 + p^(3) + p^(1/2)', cfunction1.to_string())
        self.assertEqual(734, cfunction1.evaluate(9))
        cfunction1 = cfunction_sp + 3
        self.assertEqual('5 + p^(1/2)', cfunction1.to_string())
        self.assertEqual(8, cfunction1.evaluate(9))

        self.assertRaises(ValueError, lambda: cfunction + cfunction_sp)
        self.assertRaises(ValueError, lambda: cfunction_sp + cfunction)


class TestDivideNo0(unittest.TestCase):

    def test_number_division(self):
        self.assertEqual(1, divide_no0(1, 1))
        self.assertEqual(0, divide_no0(0, 1))
        self.assertEqual(1, divide_no0(10, 10))
        self.assertEqual(1, divide_no0(42, 42))

    def test_number_division_by_zero(self):
        self.assertEqual(1, divide_no0(0, 0))
        self.assertRaises(ZeroDivisionError, divide_no0, 1, 0)
        self.assertRaises(ZeroDivisionError, divide_no0, 10, 0)
        self.assertRaises(ZeroDivisionError, divide_no0, 42, 0)

    def test_function_division(self):
        function2 = MultiParameterFunction(MultiParameterTerm((2, CompoundTerm.create(3, 1, 0))))
        mpterm1 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 0)), (1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 2
        function = MultiParameterFunction(mpterm1)
        function.constant_coefficient = 2
        cfunction = ComputationFunction(function)
        cfunction1 = divide_no0(cfunction, cfunction)
        self.assertEqual('1', cfunction1.to_string())
        self.assertEqual(1, cfunction1.evaluate([9, 5]))
        self.assertEqual(cfunction / cfunction, cfunction1)
        cfunction1 = divide_no0(cfunction, function2)
        true_function = cfunction / function2
        self.assertEqual(true_function.to_string(), cfunction1.to_string())
        self.assertEqual(true_function.evaluate([9, 5, 2]), cfunction1.evaluate([9, 5, 2]))
        self.assertEqual(true_function, cfunction1)
        cfunction1 = divide_no0(cfunction, 3)
        true_function = cfunction / 3
        self.assertEqual(true_function.to_string(), cfunction1.to_string())
        self.assertEqual(true_function.evaluate([9, 5, 2]), cfunction1.evaluate([9, 5]))
        self.assertEqual(true_function, cfunction1)
        cfunction1 = divide_no0(function2, cfunction)
        true_function = function2 / cfunction
        self.assertEqual(true_function.to_string(), cfunction1.to_string())
        self.assertEqual(true_function.evaluate([9, 5, 2]), cfunction1.evaluate([9, 5, 2]))
        self.assertEqual(true_function, cfunction1)
        cfunction1 = divide_no0(3, cfunction)
        true_function = 3 / cfunction
        self.assertEqual(true_function.to_string(), cfunction1.to_string())
        self.assertEqual(true_function.evaluate([9, 5, 2]), cfunction1.evaluate([9, 5, 2]))
        self.assertEqual(true_function, cfunction1)

    def test_function_division_by_zero(self):
        function2 = MultiParameterFunction(MultiParameterTerm((2, CompoundTerm.create(3, 1, 0))))
        mpterm1 = MultiParameterTerm((0, CompoundTerm.create(1, 2, 0)), (1, CompoundTerm.create(1, 1, 0)))
        mpterm1.coefficient = 2
        function = MultiParameterFunction(mpterm1)
        function.constant_coefficient = 2
        cfunction = ComputationFunction(function)
        cfunction1 = divide_no0(cfunction - cfunction, cfunction - cfunction)
        self.assertEqual('1', cfunction1.to_string())
        self.assertEqual(1, cfunction1.evaluate([9, 5]))
        self.assertEqual(ComputationFunction(ConstantFunction(1)), cfunction1)

        cfunction1 = divide_no0(cfunction - cfunction, 0)
        self.assertEqual('1', cfunction1.to_string())
        self.assertEqual(1, cfunction1.evaluate([9, 5]))
        self.assertEqual(ComputationFunction(ConstantFunction(1)), cfunction1)

        cfunction1 = divide_no0(0, cfunction - cfunction)
        self.assertEqual('1', cfunction1.to_string())
        self.assertEqual(1, cfunction1.evaluate([9, 5]))
        self.assertEqual(ComputationFunction(ConstantFunction(1)), cfunction1)

        self.assertRaises(ZeroDivisionError, divide_no0, cfunction, 0)
        self.assertRaises(ZeroDivisionError, divide_no0, function2, 0)
        self.assertRaises(ZeroDivisionError, divide_no0, 1, cfunction - cfunction)
        self.assertRaises(ZeroDivisionError, divide_no0, 10, cfunction - cfunction)
        self.assertRaises(ZeroDivisionError, divide_no0, 42, cfunction - cfunction)


if __name__ == '__main__':
    unittest.main()