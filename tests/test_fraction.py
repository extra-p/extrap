import math
import unittest

from extrap.entities.fraction import Fraction


class TestFraction(unittest.TestCase):
    def ASSERT_FRACTION_EQUAL(self, frac, string, value):
        self.assertAlmostEqual(value, frac.eval())  # tests deprecated function
        self.assertAlmostEqual(value, float(frac))
        self.assertEqual(string, str(frac))

    def test_evaluation_and_formatting(self):
        self.ASSERT_FRACTION_EQUAL(Fraction(1, 5), "1/5", 0.2)
        self.ASSERT_FRACTION_EQUAL(Fraction(-1, 5), "-1/5", -0.2)
        self.ASSERT_FRACTION_EQUAL(Fraction(1, -5), "-1/5", -0.2)
        self.ASSERT_FRACTION_EQUAL(Fraction(-1, -5), "1/5", 0.2)
        self.ASSERT_FRACTION_EQUAL(Fraction(-10, 5), "-2", -2)
        self.ASSERT_FRACTION_EQUAL(Fraction(10, -5), "-2", -2)
        self.ASSERT_FRACTION_EQUAL(Fraction(-10, -5), "2", 2)
        self.ASSERT_FRACTION_EQUAL(Fraction(0, -10), "0", 0)

    def test_overloaded_operators(self):
        f = Fraction(2, 1)
        self.ASSERT_FRACTION_EQUAL(f + Fraction(5, 2), "9/2", 9. / 2)
        self.ASSERT_FRACTION_EQUAL(f - Fraction(5, 2), "-1/2", -1. / 2)
        self.ASSERT_FRACTION_EQUAL(f * Fraction(5, 2), "5", 5)
        self.ASSERT_FRACTION_EQUAL(f / Fraction(5, 2), "4/5", 4. / 5)
        self.ASSERT_FRACTION_EQUAL(-Fraction(5, 2), "-5/2", -5. / 2)
        self.assertLess(Fraction(2, 3), Fraction(3, 3))
        self.assertGreater(Fraction(4, 3), Fraction(3, 3))
        self.assertEqual(Fraction(3, 3), Fraction(9, 9))
        self.assertNotEqual(Fraction(3, 3), Fraction(10, 9))
        self.assertFalse(Fraction(3, 3) != Fraction(9, 9))

    def test_fraction_extensions(self):
        f = Fraction(1, 1) + Fraction(1, 1)
        f2 = Fraction(1, 2) + Fraction(2, 1)
        self.ASSERT_FRACTION_EQUAL(f.compute_mediant(f2), "7/3", 7. / 3)
        self.assertEqual(2, f2.get_integral_part())
        self.assertEqual(Fraction(1, 2), f2.get_fractional_part())
        self.assertFalse(f.numerator_is_zero())

    def test_integral_part(self):
        self.assertEqual(1, Fraction(10, 7).get_integral_part())
        self.assertEqual(2, Fraction(14, 7).get_integral_part())
        self.assertEqual(0, Fraction(0, 1).get_integral_part())
        self.assertEqual(1, Fraction(7, 7).get_integral_part())
        self.assertEqual(-1, Fraction(-7, 7).get_integral_part())
        self.assertEqual(-1, Fraction(-10, 7).get_integral_part())
        self.assertEqual(-2, Fraction(-14, 7).get_integral_part())

    def test_fractional_part(self):
        self.assertEqual(Fraction(3, 7), Fraction(10, 7).get_fractional_part())
        self.assertEqual(Fraction(0, 1), Fraction(14, 7).get_fractional_part())
        self.assertEqual(Fraction(0, 1), Fraction(0, 1).get_fractional_part())
        self.assertEqual(Fraction(0, 1), Fraction(7, 7).get_fractional_part())
        self.assertEqual(Fraction(0, 1), Fraction(-7, 7).get_fractional_part())
        self.assertEqual(Fraction(-3, 7), Fraction(-10, 7).get_fractional_part())
        self.assertEqual(Fraction(0, 1), Fraction(-14, 7).get_fractional_part())

    def test_approximate(self):
        self.assertEqual(Fraction(-2, 3), Fraction.approximate(-2. / 3))
        self.assertEqual(Fraction(5, -3), Fraction.approximate(5. / -3))
        self.assertEqual(Fraction(1997, 2000), Fraction.approximate(1997. / 2000))
        self.assertAlmostEqual(math.pi, Fraction.approximate(math.pi), delta=1e-10)

    def test_approximate_farey(self):
        self.assertEqual(Fraction(0, 1), Fraction.approximate_farey(0.0, 5))
        self.assertEqual(Fraction(2, 1), Fraction.approximate_farey(2.0, 5))
        self.assertEqual(Fraction(26, 5), Fraction.approximate_farey(5.2, 5))
        self.assertEqual(Fraction(-2, 1), Fraction.approximate_farey(-2.0, 5))
        self.assertEqual(Fraction(2, 3), Fraction.approximate_farey(2. / 3, 5))
        self.assertEqual(Fraction(-2, 3), Fraction.approximate_farey(-2. / 3, 5))
        self.assertEqual(Fraction(5, -3), Fraction.approximate_farey(5. / -3, 5))
