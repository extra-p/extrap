# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import inspect
from fractions import Fraction as _PyFraction
from math import floor, fabs


# noinspection PyAbstractClass
class Fraction(_PyFraction):
    """
    This class is used to represent the exponent of a simple term as a fraction.
    It is used in the single parameter function modeler in order to iteratively refine the exponents.
    """

    def numerator_is_zero(self):
        """
        Test if the numerator is 0.
        """
        return self.numerator == 0

    def get_fractional_part(self):
        """
        Returns the fractional part of this fraction, equivalent to a modulo operation of numerator and denominator.
        Note that for negative numbers this differs from the usual definition.
        """
        numerator = self.numerator % self.denominator
        if self.numerator < 0 and numerator != 0:
            numerator = numerator - self.denominator
        return Fraction(numerator, self.denominator)

    def get_integral_part(self):
        """
        Returns the integral (integer) part of this fraction, essentially equivalent to a round-towards-zero operation.
        Note that for negative numbers this differs from the usual definition.
        """
        return int(self.numerator / self.denominator)  # do not use // because rounding to zero is needed

    @staticmethod
    def approximate(x0, accuracy=1e-10):
        """
        Converts a floating point value to a fraction.
        This implementation is based on a paper by John Kennedy, "Algorithm To Convert A Decimal To A Fraction"
        see https://sites.google.com/site/johnkennedyshome/home/downloadable-papers
        """
        sign = (0 < x0) - (x0 < 0)
        x0_abs = abs(x0)
        z = x0_abs
        prev_denom = 0
        denom = 1
        counter = 0

        while counter < 1e6:
            z = 1.0 / (z - floor(z))
            tmp = denom
            denom = denom * int(z) + prev_denom
            prev_denom = tmp
            num = int(floor(x0_abs * denom + 0.5))
            if abs(sign * (float(num) / denom) - x0) < accuracy:
                return Fraction(sign * int(num), denom)
            counter += 1

        # when the algorithm fails to find a fraction
        return None

    @staticmethod
    def approximate_farey(x0, max_denominator):
        """
        Converts a floating point value to a fraction.
        This implementation uses the Farey sequence in order to approximate the floating point value to a fraction.
        """
        integer_part = int(floor(x0))
        x0 = x0 - integer_part

        if x0 == 0:
            return Fraction(integer_part, 1)

        lo_num = 0
        lo_denom = 1
        hi_num = 1
        hi_denom = 1
        counter = 1

        # do a binary search through the mediants up to rank maxDenominator
        while counter < max_denominator:
            med_num = lo_num + hi_num
            med_denom = lo_denom + hi_denom
            if x0 < float(med_num) / med_denom:
                # adjust upper boundary, if possible
                if med_denom <= max_denominator:
                    hi_num = med_num
                    hi_denom = med_denom
            else:
                # adjust lower boundary, if possible
                if med_denom <= max_denominator:
                    lo_num = med_num
                    lo_denom = med_denom
            counter += 1

        # which of the two bounds is closer to the real value?
        delta_hi = fabs(float(hi_num) / hi_denom - x0)
        delta_lo = fabs(float(lo_num) / lo_denom - x0)
        if delta_hi < delta_lo:
            return Fraction(hi_num + integer_part * hi_denom, hi_denom)
        else:
            return Fraction(lo_num + integer_part * lo_denom, lo_denom)

    def compute_mediant(self, other):
        """
        Computes the mediant of this fraction and another fraction and returns a new fraction.
        """
        return Fraction(self.numerator + other.numerator, self.denominator + other.denominator)

    def mediant(self, other):
        """
        Computes the mediant of this fraction and another fraction and returns a new fraction.
        """
        return Fraction(self.numerator + other.numerator, self.denominator + other.denominator)


# extend python fraction type
for n, f in inspect.getmembers(Fraction, predicate=inspect.isfunction):
    if not hasattr(_PyFraction, n):
        setattr(_PyFraction, n, f)
