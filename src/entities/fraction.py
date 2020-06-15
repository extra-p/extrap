"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""
 
 
from math import floor, fabs


class Fraction:
    """
    This class is used to represent the exponent of a simple term as a fraction.
    It is used in the single parameter function modeler in order to iteratively refine the exponents.
    """


    def __init__(self, n, d):
        """
        Initialize a fraction with numerator n and denominator d.
        """
        
        # set initial values of the numerator and denominator
        self.num = 0
        self.denom = 1
        
        # reduce fraction using extended euclidean algorithm
        _, _, _, t, s = self.compute_extended_euclidean(n, d)
        self.num = -t
        self.denom = s

        # make sure that for negative fractions, the numerator is negative and the denominator is positive
        if self.denom < 0:
            self.num = -self.num
            self.denom = -self.denom
        
        
    def compute_extended_euclidean(self, a, b):
        """
        An implementation of the Extended Euclidean Algorithm.
        For inputs a, b returns the values x, y and gcd such that  x*a + y*b = gcd.
        Also returns t, s such that (-t) = (a div gcd) and s = (b div gcd) where div is integer division.
        """
        s = 0
        old_s = 1
        t = 1
        old_t = 0
        r = b
        old_r = a
        while r != 0:
            quotient = old_r / r
            tmp = r
            r = old_r - quotient * r
            old_r = tmp
            tmp = s
            s = old_s - quotient * s
            old_s = tmp
            tmp = t
            t = old_t - quotient * t
            old_t = tmp
        gcd = old_r
        x = old_s
        y = old_t
        return gcd, x, y, t, s
        
    
    def eval(self):
        """
        Evaluates the fraction and returns the result as a floating point number.
        """
        return self.num / self.denom
        
        
    def numerator_is_zero(self):
        """
        Test if the numerator is 0.
        """
        return self.num == 0
    
    
    def get_numerator(self):
        """
        Returns the value of the numerator.
        """
        return self.num
    
    
    def get_denominator(self):
        """
        Returns the value of the denominator.
        """
        return self.denom
    
    
    def get_fractional_part(self):
        """
        Returns the fractional part of this fraction, equivalent to a modulo operation of numerator and denominator.
        Note that for negative numbers this differs from the usual definition.
        """
        return Fraction(self.num % self.denom, self.denom)
    
    
    def get_integral_part(self):
        """
        Returns the integral (integer) part of this fraction, essentially equivalent to a round-towards-zero operation.
        Note that for negative numbers this differs from the usual definition.
        """
        return self.num / self.denom
    
    
    def approximate(self, x0, accuracy=1e-10):
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
            z = 1.0 / (z-floor(z))
            tmp = denom
            denom = denom * int(z) + prev_denom
            prev_denom = tmp
            num = int(floor( x0_abs * denom + 0.5 ))
            if abs( sign * (float(num) / denom) - x0) < accuracy:
                return Fraction( sign * int(num), denom );
            counter += 1
            
        # when the algorithm fails to find a fraction
        return None
        
    
    def approximate_farey(self, x0, max_denominator):
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
        return Fraction(self.num + other.num, self.denom + other.denom)
    
    
    def __add__(self, other):
        """
        Defines the binary arithmetic operation +.
        """
        n = self.num * other.denom + other.num * self.denom
        d = self.denom * other.denom
        return Fraction(n,d)
    
    
    def __sub__(self, other):
        """
        Defines the binary arithmetic operation -.
        """
        n = self.num * other.denom - other.num * self.denom
        d = self.denom * other.denom
        return Fraction(n,d)
    
    
    def __mul__(self, other):
        """
        Defines the binary arithmetic operation *.
        """
        n = self.m_num * other.m_num
        d = self.m_denom * other.m_denom
        return Fraction(n,d)
    
    
    def __truediv__(self, other):
        """
        Defines the binary arithmetic operation /.
        """
        n = self.num * other.denom
        d = self.denom * other.num
        return Fraction(n,d)
    
    
    def __neg__(self):
        """
        Defines the unary arithmetic operation -.
        """
        return Fraction(-self.num,self.denom)
    
    
    def __lt__(self, other):
        """
        Defines the comparison <.
        """
        return self.eval() < other.eval()
    
    
    def __le__(self, other):
        """
        Defines the comparison <=.
        """
        return self.eval() <= other.eval()
    
    
    def __eq__(self, other):
        """
        Defines the comparison ==.
        """
        # just compare numerator and denominator, because we reduce (normalize) every fraction when it's created
        return self.num == other.num and self.denom == other.denom
    
    
    def __ne__(self, other):
        """
        Defines the comparison !=.
        """
        #TODO: not sure if this works!
        return not ( self == other )
    
    
    def __gt__(self, other):
        """
        Defines the comparison >.
        """
        return self.eval() > other.eval()
    
    
    def __ge__(self, other):
        """
        Defines the comparison >=.
        """
        return self.eval() >= other.eval()
    