#ifndef FRACTION_H
#define FRACTION_H

#include "Utilities.h"
#include "Types.h"
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cmath>

namespace EXTRAP
{
class Fraction
{
private:
    int m_num, m_denom;

public:
    Fraction();

    Fraction( int n,
              int d );

    /**
     * An implementation of the Extended Euclidean Algorithm.
     * For inputs a, b returns the values x, y and gcd such that  x*a + y*b = gcd.
     * Also returns t, s such that (-t) = (a div gcd) and s = (b div gcd) where div is integer division.
     */
    static void
    extended_euclidean( int  a,
                        int  b,
                        int* x,
                        int* y,
                        int* t,
                        int* s,
                        int* gcd );

    Value
    eval( void ) const;

    bool
    isZero( void ) const;

    int
    numerator( void ) const;

    int
    denominator( void ) const;

    /**
     * Returns the fractional part of this fraction, equivalent to a modulo operation of numerator and denominator.
     * Note that for negative numbers this differs from the usual definition.
     */
    Fraction
    fractional_part( void ) const;

    /**
     * Returns the integral (integer) part of this fraction, essentially equivalent to a round-towards-zero operation.
     * Note that for negative numbers this differs from the usual definition.
     */
    int
    integral_part( void ) const;

    static Fraction
    approximate( double x0,
                 double accuracy = 1e-10 );

    static Fraction
    approximate_farey( double x0,
                       int    maxDenominator );

    Fraction
    mediant( const Fraction& other ) const;

    friend Fraction
    operator+( const Fraction& left,
               const Fraction& right );

    friend Fraction
    operator-( const Fraction& left,
               const Fraction& right );

    friend Fraction
    operator*( const Fraction& left,
               const Fraction& right );

    friend Fraction
    operator/( const Fraction& left,
               const Fraction& right );

    friend Fraction
    operator-( const Fraction& frac );

    friend bool
    operator==( const Fraction& left,
                const Fraction& right );

    friend bool
    operator!=( const Fraction& left,
                const Fraction& right );

    friend bool
    operator<( const Fraction& left,
               const Fraction& right );

    friend bool
    operator>( const Fraction& left,
               const Fraction& right );

    friend bool
    operator<=( const Fraction& left,
                const Fraction& right );

    friend bool
    operator>=( const Fraction& left,
                const Fraction& right );

    friend std::ostream&
    operator<<( std::ostream&   out,
                const Fraction& fraction );

    friend std::wostream&
    operator<<( std::wostream&  out,
                const Fraction& fraction );
};
};
#endif