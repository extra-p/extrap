#include "Fraction.h"

namespace EXTRAP
{
Fraction::Fraction() : m_num( 0 ), m_denom( 1 )
{
}

Fraction::Fraction( int n, int d )
{
    if ( d == 0 )
    {
        ErrorStream << "Cannot create fraction with zero denominator ("
                    << n
                    << "/"
                    << d
                    << ")" << std::endl;
        std::abort();
    }

    // reduce fraction using extended euclidean algorithm
    int gcd, x, y, t, s;
    extended_euclidean( n, d, &x, &y, &t, &s, &gcd );

    m_num   = -t;
    m_denom = s;
    if ( m_denom < 0 )
    {
        // make sure that for negative fractions, the numerator is negative and the denominator is positive
        m_num   = -m_num;
        m_denom = -m_denom;
    }
}

/*static*/ void
Fraction::extended_euclidean( int a, int b, int* x, int* y, int* t, int* s, int* gcd )
{
    *s = 0;
    int old_s = 1;
    *t = 1;
    int old_t = 0;
    int r     = b;
    int old_r = a;
    while ( r != 0 )
    {
        int quotient = old_r / r;
        int tmp;
        tmp   = r;
        r     = old_r - quotient * r;
        old_r = tmp;
        tmp   = *s;
        *s    = old_s - quotient * ( *s );
        old_s = tmp;
        tmp   = *t;
        *t    = old_t - quotient * ( *t );
        old_t = tmp;
    }
    *gcd = old_r;
    *x   = old_s;
    *y   = old_t;
}

Value
Fraction::eval( void ) const
{
    return ( double )m_num / m_denom;
}

bool
Fraction::isZero( void ) const
{
    return m_num == 0;
}

int
Fraction::numerator( void ) const
{
    return m_num;
}

int
Fraction::denominator( void ) const
{
    return m_denom;
}

Fraction
Fraction::fractional_part( void ) const
{
    return Fraction( m_num % m_denom, m_denom );
}

int
Fraction::integral_part( void ) const
{
    return m_num / m_denom;
}

/*static*/ Fraction
Fraction::approximate( double x0, double accuracy )
{
    // This implementation is based on a paper by John Kennedy, "Algorithm To Convert A Decimal To A Fraction"
    // (see https://sites.google.com/site/johnkennedyshome/home/downloadable-papers)

    int                sign       = ( 0 < x0 ) - ( x0 < 0 );
    double             x0_abs     = std::abs( x0 );
    double             z          = x0_abs;
    unsigned long long prev_denom = 0;
    unsigned long long denom      = 1;
    unsigned int       iter       = 0;
    do
    {
        z = 1.0 / ( z - std::floor( z ) );
        unsigned long long tmp = denom;
        denom      = denom * ( unsigned long long )z + prev_denom;
        prev_denom = tmp;
        unsigned long long num = ( unsigned long long )std::floor( x0_abs * denom + 0.5 );
        if ( std::abs( sign * ( ( double )num / denom ) - x0 ) < accuracy )
        {
            return Fraction( sign * ( int )num, denom );
        }
    }
    while ( iter++ < 1e6 );         // limit number of iterations
    ErrorStream << "Failed to find a fraction for " << x0 << std::endl;
    return Fraction();
}

/*static*/ Fraction
Fraction::approximate_farey( double x0, int maxDenominator )
{
    int integer_part = ( int )floor( x0 );
    x0 = x0 - integer_part;
    if ( x0 == 0 )
    {
        return Fraction( integer_part, 1 );
    }

    int lo_num = 0, lo_denom = 1, hi_num = 1, hi_denom = 1;
    // do a binary search through the mediants up to rank `maxDenominator`
    for ( int i = 1; i < maxDenominator; i++ )
    {
        int med_num   = lo_num + hi_num;
        int med_denom = lo_denom + hi_denom;
        if ( x0 < ( double )med_num / med_denom )
        {
            // adjust upper boundary, if possible
            if ( med_denom <= maxDenominator )
            {
                hi_num   = med_num;
                hi_denom = med_denom;
            }
        }
        else
        {
            // adjust lower boundary, if possible
            if ( med_denom <= maxDenominator )
            {
                lo_num   = med_num;
                lo_denom = med_denom;
            }
        }
    }

    // which of the two bounds is closer to the real value?
    double delta_hi = fabs( ( double )hi_num / hi_denom - x0 );
    double delta_lo = fabs( ( double )lo_num / lo_denom - x0 );
    if ( delta_hi < delta_lo )
    {
        return Fraction( hi_num + integer_part * hi_denom, hi_denom );
    }
    else
    {
        return Fraction( lo_num + integer_part * lo_denom, lo_denom );
    }
}

Fraction
Fraction::mediant( const Fraction& other ) const
{
    return Fraction( m_num + other.m_num, m_denom + other.m_denom );
}

Fraction
operator+( const Fraction& left, const Fraction& right )
{
    int n = left.m_num * right.m_denom + right.m_num * left.m_denom;
    int d = left.m_denom * right.m_denom;
    return Fraction( n, d );
}

Fraction
operator-( const Fraction& left, const Fraction& right )
{
    int n = left.m_num * right.m_denom - right.m_num * left.m_denom;
    int d = left.m_denom * right.m_denom;
    return Fraction( n, d );
}

Fraction
operator*( const Fraction& left, const Fraction& right )
{
    int n = left.m_num * right.m_num;
    int d = left.m_denom * right.m_denom;
    return Fraction( n, d );
}

Fraction
operator/( const Fraction& left, const Fraction& right )
{
    int n = left.m_num * right.m_denom;
    int d = left.m_denom * right.m_num;
    return Fraction( n, d );
}

Fraction
operator-( const Fraction& frac )
{
    // unary minus
    return Fraction( -frac.m_num, frac.m_denom );
}

bool
operator==( const Fraction& left, const Fraction& right )
{
    // just compare numerator and denominator, because we reduce (normalize) every fraction when it's created
    return left.m_num == right.m_num && left.m_denom == right.m_denom;
}

bool
operator!=( const Fraction& left, const Fraction& right )
{
    return !( left == right );
}

bool
operator<( const Fraction& left, const Fraction& right )
{
    return left.eval() < right.eval();
}

bool
operator>( const Fraction& left, const Fraction& right )
{
    return left.eval() > right.eval();
}

bool
operator<=( const Fraction& left, const Fraction& right )
{
    return left.eval() <= right.eval();
}

bool
operator>=( const Fraction& left, const Fraction& right )
{
    return left.eval() >= right.eval();
}

std::ostream&
operator<<( std::ostream& out, const Fraction& fraction )
{
    if ( fraction.m_denom == 1 )
    {
        out << fraction.m_num;
    }
    else
    {
        out << fraction.m_num << "/" << fraction.m_denom;
    }

    return out;
}

std::wostream&
operator<<( std::wostream& out, const Fraction& fraction )
{
    if ( fraction.m_denom == 1 )
    {
        out << fraction.m_num;
    }
    else
    {
        out << fraction.m_num << "/" << fraction.m_denom;
    }

    return out;
}
};