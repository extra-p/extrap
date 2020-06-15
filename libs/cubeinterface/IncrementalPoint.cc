#include "IncrementalPoint.h"
#include "Utilities.h"
#include <algorithm>
#include <math.h>

namespace EXTRAP
{
/* ****************************************************************************
 * static helper functions
 * ***************************************************************************/
/*
   static EXTRAP::Value
   computeTValue( int SampleCount )
   {
    EXTRAP::Value value;
    std::ifstream dataFile;
    dataFile.open( EXTRAP_TVALUES, std::ios::in );
    std::string lineString;
    std::string fieldName;
    if ( dataFile.is_open() == false )
    {
        ErrorStream << "Unable to open the tvalues data file from " << EXTRAP_TVALUES << std::endl;
        exit( EXIT_FAILURE );
    }
    int count = 0;
    while ( count + 2 < SampleCount )
    {
        count++;
        std::getline( dataFile, lineString, '\n' );
    }
    std::getline( dataFile, lineString, '\n' );
    std::istringstream iss( lineString );
    iss >> value;
    dataFile.close();
    return value;
   }

   static EXTRAP::Interval
   computeMeanCI( EXTRAP::Value std_dev, EXTRAP::Value mean, int reps )
   {
    EXTRAP::Interval confidenceInterval;
    if ( reps > 1 )
    {
        EXTRAP::Value tvalue = computeTValue( reps );
        confidenceInterval.end   = mean + tvalue * std_dev / sqrt( reps );
        confidenceInterval.start = mean - tvalue * std_dev / sqrt( reps );
    }
    else
    {

        confidenceInterval.start = mean;
        confidenceInterval.end   = mean;
    }
    return confidenceInterval;
   }
 */

static EXTRAP::Value
computeMedian( std::vector<EXTRAP::Value> sortedValues )
{
    unsigned long N = sortedValues.size();
    EXTRAP::Value median;
    if ( N % 2 == 0 )
    {
        median = ( sortedValues[ N / 2 - 1 ] + sortedValues[ N / 2 ] ) / 2.0;
    }
    else
    {
        median = sortedValues[ ( N - 1 ) / 2 ];
    }
    return median;
}

static EXTRAP::Interval
computeMedianCI( std::vector<EXTRAP::Value> sortedValues )
{
    EXTRAP::Interval confidenceInterval;
    unsigned long    N = sortedValues.size();

    if ( N >= 8 )
    {
        int ci_left_idx  = ( int )floor( ( N - 1.96 * sqrt( N ) ) / 2.0 ) - 1;
        int ci_right_idx = ( int )ceil( ( N + 1.96 * sqrt( N ) ) / 2.0 ) + 1 - 1;
        confidenceInterval.start = sortedValues[ ci_left_idx ];
        confidenceInterval.end   = sortedValues[ ci_right_idx ];
    }
    else
    {
        // no CI possible for N < 8 (leads to ranks outside of boundaries)
        confidenceInterval.start = sortedValues[ 0 ];
        confidenceInterval.end   = sortedValues[ N - 1 ];
    }

    return confidenceInterval;
}

static EXTRAP::Interval
calculateAsymmetricConfidenceInterval( std::vector<EXTRAP::Value> vals, double percentage )
{
    std::sort( vals.begin(), vals.end() );
    unsigned long n = vals.size();

    double alpha = 1.0 - percentage;

    EXTRAP::Interval retVal;
    retVal.start = vals[ floor( ( alpha / 2.0 ) * n ) ];
    retVal.end   = vals[ ceil( ( percentage + ( alpha / 2.0 ) ) * n ) ];

    return retVal;
}

static EXTRAP::Interval
calculateLeBoudecConfidenceIntervalFor95( std::vector<EXTRAP::Value> vals )
{
    std::sort( vals.begin(), vals.end() );
    unsigned long n = vals.size();

    // The value of the normal distribution at 0.025 (with alpha set to 5%, we need z(alpha / 2))
    double z = 1.96;

    EXTRAP::Interval retVal;
    retVal.start = vals[ floor( ( n - z * sqrt( n ) ) / 2 ) - 1 ];
    retVal.end   = vals[ ceil( 1 + ( n + z * sqrt( n ) ) / 2 ) - 1 ];

    return retVal;
}

/*
   static EXTRAP::Value
   computeTValue( int SampleCount, std::string tvaluesPath )
   {
    EXTRAP::Value value;
    std::ifstream dataFile;
    dataFile.open( tvaluesPath.c_str(), std::ios::in );
    std::string lineString;

    if ( dataFile.is_open() == false )
    {
        std::cerr << "Error: Unable to open the tvalues data file from " << tvaluesPath << std::endl;
        exit( EXIT_FAILURE );
    }
    int count = 0;
    while ( count + 2 < SampleCount )
    {
        count++;
        std::getline( dataFile, lineString, '\n' );
    }
    std::getline( dataFile, lineString, '\n' );
    std::istringstream iss( lineString );
    iss >> value;
    dataFile.close();
    return value;
   }

   EXTRAP::Interval
   computeTConfidenceInterval( EXTRAP::Value mean, EXTRAP::Value std_dev, int number_of_repeats,
                            std::string const& tvaluesPath )
   {
    EXTRAP::Interval confidenceInterval;
    if ( number_of_repeats > 1 )
    {
        EXTRAP::Value tvalue = computeTValue( number_of_repeats, tvaluesPath );
        confidenceInterval.start = mean - tvalue * std_dev / sqrt( number_of_repeats );
        confidenceInterval.end   = mean + tvalue * std_dev / sqrt( number_of_repeats );
    }
    else
    {
        confidenceInterval.start = mean;
        confidenceInterval.end   = mean;
    }

    return confidenceInterval;
   }
 */

/* ****************************************************************************
 * class IncrementalPoint
 * ***************************************************************************/

void
IncrementalPoint::addValue( Value value )
{
    m_values.push_back( value );
}

void
IncrementalPoint::clear( void )
{
    m_values.clear();
}

int
IncrementalPoint::getSize( void )
{
    return m_values.size();
}

ExperimentPoint*
IncrementalPoint::getExperimentPoint( const Coordinate* coordinates,
                                      const Metric*     metric,
                                      const Callpath*   callpath )
{
    if ( m_values.empty() )
    {
        return NULL;
    }

    std::sort( m_values.begin(), m_values.end(), std::less<Value>() );

    int      sample_count       = m_values.size();
    Value    mean_value         = 0;
    Interval mean_CI            = { 0, 0 };
    Value    standard_deviation = 0;
    Value    median             = computeMedian( m_values );
    Interval median_CI          = computeMedianCI( m_values );
    Value    minimum            = sample_count > 0 ? m_values[ 0 ] : 0;
    Value    maximum            = sample_count > 0 ? m_values[ sample_count - 1 ] : 0;

    // mean
    for ( int i = 0; i < sample_count; i++ )
    {
        mean_value += m_values[ i ];
    }
    mean_value /= sample_count;

    // standard deviation
    if ( sample_count > 1 )
    {
        for ( int i = 0; i < sample_count; i++ )
        {
            standard_deviation += ( m_values[ i ] - mean_value ) * ( m_values[ i ] - mean_value );
        }
        standard_deviation = sqrt( standard_deviation / ( ( Value )sample_count - 1 ) );
    }

    // mean CI
    //mean_CI = computeMeanCI( standard_deviation, mean_value, sample_count );

    // create new object
    return new ExperimentPoint( coordinates,
                                sample_count,
                                mean_value,
                                mean_CI,
                                standard_deviation,
                                median,
                                median_CI,
                                minimum,
                                maximum,
                                callpath,
                                metric );
}
};