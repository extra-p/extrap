#ifndef EXPERIMENT_POINT_H
#define EXPERIMENT_POINT_H

#include "Parameter.h"
#include "Callpath.h"
#include "Metric.h"

/**
 * Trick the SWIG wrapper generator.
 * If the return value is a const class reference, we can not access
 * methods of the class in python. Thus, we make SWIG think the return
 * values not const via defining an empty CONST. In all other cases
 * we want to have the returned reference of to be const to ensure
 * side-affect free programming.
 */
#ifndef CONST
#define CONST const
#endif

namespace EXTRAP
{
class Coordinate;

/**
 * This class represents one data point for one metric/callpath pair.
 */
class ExperimentPoint
{
public:
    static const std::string EXPERIMENTPOINT_PREFIX;
    /**
     * Creates a new ExperimentPoint object.
     * @param coordinates  A list of parameters with values that defines the
     *                     location of the point in the parameter space.
     *                     The new object does NOT take ownership of the
     *                     coordinates.
     * @param meanValue    The mean value
     * @param confidenceInterval The confidence interval
     * @param standardDeviation  The standard deviation
     * @param median       The median
     * @param minimum      The minimum
     * @param maximum      The maximum
     */
    ExperimentPoint( const Coordinate* coordinates,
                     int               sampleCount,
                     Value             meanValue,
                     Interval          meanCI,
                     Value             standardDeviation,
                     Value             median,
                     Interval          medianCI,
                     Value             minimum,
                     Value             maximum,
                     const Callpath*   callpath,
                     const Metric*     metric );

    virtual
    ~ExperimentPoint();

    /**
     * Returns the value of a given parameter
     */
    virtual Value
    getParameterValue( const Parameter& parameter ) const;

    /**
     * Returns the coordinates.
     */
    virtual CONST Coordinate&
    getCoordinate( void ) const;

    /**
     * Returns the sample count.
     */
    virtual int
    getSampleCount( void ) const;

    /**
     * Returns the mean.
     */
    virtual Value
    getMean( void ) const;

    /**
     * Returns the confidence interval for the mean.
     */
    virtual Interval
    getMeanCI( void ) const;

    /**
     * Returns standard deviation.
     */
    virtual Value
    getStandardDeviation( void ) const;

    /**
     * Returns the median.
     */
    virtual Value
    getMedian( void ) const;

    /**
     * Returns the confidence interval for the median.
     */
    virtual Interval
    getMedianCI( void ) const;

    /**
     * Returns the minimum.
     */
    virtual Value
    getMinimum( void ) const;

    /**
     * Returns the maximum.
     */
    virtual Value
    getMaximum( void ) const;

    virtual CONST Callpath*
    getCallpath( void ) const;

    virtual CONST Metric*
    getMetric( void ) const;

    virtual bool
    serialize(
        IoHelper* ioHelper ) const;

    static ExperimentPoint*
    deserialize(
        const Experiment* experiment,
        IoHelper*         ioHelper );

protected:

    /**
     * Stores the mean.
     */
    Value m_mean;

    /**
     * Stores the number of samples the data point was derived from.
     */
    int m_sample_count;

    /**
     * Stores a confidence interval for the mean. Only valid if sample count > 1.
     */
    Interval m_confidence_interval_mean;

    /**
     * Stores the standard deviation.
     */
    Value m_standard_deviation;

    /**
     * Stores the median.
     */
    Value m_median;

    /**
     * Stores a confidence interval for the median. Only valid if sample count > 1.
     */
    Interval m_confidence_interval_median;

    /**
     * Stores the minimum.
     */
    Value m_minimum;

    /**
     * Stores the maximum.
     */
    Value m_maximum;

    /**
     * Stores a pointer to the coordinares.
     */
    const Coordinate* m_coordinates;

    /**
     * Stores a pointer to the associated Region.
     */
    const Callpath* m_callpath;

    /**
     * Stores a pointer to the associated Metric.
     */
    const Metric* m_metric;
};

bool
equal( const ExperimentPoint* lhs,
       const ExperimentPoint* rhs );

bool
lessExperimentPoint( const ExperimentPoint* lhs,
                     const ExperimentPoint* rhs );

/**
 * The type of a data point list.
 */
typedef std::vector<ExperimentPoint*> ExperimentPointList;
};

#endif