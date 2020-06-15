#ifndef DATA_POINT_H
#define DATA_POINT_H

#include "Parameter.h"
#include "Region.h"
#include "Metric.h"
#include "Coordinate.h"

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
/**
 * This class represents all necessary data for the model generator to model a single function.
 */
class DataPoint
{
public:
    /**
     * Creates a new DataPoint object.
     * @param coordinates  A list of parameters with values that defines the
     *                     location of the point in the parameter space.
     *                     The new object does NOT take ownership of the
     *                     coordinates.
     * @param sampleCount  The number of samples from which this data point is derived.
     * @param value    The value of the point. This can be a mean, a median or any other value.
     * @param confidenceInterval The confidence interval that corresponds to the value. E.g. this can be a CI around the mean or the median.
     *                           The CI is considered invalid if the sampleCount is 1, because there can be no meaningful CI for a single sample.
     */
    DataPoint( const ParameterValueList* coordinates,
               int                       sampleCount,
               Value                     value,
               Interval                  confidenceInterval );

    /**
     * Returns the value of a given parameter
     */
    virtual Value
    getParameterValue( const Parameter& parameter ) const;

    /**
     * Returns the coordinates.
     */
    virtual CONST ParameterValueList&
    getParameterList( void ) const;

    /**
     * Returns the sample count.
     */
    virtual int
    getSampleCount( void ) const;

    /**
     * Returns the mean.
     */
    virtual Value
    getValue( void ) const;

    /**
     * Returns the confidence interval.
     */
    virtual Interval
    getConfidenceInterval( void ) const;

protected:

    /**
     * Stores the value.
     */
    Value m_value;

    /**
     * Stores the number of samples the data point was derived from.
     */
    int m_sample_count;

    /**
     * Stores the confidence interval.
     */
    Interval m_confidence_interval;

    /**
     * Stores a pointer to the coordinares.
     */
    const ParameterValueList* m_coordinates;
};
};

#endif