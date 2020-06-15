#include "DataPoint.h"
#include <cmath>
#include "Utilities.h"

namespace EXTRAP
{
DataPoint::DataPoint( const ParameterValueList* coordinates,
                      int                       sampleCount,
                      Value                     value,
                      Interval                  confidenceInterval )
    : m_coordinates( coordinates ),
    m_sample_count( sampleCount ),
    m_value( value ),
    m_confidence_interval( confidenceInterval )
{
}

const ParameterValueList&
DataPoint::getParameterList( void ) const
{
    return *m_coordinates;
}

Value
DataPoint::getParameterValue( const Parameter& parameter ) const
{
    ParameterValueList::const_iterator it = m_coordinates->find( parameter );
    if ( it != m_coordinates->end() )
    {
        return it->second;
    }
    return INVALID_VALUE;
}

int
DataPoint::getSampleCount( void ) const
{
    return m_sample_count;
}

Value
DataPoint::getValue( void ) const
{
    return m_value;
}

Interval
DataPoint::getConfidenceInterval( void ) const
{
    return m_confidence_interval;
}
}; // Close namespace