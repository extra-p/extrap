#include "Experiment.h"
#include "Utilities.h"
#include <cmath>

namespace EXTRAP
{
const std::string ExperimentPoint::EXPERIMENTPOINT_PREFIX = "ExperimentPoint";
ExperimentPoint::ExperimentPoint( const Coordinate* coordinates,
                                  int               sampleCount,
                                  Value             meanValue,
                                  Interval          meanCI,
                                  Value             standardDeviation,
                                  Value             median,
                                  Interval          medianCI,
                                  Value             minimum,
                                  Value             maximum,
                                  const Callpath*   callpath,
                                  const Metric*     metric )
    : m_coordinates( coordinates ),
    m_sample_count( sampleCount ),
    m_mean( meanValue ),
    m_confidence_interval_mean( meanCI ),
    m_standard_deviation( standardDeviation ),
    m_median( median ),
    m_confidence_interval_median( medianCI ),
    m_minimum( minimum ),
    m_maximum( maximum ),
    m_callpath( callpath ),
    m_metric( metric )
{
}

ExperimentPoint::~ExperimentPoint()
{
}

const Coordinate&
ExperimentPoint::getCoordinate( void ) const
{
    return *m_coordinates;
}



Value
ExperimentPoint::getParameterValue( const Parameter& parameter ) const
{
    Coordinate::const_iterator it = m_coordinates->find( parameter );
    if ( it != m_coordinates->end() )
    {
        return it->second;
    }
    return INVALID_VALUE;
}

int
ExperimentPoint::getSampleCount( void ) const
{
    return m_sample_count;
}

Value
ExperimentPoint::getMean( void ) const
{
    return m_mean;
}

Interval
ExperimentPoint::getMeanCI( void ) const
{
    return m_confidence_interval_mean;
}

Value
ExperimentPoint::getStandardDeviation( void ) const
{
    return m_standard_deviation;
}

Value
ExperimentPoint::getMedian( void ) const
{
    return m_median;
}

Interval
ExperimentPoint::getMedianCI( void ) const
{
    return m_confidence_interval_median;
}

Value
ExperimentPoint::getMinimum( void ) const
{
    return m_minimum;
}

Value
ExperimentPoint::getMaximum( void ) const
{
    return m_maximum;
}

const Callpath*
ExperimentPoint::getCallpath( void ) const
{
    return m_callpath;
}

const Metric*
ExperimentPoint::getMetric( void ) const
{
    return m_metric;
}

bool
ExperimentPoint::serialize( IoHelper* ioHelper ) const
{
    ioHelper->writeString(  EXPERIMENTPOINT_PREFIX );
    ioHelper->writeId(  m_coordinates->getId() );
    ioHelper->writeInt(  getSampleCount() );
    ioHelper->writeValue(  getMean() );
    ioHelper->writeValue(  this->getMeanCI().start );
    ioHelper->writeValue(  this->getMeanCI().end );
    ioHelper->writeValue(  this->getStandardDeviation() );
    ioHelper->writeValue(  getMedian() );
    ioHelper->writeValue(  this->getMedianCI().start );
    ioHelper->writeValue(  this->getMedianCI().end );
    ioHelper->writeValue(  getMinimum() );
    ioHelper->writeValue(  getMaximum() );
    ioHelper->writeId(  getMetric()->getId() );
    ioHelper->writeId(  getCallpath()->getId() );
    DebugStream << "Write ExperimentPoint:\n"
                << "  Samples:  " << getSampleCount() << "\n"
                << "  Mean:     " << getMean() << "\n"
                << "  MeanCI:   [" << getMeanCI().start << "," << getMeanCI().end << "]\n"
                << "  StdDev:   " << getStandardDeviation() << "\n"
                << "  Median:   " << getMedian() << "\n"
                << "  MedianCI: [" << getMedianCI().start << "," << getMedianCI().end << "]\n"
                << "  Maximum:  " << getMaximum() << "\n"
                << "  Minimum:  " << getMinimum() << "\n"
                << "  Metric:   " << getMetric()->getName() << "\n"
                << "  Callpath: " << getCallpath()->getFullName() << std::endl;
    return true;
}

ExperimentPoint*
ExperimentPoint::deserialize(
    const Experiment* experiment, IoHelper* ioHelper )
{
    int64_t           coordinate_id = ioHelper->readId();
    const Coordinate* coordinate    = experiment->getCoordinate( coordinate_id );
    int               sampleCount   = ioHelper->readInt();
    Value             mean          = ioHelper->readValue();
    Interval          meanCI;
    meanCI.start = ioHelper->readValue();
    meanCI.end   = ioHelper->readValue();
    Value    standardDeviation = ioHelper->readValue();
    Value    median            = ioHelper->readValue();
    Interval medianCI;
    medianCI.start = ioHelper->readValue();
    medianCI.end   = ioHelper->readValue();
    Value            minimum    = ioHelper->readValue();
    Value            maximum    = ioHelper->readValue();
    int64_t          metricId   = ioHelper->readId();
    int64_t          callpathId = ioHelper->readId();
    Metric*          metric     = experiment->getMetric( metricId );
    Callpath*        callpath   = experiment->getCallpath( callpathId );
    ExperimentPoint* point      = new ExperimentPoint( coordinate,
                                                       sampleCount,
                                                       mean,
                                                       meanCI,
                                                       standardDeviation,
                                                       median,
                                                       medianCI,
                                                       minimum,
                                                       maximum,
                                                       callpath,
                                                       metric );
    return point;
}

bool
equal( const ExperimentPoint* lhs, const ExperimentPoint* rhs )
{
    if ( lhs == rhs )
    {
        return true;
    }
    if ( lhs == NULL || rhs == NULL )
    {
        return false;
    }
    bool result = true;
    result &= equal( lhs->getCallpath(), rhs->getCallpath() );
    result &= equal( lhs->getMetric(), rhs->getMetric() );
    Coordinate lhspvl = lhs->getCoordinate();
    Coordinate rhspvl = rhs->getCoordinate();
    result &= equal( &( lhspvl ), &( rhspvl ) );
    result &= lhs->getSampleCount() == rhs->getSampleCount();
    result &= lhs->getMean() == rhs->getMean();
    result &= lhs->getMeanCI().start == rhs->getMeanCI().start;
    result &= lhs->getMeanCI().end == rhs->getMeanCI().end;
    result &= lhs->getStandardDeviation() == rhs->getStandardDeviation();
    result &= lhs->getMedian() == rhs->getMedian();
    result &= lhs->getMedianCI().start == rhs->getMedianCI().start;
    result &= lhs->getMedianCI().end == rhs->getMedianCI().end;
    result &= lhs->getMinimum() == rhs->getMinimum();
    result &= lhs->getMaximum() == rhs->getMaximum();
    return result;
}

bool
lessExperimentPoint( const ExperimentPoint* lhs, const ExperimentPoint* rhs )
{
    for ( ParameterValueList::const_iterator it = lhs->getCoordinate().begin(); it != lhs->getCoordinate().end(); it++ )
    {
        if ( it->second < rhs->getCoordinate().find( it->first )->second )
        {
            return true;
        }
        else if ( it->second > rhs->getCoordinate().find( it->first )->second )
        {
            return false;
        }
        else
        {
            continue;
        }
    }
    return false;
}
}; // Close namespace