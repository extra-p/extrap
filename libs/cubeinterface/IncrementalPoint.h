#ifndef INCREMENTAL_POINT_H
#define INCREMENTAL_POINT_H

#include "ExperimentPoint.h"

namespace EXTRAP
{
/**
 * This class represents one data point for one metric/callpath pair.
 */
class IncrementalPoint
{
public:
    virtual void
    addValue( Value value );

    virtual void
    clear( void );

    virtual int
    getSize( void );

    virtual ExperimentPoint*
    getExperimentPoint( const Coordinate* coordinates,
                        const Metric*     metric,
                        const Callpath*   callpath );

private:
    std::vector<Value> m_values;
};
}; // namespace EXTRAP

#endif