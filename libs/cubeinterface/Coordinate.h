#ifndef COORDINATE_H
#define COORDINATE_H

#include "Parameter.h"
#include "IoHelper.h"

namespace EXTRAP
{
class Coordinate : public ParameterValueList
{
public:
    static const std::string COORDINATE_PREFIX;

    Coordinate*
    copy() const;

    virtual void
    setId( int64_t id );

    virtual int64_t
    getId( void ) const;

    virtual bool
    serialize(
        IoHelper* helper );

    virtual const std::string
    toString() const;

    static Coordinate*
    deserialize(
        const Experiment* experiment,
        IoHelper*         helper );

private:
    int64_t m_id;
};
bool
lessCoordinate( const Coordinate* lhs,
                const Coordinate* rhs );

typedef std::vector<Coordinate*> CoordinateList;
};

#endif