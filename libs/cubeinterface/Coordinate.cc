#include "Coordinate.h"
#include "Utilities.h"
#include "Experiment.h"
#include <cmath>

namespace EXTRAP
{
const std::string Coordinate::COORDINATE_PREFIX = "Coordinate";

int64_t
Coordinate::getId( void ) const
{
    return m_id;
}

void
Coordinate::setId( int64_t id )
{
    m_id = id;
}

const std::string
Coordinate::toString() const
{
    std::stringstream s;
    for ( Coordinate::const_iterator it = begin(); it != end(); it++ )
    {
        s << "(" << it->first.getName() << "," << it->second << ")";
    }
    return s.str();
}

bool
Coordinate::serialize( IoHelper* ioHelper )
{
    SAFE_RETURN( ioHelper->writeString(  Coordinate::COORDINATE_PREFIX ) );
    SAFE_RETURN( ioHelper->writeId(  getId() ) );
    SAFE_RETURN( ioHelper->writeInt(  size() ) );

    //write ParameterValues
    for ( Coordinate::iterator it = begin();
          it != end();
          it++ )
    {
        Parameter p = it->first;
        SAFE_RETURN( ioHelper->writeString(  p.getName() ) );
        Value val = it->second;
        SAFE_RETURN( ioHelper->writeValue(  val ) );
    }
    return true;
}

Coordinate*
Coordinate::deserialize(                          const Experiment* experiment, IoHelper* ioHelper )
{
    int64_t id     = ioHelper->readId();
    int     length = ioHelper->readInt();

    DebugStream << "Read Coordinate:\n"
                << " coordinate id:  " << id << "\n"
                << " num parameters: " << length << std::endl;

    Coordinate* coordinate = new Coordinate();
    for ( int i = 0; i < length; i++ )
    {
        Parameter param( ioHelper->readString() );
        Value     val = ioHelper->readValue();
        coordinate->insert( std::pair<Parameter, Value>( param, val ) );
    }
    return coordinate;
}

bool
lessCoordinate( const Coordinate* lhs, const Coordinate* rhs )
{
    for ( ParameterValueList::const_iterator it = lhs->begin(); it != lhs->end(); it++ )
    {
        if ( it->second < rhs->find( it->first )->second )
        {
            return true;
        }
        else if ( it->second > rhs->find( it->first )->second )
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

Coordinate*
Coordinate::copy() const
{
    Coordinate* result = new Coordinate();
    for ( Coordinate::const_iterator it = this->begin(); it != this->end(); it++ )
    {
        ( *result )[ it->first ] = it->second;
    }
    return result;
}
};