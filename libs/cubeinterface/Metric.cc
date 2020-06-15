#include "Metric.h"
#include "IoHelper.h"
#include "MessageStream.h"
#include "Utilities.h"

namespace EXTRAP
{
const std::string Metric::METRIC_PREFIX = "Metric";

Metric::Metric( const std::string& name,
                const std::string& unit )
    : m_name( name ), m_unit( unit )
{
}

Metric::~Metric()
{
}

std::string
Metric::getName( void ) const
{
    return m_name;
}

std::string
Metric::getUnit( void ) const
{
    return m_unit;
}

int64_t
Metric::getId( void ) const
{
    return m_id;
}

void
Metric::setId( int64_t id )
{
    m_id = id;
}

bool
Metric::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  METRIC_PREFIX ) );
    SAFE_RETURN( ioHelper->writeId(  getId() ) );
    SAFE_RETURN( ioHelper->writeString(  getName() ) );
    SAFE_RETURN( ioHelper->writeString(  getUnit() ) );
    return true;
}

Metric*
Metric::deserialize( IoHelper* ioHelper )
{
    int64_t     id   = ioHelper->readId();
    std::string name = ioHelper->readString();
    std::string unit = ioHelper->readString();

    DebugStream << "Read Metric:\n"
                << "  id:   " << id << "\n"
                << "  name: " << name << "\n"
                << "  unit: " << unit << std::endl;

    Metric* met = new Metric( name, unit );
    return met;
};

bool
equal( const Metric* lhs, const Metric* rhs )
{
    if ( lhs == NULL || rhs == NULL )
    {
        if ( lhs == NULL && rhs == NULL )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    bool result = true;
    //result &= lhs->getId() == rhs->getId();
    result &= 0 == lhs->getName().compare( rhs->getName() );
    result &= 0 == lhs->getUnit().compare( rhs->getUnit() );
    return result;
};

bool
less( const Metric* lhs, const Metric* rhs )
{
    return lhs->getName() < rhs->getName();
}
}; // Close namespace