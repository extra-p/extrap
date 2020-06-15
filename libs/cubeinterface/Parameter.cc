#include "Parameter.h"
#include "Utilities.h"

namespace EXTRAP
{
const std::string Parameter::PARAMETER_PREFIX = "Parameter";
Parameter::Parameter()
{
}

Parameter::Parameter( const std::string& name ) : m_name( name )
{
}

Parameter::~Parameter()
{
}

std::string
Parameter::getName( void ) const
{
    return m_name;
}

bool
Parameter::operator<( const Parameter& param ) const
{
    return getName() < param.getName();
}

int64_t
Parameter::getId( void ) const
{
    return m_id;
}

void
Parameter::setId( int64_t id )
{
    m_id = id;
}

bool
Parameter::serialize( IoHelper* ioHelper ) const
{
    DebugStream << "Write Paramter:\n"
                << "  parameter id: " << getId() << "\n"
                << "  name:         " << getName() << std::endl;
    SAFE_RETURN( ioHelper->writeString(  PARAMETER_PREFIX ) );
    SAFE_RETURN( ioHelper->writeId(  getId() ) );
    SAFE_RETURN( ioHelper->writeString(  getName() ) );
    return true;
}

Parameter
Parameter::deserialize(  IoHelper* ioHelper )
{
    int         id        = ioHelper->readId();
    std::string paramName = ioHelper->readString();
    Parameter   p         = Parameter( paramName );
    return p;
}

bool
equal( const Parameter* lhs, const Parameter* rhs )
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
    //result &= rhs->getId() == lhs->getId();
    result &= rhs->getName().compare( lhs->getName() ) == 0;
    return result;
}

bool
equal( const ParameterValueList* lhs, const ParameterValueList* rhs )
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
    result &= lhs->size() == rhs->size();
    if ( !result )
    {
        return false;
    }
    for ( ParameterValueList::const_iterator it = lhs->begin(); it != lhs->end(); it++ )
    {
        if ( rhs->find( it->first ) == rhs->end() )
        {
            return false;
        }
        result &= rhs->find( it->first )->second == it->second;
    }
    return result;
}
};