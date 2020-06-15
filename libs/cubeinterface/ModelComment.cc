#include "ModelComment.h"

namespace EXTRAP
{
const std::string ModelComment::MODELCOMMENT_PREFIX = "ModelComment";
ModelComment::ModelComment( const std::string& message ) : m_message( message )
{
}

ModelComment::~ModelComment()
{
}

const std::string&
ModelComment::getMessage()
{
    return this->m_message;
}

void
ModelComment::setId( int64_t id )
{
    m_id = id;
}

int64_t
ModelComment::getId( void ) const
{
    return m_id;
}

bool
ModelComment::serialize(  IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  MODELCOMMENT_PREFIX ) );
    SAFE_RETURN( ioHelper->writeId(  getId() ) );
    SAFE_RETURN( ioHelper->writeString(  m_message ) );
    return true;
}

ModelComment*
ModelComment::deserialize( IoHelper* ioHelper )
{
    int64_t       id      = ioHelper->readId();
    std::string   message = ioHelper->readString();
    ModelComment* c       = new ModelComment( message );
    return c;
}
bool
equal( ModelComment* lhs, ModelComment* rhs )
{
    if ( lhs == rhs )
    {
        return true;
    }
    if ( lhs == NULL || rhs == NULL )
    {
        return false;
    }
    const std::string m1 = lhs->getMessage();
    const std::string m2 = rhs->getMessage();
    return m1 == m2;
}
};