#include "Region.h"
#include "Utilities.h"

namespace EXTRAP
{
const std::string Region::REGION_PREFIX = "Region";
Region::Region( const std::string& name,
                const std::string& sourceFile,
                int                lineNumber ) : m_line_no( lineNumber ), m_source_file( sourceFile ), m_name( name )
{
}

Region::~Region()
{
}

std::string
Region::getName( void ) const
{
    return m_name;
}

std::string
Region::getSourceFileName( void ) const
{
    return m_source_file;
}

int
Region::getSourceFileBeginLine( void ) const
{
    return m_line_no;
}

bool
Region::serialize(  IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  REGION_PREFIX ) );
    SAFE_RETURN( ioHelper->writeId(  getId() ) );
    SAFE_RETURN( ioHelper->writeString(  getName() ) );
    SAFE_RETURN( ioHelper->writeString(  getSourceFileName() ) );
    SAFE_RETURN( ioHelper->writeInt(  getSourceFileBeginLine() ) );
    return true;
}

Region*
Region::deserialize(  IoHelper* ioHelper )
{
    int64_t     id                  = ioHelper->readId();
    std::string name                = ioHelper->readString();
    std::string sourceFileName      = ioHelper->readString();
    int         sourceFileBeginLine = ioHelper->readInt();
    Region*     reg                 = new Region( name, sourceFileName, sourceFileBeginLine );
    return reg;
};

int64_t
Region::getId( void ) const
{
    return this->m_id;
};

void
Region::setId( int64_t id )
{
    this->m_id = id;
};

void
Region::print( void ) const
{
    std::cout << "Name:        " << getName() << "\n"
              << "Source file: " << getSourceFileName() << "\n"
              << "Line begin:  " << getSourceFileBeginLine() << std::endl;
}

bool
equal( const Region* lhs, const Region* rhs )
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
    bool equal = true;
    //equal &= lhs->getId() == rhs->getId();
    equal &= 0 == lhs->getName().compare( rhs->getName() );
    equal &= 0 == lhs->getSourceFileName().compare( rhs->getSourceFileName() );
    equal &= lhs->getSourceFileBeginLine() == rhs->getSourceFileBeginLine();
    return equal;
};
};