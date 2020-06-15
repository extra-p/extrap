#include "Callpath.h"
#include "IoHelper.h"
#include "Utilities.h"
#include "Experiment.h"

namespace EXTRAP
{
const std::string Callpath::CALLPATH_PREFIX = "Callpath";
Callpath::Callpath( Region* region, Callpath* parent )
    : m_region( region ),
    m_parent( parent )
{
    if ( parent != NULL )
    {
        parent->addChild( this );
    }
}

Callpath::~Callpath()
{
}

Region*
Callpath::getRegion( void ) const
{
    return m_region;
}

Callpath*
Callpath::getParent( void ) const
{
    return m_parent;
}

const CallpathList&
Callpath::getChildren( void ) const
{
    return m_children;
}

void
Callpath::addChild( Callpath* newChild )
{
    m_children.push_back( newChild );
}

int64_t
Callpath::getId( void ) const
{
    return m_id;
}

void
Callpath::setId( int64_t id )
{
    m_id = id;
}

std::string
Callpath::getFullName( std::string seperator ) const
{
    std::string     buffer  = m_region->getName();
    const Callpath* current = m_parent;
    while ( current != NULL )
    {
        buffer.insert( 0, seperator );
        buffer.insert( 0, current->getRegion()->getName() );
        current = current->m_parent;
    }
    return buffer;
}

bool
Callpath::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  CALLPATH_PREFIX ) );
    SAFE_RETURN( ioHelper->writeId(  getId() ) );
    SAFE_RETURN( ioHelper->writeId(  m_region->getId() ) );

    //write parent if there is any
    int64_t parentId;
    if ( this->m_parent == NULL )
    {
        parentId = -1;
    }
    else
    {
        parentId = m_parent->getId();
    }
    SAFE_RETURN( ioHelper->writeId(  parentId ) );
    return true;
};

Callpath*
Callpath::deserialize(  const Experiment* experiment, IoHelper* ioHelper )
{
    int64_t id        = ioHelper->readId();
    int64_t region_id = ioHelper->readId();
    int64_t parent_id = ioHelper->readId();

    DebugStream << "Read Callpath: \n"
                << "  Callpath id: " << id << "\n"
                << "  Region id:   " << region_id << "\n"
                << "  Parent id:   " << parent_id << std::endl;

    Region*   reg = experiment->getRegion( region_id );
    Callpath* parent;
    if ( parent_id == -1 )
    {
        parent = NULL;
    }
    else
    {
        parent = experiment->getCallpath( parent_id );
    }

    Callpath* cp = new Callpath( reg, parent );
    return cp;
};

bool
equal( const Callpath* lhs, const Callpath* rhs )
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
    result &= equal( lhs->getParent(), rhs->getParent() );
    result &= equal( lhs->getRegion(), rhs->getRegion() );
    result &= lhs->getChildren().size() == rhs->getChildren().size();
    return result;
};

bool
lessCallpath( const Callpath* lhs, const Callpath* rhs )
{
    return lhs->getFullName() < rhs->getFullName();
}
};           // Close namespace