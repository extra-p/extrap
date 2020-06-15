#ifndef CALLPATH_H
#define CALLPATH_H

/**
 * Trick the SWIG wrapper generator.
 * If the return value is a const class reference, we can not access
 * methods of the class in python. Thus, we make SWIG think the return
 * values not const via defining an empty CONST. In all other cases
 * we want to have the returned reference of to be const to ensure
 * side-affect free programming.
 */
#ifndef CONST
#define CONST const
#endif

#include "Region.h"
namespace EXTRAP
{
class Callpath;
class Experiment;
class Visitor;

/**
 * Defines the type for a list of Callpath objects.
 */
typedef std::vector<Callpath*> CallpathList;

/**
 * This class represents a callpath definition.
 */
class Callpath
{
public:
    static const std::string CALLPATH_PREFIX;
    /**
     * Creates a new Callpath.
     * @param region  Pointer to the top region of the callpath.
     *                The new object does NOT take ownership of the region.
     * @param parent  Pointer to the parent callpath definition.
     *                The new object does NOT take ownership of the parent.
     */
    Callpath( Region*   region,
              Callpath* parent );

    virtual
    ~Callpath();

    /**
     * Returns a pointer to the region definition.
     */
    virtual Region*
    getRegion( void ) const;

    /**
     * Returns a pointer to the parent callpath definition.
     */
    virtual Callpath*
    getParent( void ) const;

    /**
     * Returns a list of children
     */
    virtual CONST CallpathList&
    getChildren( void ) const;

    /**
     * Adds a new child callpath.
     * @param newChild  Pointer to the new child callpath definition.
     *                  This does NOT take ownership of newChild.
     */
    virtual void
    addChild( Callpath* newChild );

    /**
     * Returns the unique identifier. The identifier can be used
     * as index for the data in the Experiement object.
     */
    int64_t
    getId( void ) const;

    /**
     * Sets the id.
     * @param id  The new identifier.
     */
    void
    setId( int64_t id );

    /**
     * Gets the full name of the region and all its parents, separated by '::'
     */
    std::string
    getFullName( std::string seperator = "->" ) const;

    virtual bool
    serialize(
        IoHelper* ioHelper ) const;

    static Callpath*
    deserialize(
        const Experiment* experiment,
        IoHelper*         ioHelper );

    void
    accept( Visitor& v );

protected:
    /**
     * Stores the pointer to the region definition.
     */
    Region* m_region;

    /**
     * Stores the pointer to the parent callpath definition
     */
    Callpath* m_parent;

    /**
     * Stores the list of child callpaths.
     */
    CallpathList m_children;

private:
    /**
     * Stores the identifier.
     */
    int64_t m_id;
};
bool
equal( const Callpath* lhs,
       const Callpath* rhs );

bool
lessCallpath( const Callpath* lhs,
              const Callpath* rhs );
};

#endif