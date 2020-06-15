#ifndef REGION_H
#define REGION_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <cassert>
#include "IoHelper.h"

namespace EXTRAP
{
/**
 * This class represents a source code region. It is part of the definitions.
 */
class Region
{
public:
    static const std::string REGION_PREFIX;
    /**
     * Creates a source code region.
     * @param name       The name of the source code region, e.g., the function
     *                   name.
     * @param sourceFile The name of the source file of the region.
     * @param lineNumber The line number of the starting line of the region.
     */
    Region( const std::string& name,
            const std::string& sourceFile,
            int                lineNumber );

    virtual
    ~Region();

    /**
     * Prints the region data to the screen.
     */
    void
    print( void ) const;

    /**
     * Retruns the region name.
     */
    virtual std::string
    getName( void ) const;

    /**
     * Returns the source file name.
     */
    virtual std::string
    getSourceFileName( void ) const;

    /**
     * Returns the line number where the region starts.
     */
    virtual int
    getSourceFileBeginLine( void ) const;

    virtual int64_t
    getId( void ) const;

    virtual void
    setId( int64_t id );

    virtual bool
    serialize(
        IoHelper* ioHelper ) const;

    static Region*
    deserialize(
        IoHelper* ioHelper );

protected:
    /**
     * Stores the region name.
     */
    std::string m_name;

    /**
     * Stores the source file name.
     */
    std::string m_source_file;

    /**
     * Stores the start line number.
     */
    int m_line_no;

    int64_t m_id;
};
bool
equal( const Region* lhs,
       const Region* rhs );

/**
 * Defines the type for a list of regions.
 */
typedef std::vector<Region*> RegionList;
};

#endif