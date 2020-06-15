#ifndef PARAMETER_H
#define PARAMETER_H

#include "Types.h"
#include "IoHelper.h"
#include <string>
#include <map>
#include <vector>
#include <set>
#include <fstream>
#include <cassert>

namespace EXTRAP
{
class Experiment;

/**
 * This class represents a parameter definition.
 */
class Parameter
{
public:

    static const std::string PARAMETER_PREFIX;
    /**
     * Creates an empty parameter.
     */
    Parameter();

    /**
     * Creates a parameter.
     * @param name  The name of the parameter.
     */
    Parameter( const std::string& name );

    /**
     * Make destructor virtual
     */
    virtual
    ~Parameter();

    /**
     * Returns the parameter name.
     */
    virtual std::string
    getName( void ) const;

    /**
     * Compares to parameter definitions and defines a less-than
     * order which is based on the parameter name. Necessary to use
     * Parameter objects as keys.
     */
    virtual bool
    operator<( const Parameter& param ) const;

    virtual int64_t
    getId( void ) const;

    virtual void
    setId( int64_t id );

    virtual bool
    serialize(
        IoHelper* ioHelper ) const;

    static Parameter
    deserialize(
        IoHelper* ioHelper );

protected:
    /**
     * Stores the parameter name.
     */
    std::string m_name;
private:
    int64_t m_id;
};
bool
equal( const Parameter* lhs,
       const Parameter* rhs );

/**
 * Defines the type for a list of parameter definitions.
 */
typedef std::vector<Parameter> ParameterList;

/**
 * Defines the type for list where each Parameter has a value assigned.
 * It is used to define a point in the parameter space.
 */
typedef std::map<Parameter, Value> ParameterValueList;

bool
equal( const ParameterValueList* lhs,
       const ParameterValueList* rhs );
};

#endif