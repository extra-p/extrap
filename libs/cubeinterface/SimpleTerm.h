#ifndef SIMPLETERM_H
#define SIMPLETERM_H

#include "Function.h"

namespace EXTRAP
{
// The values in this enum are ordered by growth, so the fastest growing function comes last.
enum FunctionType { logarithm, polynomial };
static const char* FunctionTypeNames[ 2 ] = { "logarithm", "polynomial" };

class SimpleTerm : public Function
{
public:
    static const std::string SIMPLETERM_PREFIX;
    virtual Value
    evaluate( const ParameterValueList& parameterValues ) const;
    void
    setFunctionType( FunctionType ft );
    void
    setExponent( double m_exponent );
    FunctionType
    getFunctionType( void ) const;
    double
    getExponent( void ) const;
    std::string
    toString( void );

    virtual std::string
    getAsString( const std::string& parameterName ) const;

    virtual std::string
    getAsStringMultiParameter( const std::string& parameterName ) const;
    virtual std::string
    getAsString( const ParameterList& parameterNames ) const;

    virtual Function*
    clone( void ) const;

    virtual bool
    serialize(
        IoHelper* ioHelper ) const;

    static SimpleTerm
    deserialize(
        IoHelper* ioHelper );

    virtual FunctionClass
    getFunctionClass() const;

private:
    FunctionType m_function_type;
    Value        m_exponent;
};
bool
equal( const SimpleTerm* lhs,
       const SimpleTerm* rhs );
};
#endif