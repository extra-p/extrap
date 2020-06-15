#ifndef COMPOUNDTERM_H
#define COMPOUNDTERM_H

#include "SimpleTerm.h"
#include "Function.h"

namespace EXTRAP
{
class CompoundTerm : public Function
{
public:
    static const std::string COMPOUNDTERM_PREFIX;
    CompoundTerm( void );
    CompoundTerm( const CompoundTerm& obj );

    static CompoundTerm
    fromLegacy( int a,
                int b,
                int c );

    virtual Value
    evaluate( const ParameterValueList& parameterValues ) const;

    void
    addSimpleTerm( const SimpleTerm& st );

    const std::vector<SimpleTerm>&
    getSimpleTerms( void ) const;

    void
    setCoefficient( Value v );

    Value
    getCoefficient( void ) const;

    virtual std::string
    getAsString( const std::string& parameterName ) const;

    virtual std::string
    getAsStringMultiParameter( const std::string& parameterName ) const;

    virtual std::string
    getAsString( const ParameterList& parameterNames ) const;

    virtual Function*
    clone( void ) const;

    virtual bool
    serialize( IoHelper* ioHelper ) const;

    static CompoundTerm
    deserialize( IoHelper* ioHelper );

    virtual FunctionClass
    getFunctionClass() const;

private:
    std::vector<SimpleTerm> m_simple_terms;
    Value                   m_coefficient;
};
bool
equal( const CompoundTerm* lhs,
       const CompoundTerm* rhs );
};
#endif