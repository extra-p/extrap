#ifndef SINGLEPARAMETERFUNCTION_H
#define SINGLEPARAMETERFUNCTION_H

#include "Function.h"
#include "CompoundTerm.h"

namespace EXTRAP
{
class SingleParameterFunction : public Function
{
public:
    static const std::string SINGLEPARAMETERFUNCTION_PREFIX;
    SingleParameterFunction( void );
    SingleParameterFunction( const SingleParameterFunction& obj );

    virtual Value
    evaluate( const ParameterValueList& parameterValues ) const;

    virtual std::string
    getAsString( const std::string& parameterName ) const;
    virtual std::string
    getAsString( const ParameterList& parameterNames ) const;

    void
    addCompoundTerm( const CompoundTerm& ct );

    std::vector<CompoundTerm>&
    getCompoundTerms( void );

    const std::vector<CompoundTerm>&
    getCompoundTerms( void ) const;

    void
    setConstantCoefficient( Value v );

    Value
    getConstantCoefficient( void ) const;

    virtual Function*
    clone( void ) const;

    virtual bool
    serialize(
        IoHelper* ioHelper ) const;
    static SingleParameterFunction
    deserialize(
        IoHelper* ioHelper );

    virtual FunctionClass
    getFunctionClass() const;

private:
    std::vector<CompoundTerm> m_compound_terms;
    Value                     m_constant_coefficient;
};
bool
equal( const SingleParameterFunction* lhs,
       const SingleParameterFunction* rhs );
};

#endif