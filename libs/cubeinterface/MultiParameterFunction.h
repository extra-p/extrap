#ifndef MULTIPARAMETERFUNCTION_H
#define MULTIPARAMETERFUNCTION_H

#include "Function.h"
#include "MultiParameterTerm.h"

namespace EXTRAP
{
class MultiParameterFunction : public Function
{
public:
    static const std::string MULTIPARAMETERFUNCTION_PREFIX;
    MultiParameterFunction( void );
    MultiParameterFunction( const MultiParameterFunction& obj );

    virtual Value
    evaluate( const ParameterValueList& parameterValues ) const;

    virtual std::string
    getAsString( const ParameterList& parameterNames ) const;

    void
    addMultiParameterTerm( const MultiParameterTerm& ct );

    std::vector<MultiParameterTerm>&
    getMultiParameterTerms( void );

    const std::vector<MultiParameterTerm>&
    getMultiParameterTerms( void ) const;

    void
    setConstantCoefficient( Value v );

    Value
    getConstantCoefficient( void ) const;

    virtual Function*
    clone( void ) const;

    virtual FunctionClass
    getFunctionClass() const;

    virtual bool
    serialize( IoHelper* ioHelper ) const;

    static MultiParameterFunction*
    deserialize( IoHelper* ioHelper );

private:
    std::vector<MultiParameterTerm> m_multi_parameter_terms;
    Value                           m_constant_coefficient;
};
bool
equal( const MultiParameterFunction* lhs,
       const MultiParameterFunction* rhs );
};

#endif