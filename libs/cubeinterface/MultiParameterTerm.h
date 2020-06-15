#ifndef MULTIPARAMETERTERM_H
#define MULTIPARAMETERTERM_H

#include "CompoundTerm.h"
#include "SimpleTerm.h"
#include "Function.h"

namespace EXTRAP
{
class MultiParameterTerm : public Function
{
public:
    static const std::string MULTIPARAMETERTERM_PREFIX;
    MultiParameterTerm( void );
    MultiParameterTerm( const MultiParameterTerm& obj );


    virtual Value
    evaluate( const ParameterValueList& parameterValues ) const;

    void
    addCompoundTermParameterPair( const CompoundTerm& ct,
                                  const Parameter&    param );

    std::vector<std::pair<CompoundTerm, Parameter> >&
    getCompoundTermParameterPairs( void );

    const std::vector<std::pair<CompoundTerm, Parameter> >&
    getCompoundTermParameterPairs( void ) const;

    void
    setCoefficient( Value v );

    Value
    getCoefficient( void ) const;


    virtual std::string
    getAsString( const ParameterList& parameterNames ) const;

    virtual Function*
    clone( void ) const;

    virtual bool
    serialize( IoHelper* ioHelper ) const;

    static MultiParameterTerm
    deserialize( IoHelper* ioHelper );

private:
    std::vector<std::pair<CompoundTerm, Parameter> > m_compound_term_parameter_pairs;
    Value                                            m_coefficient;
};
bool
equal( const MultiParameterTerm* lhs,
       const MultiParameterTerm* rhs );
};
#endif