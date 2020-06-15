#ifndef MULTI_PARAMETER_SIMPLE_MODEL_GENERATOR_H
#define MULTI_PARAMETER_SIMPLE_MODEL_GENERATOR_H

#include "ModelGenerator.h"
#include "CompoundTerm.h"
#include "MultiParameterHypothesis.h"
#include "MultiParameterModelGenerator.h"
#include "MultiParameterSimpleFunctionModeler.h"

namespace EXTRAP
{
class SingleParameterFunction;
class MultiParameterFunction;

class MultiParameterSimpleModelGenerator : public MultiParameterModelGenerator
{
public:

    static const std::string MULTIPARAMETERSIMPLEMODELGENERATOR_PREFIX;
    MultiParameterSimpleModelGenerator();


    virtual bool
    serialize( IoHelper* ioHelper ) const;

    static MultiParameterSimpleModelGenerator*
    deserialize( IoHelper* ioHelper );

protected:
    virtual MultiParameterFunctionModeler&
    getFunctionModeler() const;

    MultiParameterSimpleFunctionModeler* m_modeler;
};

bool
equal( const MultiParameterSimpleModelGenerator* lhs,
       const MultiParameterSimpleModelGenerator* rhs );
};

#endif