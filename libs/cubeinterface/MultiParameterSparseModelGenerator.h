#ifndef MULTI_PARAMETER_SPARSE_MODEL_GENERATOR_H
#define MULTI_PARAMETER_SPARSE_MODEL_GENERATOR_H

#include "ModelGenerator.h"
#include "CompoundTerm.h"
#include "MultiParameterHypothesis.h"
#include "MultiParameterModelGenerator.h"
#include "MultiParameterSparseFunctionModeler.h"

namespace EXTRAP
{
class SingleParameterFunction;
class MultiParameterFunction;

class MultiParameterSparseModelGenerator : public MultiParameterModelGenerator
{
public:

    static const std::string MULTIPARAMETERSPARSEMODELGENERATOR_PREFIX;
    MultiParameterSparseModelGenerator();


    virtual bool
    serialize( IoHelper* ioHelper ) const;

    static MultiParameterSparseModelGenerator*
    deserialize( IoHelper* ioHelper );

protected:
    virtual MultiParameterFunctionModeler&
    getFunctionModeler() const;

    MultiParameterSparseFunctionModeler* m_modeler;
};

bool
equal( const MultiParameterSparseModelGenerator* lhs,
       const MultiParameterSparseModelGenerator* rhs );
};

#endif