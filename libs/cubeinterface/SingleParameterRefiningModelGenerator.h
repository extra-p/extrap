#ifndef SINGLE_PARAMETER_REFINING_MODEL_GENERATOR_H
#define SINGLE_PARAMETER_REFINING_MODEL_GENERATOR_H

#include "ModelGenerator.h"
#include "CompoundTerm.h"
#include "SingleParameterHypothesis.h"
#include "SingleParameterModelGenerator.h"
#include "SingleParameterRefiningFunctionModeler.h"

namespace EXTRAP
{
class SingleParameterFunction;

class SingleParameterRefiningModelGenerator : public SingleParameterModelGenerator
{
public:

    static const std::string SINGLEPARAMETERREFININGMODELGENERATOR_PREFIX;
    SingleParameterRefiningModelGenerator();

    virtual bool
    serialize(
        IoHelper* ioHelper ) const;

    static SingleParameterRefiningModelGenerator*
    deserialize(
        IoHelper* ioHelper );

protected:
    virtual SingleParameterFunctionModeler&
    getFunctionModeler() const;

    SingleParameterRefiningFunctionModeler* m_modeler;
};

bool
equal( const SingleParameterRefiningModelGenerator* lhs,
       const SingleParameterRefiningModelGenerator* rhs );
};

#endif