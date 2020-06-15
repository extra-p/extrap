#ifndef SINGLE_PARAMETER_EXHAUSTIVE_MODEL_GENERATOR_H
#define SINGLE_PARAMETER_EXHAUSTIVE_MODEL_GENERATOR_H

#include "ModelGenerator.h"
#include "CompoundTerm.h"
#include "SingleParameterHypothesis.h"
#include "SingleParameterModelGenerator.h"
#include "SingleParameterExhaustiveFunctionModeler.h"

namespace EXTRAP
{
class SingleParameterFunction;

class SingleParameterExhaustiveModelGenerator : public SingleParameterModelGenerator
{
public:

    static const std::string SINGLEPARAMETEREXHAUSTIVEMODELGENERATOR_PREFIX;
    SingleParameterExhaustiveModelGenerator();

    void
    openOutputFile(  const std::string& file );
    void
    closeOutputFile( void );
    void
    setFunctionName( const std::string& name );
    SingleParameterExhaustiveModelGenerator*
    deserialize(
        IoHelper* ioHelper );
    bool
    serialize(
        IoHelper* ioHelper ) const;

protected:
    virtual SingleParameterFunctionModeler&
    getFunctionModeler() const;

    SingleParameterExhaustiveFunctionModeler* m_modeler;
};

bool
equal( const SingleParameterExhaustiveModelGenerator* lhs,
       const SingleParameterExhaustiveModelGenerator* rhs );
};

#endif