#ifndef SINGLE_PARAMETER_MODEL_GENERATOR_H
#define SINGLE_PARAMETER_MODEL_GENERATOR_H

#include "ModelGenerator.h"
#include "CompoundTerm.h"
#include "SingleParameterHypothesis.h"
#include "SingleParameterSimpleFunctionModeler.h"
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
class SingleParameterFunction;

class SingleParameterModelGenerator : public ModelGenerator
{
public:
    SingleParameterModelGenerator();

    virtual
    ~SingleParameterModelGenerator();

    virtual Model*
    createModel( const Experiment*            experiment,
                 const ModelGeneratorOptions& options,
                 const ExperimentPointList&   modeledDataPointList,
                 const Function*              expectationFunction = NULL );

    void
    setCrossvalidationMethod( Crossvalidation cvMethod );

    Crossvalidation
    getCrossvalidationMethod( void ) const;

    void
        setEpsilon( Value );

    Value
    getEpsilon( void ) const;

    void
    setModelGeneratorOptions( ModelGeneratorOptions options );

    ModelGeneratorOptions
    getModelGeneratorOptions( void ) const;

    virtual bool
    serialize( IoHelper* ioHelper ) const = 0;

protected:
    virtual SingleParameterFunctionModeler&
    getFunctionModeler() const = 0;

    void
    deserialize( IoHelper* ioHelper );

    //Start of external state
    ModelGeneratorOptions m_options;
    //End of external state
};
};

#endif