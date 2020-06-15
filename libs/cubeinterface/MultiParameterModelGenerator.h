#ifndef MULTI_PARAMETER_MODEL_GENERATOR_H
#define MULTI_PARAMETER_MODEL_GENERATOR_H

#include "ModelGenerator.h"
#include "CompoundTerm.h"
#include "MultiParameterHypothesis.h"
#include "MultiParameterSimpleFunctionModeler.h"
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
class MultiParameterFunction;

class MultiParameterModelGenerator : public ModelGenerator
{
public:
    MultiParameterModelGenerator();

    virtual Model*
    createModel( const Experiment*            experiment,
                 const ModelGeneratorOptions& options,
                 const ExperimentPointList&   modeledDataPointList,
                 const Function*              expectationFunction = NULL );


    void
    setModelGeneratorOptions( ModelGeneratorOptions options );

    ModelGeneratorOptions
    getModelGeneratorOptions( void ) const;

    virtual bool
    serialize( IoHelper* ioHelper ) const;

    void
    deserialize( IoHelper* ioHelper );

protected:
    virtual MultiParameterFunctionModeler&
    getFunctionModeler() const = 0;

    //Start of external state
    ModelGeneratorOptions m_options;
    //End of external state
};
};

#endif