#ifndef MULTI_PARAMETER_SIMPLE_MODELER_H
#define MULTI_PARAMETER_SIMPLE_MODELER_H

#include "ModelGenerator.h"
#include "MultiParameterFunctionModeler.h"
#include "CompoundTerm.h"
#include "MultiParameterFunction.h"
#include "ModelGeneratorOptions.h"
#include "Fraction.h"
#include <cassert>

namespace EXTRAP
{
class MultiParameterSimpleFunctionModeler : public MultiParameterFunctionModeler
{
public:
    virtual MultiParameterHypothesis
    createModel( const Experiment*             experiment,
                 const ModelGeneratorOptions&  options,
                 const std::vector<DataPoint>& modeledDataPointList,
                 ModelCommentList&             comments,
                 const Function*               expectationFunction = NULL );

protected:

    //Start of external state
    // (nothing here)
    //End of external state
};
};

#endif