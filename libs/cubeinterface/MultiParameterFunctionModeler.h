#ifndef MULTI_PARAMETER_FUNCTION_MODELER_H
#define MULTI_PARAMETER_FUNCTION_MODELER_H

#include "ModelGenerator.h"
#include "MultiParameterTerm.h"
#include "MultiParameterHypothesis.h"
#include "DataPoint.h"
#include "SingleParameterFunctionModeler.h"
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
class MultiParameterFunction;

class MultiParameterFunctionModeler
{
public:
    MultiParameterFunctionModeler();

    virtual MultiParameterHypothesis
    createModel( const Experiment*             experiment,
                 const ModelGeneratorOptions&  options,
                 const std::vector<DataPoint>& modeledDataPointList,
                 ModelCommentList&             comments,
                 const Function*               expectationFunction = NULL ) = 0;

    virtual MultiParameterFunction*
    createConstantModel( const std::vector<DataPoint>& modeledDataPointList );

protected:
    /**
     * Returns the best valid hypothesis from the currently configured search space, starting with a given initialHypothesis.
     * The algorithm uses the abstract functions initSearchSpace, nextHypothesis, getCurrentHypothesis and compareHypotheses
     * to traverse the search space.
     *
     * The function assumes ownership of the passed initialHypothesis and its model function and will delete it
     * unless it is being returned as the best found hypothesis.
     */

    //Start of external state

    SingleParameterFunctionModeler* m_single_parameter_function_modeler;
    //End of external state
};
};

#endif