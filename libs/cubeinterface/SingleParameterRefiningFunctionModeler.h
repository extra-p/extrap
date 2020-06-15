#ifndef SINGLE_PARAMETER_REFINING_MODELER_H
#define SINGLE_PARAMETER_REFINING_MODELER_H

#include "ModelGenerator.h"
#include "SingleParameterFunctionModeler.h"
#include "CompoundTerm.h"
#include "SingleParameterFunction.h"
#include "Fraction.h"
#include <cassert>
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
class SingleParameterRefiningFunctionModeler : public SingleParameterFunctionModeler
{
public:
    virtual SingleParameterHypothesis
    createModel( const Experiment*             experiment,
                 const ModelGeneratorOptions&  options,
                 const std::vector<DataPoint>& modeledDataPointList,
                 ModelCommentList&             comments,
                 const Function*               expectationFunction = NULL );

    virtual bool
    initSearchSpace( void );

    virtual bool
    nextHypothesis( void );

    virtual SingleParameterFunction*
    buildCurrentHypothesis( void );

    virtual bool
    compareHypotheses( const SingleParameterHypothesis& first,
                       const SingleParameterHypothesis& second,
                       const std::vector<DataPoint>&    modeledDataPointList );

protected:
    enum Step
    {
        REFINING_STEP_INITIAL_POLY,
        REFINING_STEP_REFINEMENT_POLY,
        REFINING_STEP_INITIAL_LOG,
        REFINING_STEP_REFINEMENT_LOG
    };

    struct SearchState
    {
        Fraction                  left;
        Fraction                  center;
        Fraction                  right;
        SingleParameterHypothesis hypothesis;
    };

    virtual SingleParameterHypothesis
    createModelCore( const std::vector<DataPoint>& modeledDataPointList,
                     const Function*               expectationFunction,
                     ModelCommentList&             comments,
                     double*                       constantCost );

    //Start of external state
    // (nothing here)
    //End of external state

    Fraction m_current_best_exponent; // needed to pass hypothesis exponent as fraction out of findBestHypothesis
    Step     m_step;
    Fraction m_current_poly_exponent;
    Fraction m_current_log_exponent;
    Fraction m_left_exponent;
    Fraction m_right_exponent;
    bool     m_left_done;
};
};

#endif