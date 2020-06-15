#ifndef SINGLE_PARAMETER_EXHAUSTIVE_MODELER_H
#define SINGLE_PARAMETER_EXHAUSTIVE_MODELER_H

#include "ModelGenerator.h"
#include "SingleParameterFunctionModeler.h"
#include "CompoundTerm.h"
#include "SingleParameterFunction.h"
#include "Fraction.h"
#include <cassert>
#include <fstream>
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
class SingleParameterExhaustiveFunctionModeler : public SingleParameterFunctionModeler
{
    friend class SingleParameterExhaustiveModelGenerator;

public:
    static const std::string SINGLEPARAMETEREXHAUSTIVEFUNCTIONMODELER_PREFIX;

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
    //Start of external state
    // (nothing here)
    //End of external state

    Fraction m_current_poly_exponent;
    Fraction m_current_log_exponent;
    double   m_constantCost;

    std::ofstream m_outputStream;

    std::string m_functionName;

    void
    writeHeader( void );
    void
    writeModel( SingleParameterHypothesis&    hypothesis,
                double                        polyExponent,
                double                        logExponent,
                const std::vector<DataPoint>& modeledDataPointList );
};
bool
equal( const SingleParameterExhaustiveFunctionModeler* lhs,
       const SingleParameterExhaustiveFunctionModeler* rhs );
};

#endif