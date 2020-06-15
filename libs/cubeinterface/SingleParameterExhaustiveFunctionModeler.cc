#include "SingleParameterExhaustiveFunctionModeler.h"
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
const std::string SingleParameterExhaustiveFunctionModeler::SINGLEPARAMETEREXHAUSTIVEFUNCTIONMODELER_PREFIX = "SingleParameterExhaustiveFunctionModeler";
/* SMALL VARIANT
   const Fraction MIN_EXPONENT   = Fraction(0,1);
   const Fraction STEP_SIZE_POLY = Fraction(1,100);
   const Fraction STEP_SIZE_LOG  = Fraction(1,200);
   const Fraction MAX_POLY       = Fraction(3,2);
   const Fraction MAX_LOG        = Fraction(1,2);
 */

const Fraction MIN_EXPONENT   = Fraction( 0, 1 );
const Fraction STEP_SIZE_POLY = Fraction( 1, 40 );
const Fraction STEP_SIZE_LOG  = Fraction( 1, 40 );
const Fraction MAX_POLY       = Fraction( 5, 1 );
const Fraction MAX_LOG        = Fraction( 5, 2 );

SingleParameterHypothesis
SingleParameterExhaustiveFunctionModeler::createModel( const Experiment* experiment, const ModelGeneratorOptions& options, const std::vector<DataPoint>& modeledDataPointList,
                                                       ModelCommentList&             comments,
                                                       const Function*               expectationFunction )
{
    //Step 1. Compute constant model
    SingleParameterHypothesis constantHypothesis;
    {
        // NOTE: new scope so one can't refer to (potentially invalidated) constantFunction later
        SingleParameterFunction* constantFunction = this->createConstantModel( modeledDataPointList );
        // TODO: also use cross-validation at this point if enabled?
        constantHypothesis = SingleParameterHypothesis( constantFunction );
        constantHypothesis.computeCost( modeledDataPointList );
    }

    m_constantCost = constantHypothesis.getRSS();
    DebugStream << "ExhaustiveFunctionModeler: Now creating models for " << m_functionName << std::endl;
    this->writeModel( constantHypothesis, 0, 0, modeledDataPointList );

    SingleParameterHypothesis bestHypothesis = this->findBestHypothesis( modeledDataPointList, constantHypothesis );
    assert( bestHypothesis.isValid() );
    return bestHypothesis;
}


bool
SingleParameterExhaustiveFunctionModeler::initSearchSpace( void )
{
    m_current_poly_exponent = MIN_EXPONENT;
    m_current_log_exponent  = MIN_EXPONENT;
    if ( m_current_poly_exponent.isZero() && m_current_log_exponent.isZero() )
    {
        // prevent exponent == 0, because it leads to NaN in estimateParameters()
        m_current_poly_exponent = m_current_poly_exponent + STEP_SIZE_POLY;
    }
    return m_current_poly_exponent <= MAX_POLY;
}

bool
SingleParameterExhaustiveFunctionModeler::nextHypothesis( void )
{
    if ( m_current_poly_exponent == MAX_POLY )
    {
        if ( m_current_log_exponent == MAX_LOG )
        {
            return false;
        }
        else
        {
            m_current_log_exponent  = m_current_log_exponent + STEP_SIZE_LOG;
            m_current_poly_exponent = MIN_EXPONENT;
            return true;
        }
    }
    else
    {
        m_current_poly_exponent = m_current_poly_exponent + STEP_SIZE_POLY;
        return true;
    }
}

SingleParameterFunction*
SingleParameterExhaustiveFunctionModeler::buildCurrentHypothesis( void )
{
    double               polyExponent = m_current_poly_exponent.eval();
    double               logExponent  = m_current_log_exponent.eval();
    EXTRAP::SimpleTerm   polyTerm;
    EXTRAP::SimpleTerm   logTerm;
    EXTRAP::CompoundTerm compoundTerm;

    polyTerm.setFunctionType( polynomial );
    logTerm.setFunctionType( logarithm );
    polyTerm.setExponent( polyExponent );
    logTerm.setExponent( logExponent );
    compoundTerm.addSimpleTerm( polyTerm );
    compoundTerm.addSimpleTerm( logTerm );
    compoundTerm.setCoefficient( 1 );
    SingleParameterFunction* simpleFunction = new SingleParameterFunction();
    simpleFunction->addCompoundTerm( compoundTerm );
    return simpleFunction;
}

bool
SingleParameterExhaustiveFunctionModeler::compareHypotheses( const SingleParameterHypothesis& old, const SingleParameterHypothesis& candidate, const std::vector<DataPoint>& modeledDataPointList )
{
    this->writeModel( const_cast<SingleParameterHypothesis&>( candidate ), m_current_poly_exponent.eval(), m_current_log_exponent.eval(), modeledDataPointList );
    return false; // we don't care about best model
}

void
SingleParameterExhaustiveFunctionModeler::writeHeader( void )
{
    m_outputStream << "Function,PolynomialExponent,LogarithmicExponent,Constant,Coefficient,RSS,RelativeRSS,AR2,MaxTermContribution,SMAPE" << std::endl;
}

void
SingleParameterExhaustiveFunctionModeler::writeModel( SingleParameterHypothesis& hypothesis, double polyExponent, double logExponent, const std::vector<DataPoint>& modeledDataPointList  )
{
    SingleParameterFunction* fun            = hypothesis.getFunction();
    double                   coefficient    = 0;
    double                   maxTermContrib = 0;
    if ( fun->getCompoundTerms().size() > 0 )
    {
        CompoundTerm ct = fun->getCompoundTerms()[ 0 ];
        coefficient    = ct.getCoefficient();
        maxTermContrib = hypothesis.calculateMaximalTermContribution( 0, modeledDataPointList );
        hypothesis.computeAdjustedRSquared( this->m_constantCost, modeledDataPointList );
    }
    else
    {
        // if there is no compound term, it's the constant model
        hypothesis.setAR2( 0 );
    }

    m_outputStream << m_functionName << "," << polyExponent << "," << logExponent << "," << fun->getConstantCoefficient() << "," << coefficient << "," << hypothesis.getRSS() << "," << hypothesis.getrRSS() << "," << hypothesis.getAR2() << "," << maxTermContrib << "," << hypothesis.getSMAPE() << std::endl;
}
}; // Close namespace