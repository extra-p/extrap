#include "SingleParameterRefiningFunctionModeler.h"
#include <sstream>
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
const std::string BAD_MODEL               = "This model has a large error metric (SMAPE). Probably the measurements can not be explained by PMNF.";
const std::string BETTER_USER_EXPECTATION = "Expectation function specified by user is a better model than the best hypothesis.";

const int    MAX_LOG_EXPONENT       = 2;
const int    MAX_POLY_EXPONENT      = 5;
const double ACCEPTANCE_THRESHOLD   = 1.5;
const double TERMINATION_THRESHOLD  = 2.0;
const double NONCONSTANCY_THRESHOLD = 1.3;

const bool DEBUG_COMMENTS = false; // NOTE: enabling those leads to memory leaks

SingleParameterHypothesis
SingleParameterRefiningFunctionModeler::createModel( const Experiment* experiment, const ModelGeneratorOptions& options, const std::vector<DataPoint>& modeledDataPointList,
                                                     ModelCommentList&             comments,
                                                     const Function*               expectationFunction )
{
    double constantCost;

    // This is outsourced into its own function mainly that we can `return` a hypothesis directly (to simplify control flow)
    SingleParameterHypothesis modelHypothesis = createModelCore( modeledDataPointList, expectationFunction, comments, &constantCost );
    // Calculate AR2, because it is (currently) not needed inside createModelCore, but needed for the model below
    modelHypothesis.computeAdjustedRSquared( constantCost, modeledDataPointList );

    // The resulting model will take ownership of the function
    return modelHypothesis;
}

SingleParameterHypothesis
SingleParameterRefiningFunctionModeler::createModelCore( const std::vector<DataPoint>& modeledDataPointList,
                                                         const Function*               expectationFunction,
                                                         ModelCommentList&             comments,
                                                         double*                       constantCost )
{
    SingleParameterHypothesis expectationHypothesis;

    if ( expectationFunction != NULL )
    {
        // This code extracts information about the expectation function. This is currently not used
        // (we don't have a way to pass an expectation function to the model generator from the GUI),
        // but should probably be included at a higher level rather than in this specific modeler, because it
        // is useful for any kind of model generator.

        //const int MAX_APPROX_DENOMINATOR = 5;

        // expectationFunction must either be NULL or a valid SingleParameterFunction
        // need to clone here, because hypothesis requires a mutable function (at least for estimateParameters)
        SingleParameterFunction* singleParameterExpectation = dynamic_cast<SingleParameterFunction*>( expectationFunction->clone() );
        assert( singleParameterExpectation != NULL ); // assert that cast worked correctly

        // compute cost when expectation function is used as hypothesis
        expectationHypothesis = SingleParameterHypothesis( singleParameterExpectation );
        expectationHypothesis.estimateParameters( modeledDataPointList );
        expectationHypothesis.computeCost( modeledDataPointList );

        // find starting hypothesis if user expectation is given, by extracting the fastest growing
        // simple term (expectation might be a complex term, but we can only deal with one single term)
        std::vector<CompoundTerm> terms = singleParameterExpectation->getCompoundTerms();
        assert( terms.size() > 0 );
        FunctionType maxFunctionType = terms[ 0 ].getSimpleTerms()[ 0 ].getFunctionType();
        double       maxExponent     = -INFINITY;
        for ( int ict = 0; ict < terms.size(); ict++ )
        {
            assert( terms[ ict ].getSimpleTerms().size() > 0 );
            for ( int ist = 0; ist < terms[ ict ].getSimpleTerms().size(); ist++ )
            {
                SimpleTerm st = terms[ ict ].getSimpleTerms()[ ist ];
                if ( st.getFunctionType() > maxFunctionType )
                {
                    maxFunctionType = st.getFunctionType();
                    maxExponent     = st.getExponent();
                }
                else if ( st.getFunctionType() == maxFunctionType && st.getExponent() > maxExponent )
                {
                    maxExponent = st.getExponent();
                }
            }
        }

        // Approximate the exponent with a fraction that has a denominator not larger than MAX_APPROX_DENOMINATOR
        //Fraction     expectedExponent = Fraction::approximate_farey( maxExponent, MAX_APPROX_DENOMINATOR );
        //FunctionType expectedType     = maxFunctionType;
        // These values are not used anywhere.
    }

    //Step 1. Compute constant model
    SingleParameterHypothesis constantHypothesis;
    {
        // NOTE: new scope so one can't refer to (potentially invalidated) constantFunction later
        SingleParameterFunction* constantFunction = this->createConstantModel( modeledDataPointList );
        // TODO: also use cross-validation at this point if enabled?
        constantHypothesis = SingleParameterHypothesis( constantFunction );
        constantHypothesis.computeCost( modeledDataPointList );
    }

    *constantCost = constantHypothesis.getRSS();

    if ( *constantCost == 0 )
    {
        // Optimization: return constant hypothesis if it is a perfect model
        return constantHypothesis;
    }

    if ( DEBUG_COMMENTS )
    {
        std::stringstream ss;
        ss << "Constant hypothesis "
           << " (RSS: " << constantHypothesis.getRSS() << ", SMAPE: " << constantHypothesis.getSMAPE() << ")";
        comments.push_back( new ModelComment( ss.str() ) );
    }

    Value min_parameter_value, max_parameter_value;
    this->getSingleParameterInterval( modeledDataPointList, &min_parameter_value, &max_parameter_value );

    std::vector<SearchState> states;

    std::stringstream init_debug_comment;
    if ( DEBUG_COMMENTS )
    {
        init_debug_comment << "Step 0:";
    }

    int globalBestHypothesisIndex = -1;

    for ( int slice_idx = 0; slice_idx <= ( MAX_LOG_EXPONENT + 1 ); slice_idx++ )
    {
        // initialize for constant model
        m_current_best_exponent = Fraction( 0, 1 );

        // STEP 1: INTEGER EXPONENTS
        if ( slice_idx == ( MAX_LOG_EXPONENT + 1 ) )
        {
            m_step                  = REFINING_STEP_INITIAL_LOG;
            m_current_poly_exponent = Fraction( 0, 1 );
        }
        else
        {
            m_step                 = REFINING_STEP_INITIAL_POLY;
            m_current_log_exponent = Fraction( slice_idx, 1 );
        }

        // NOTE: calling `findBestHypothesis` will also set m_current_best_exponent to the fraction exponents of the best hypothesis
        // NOTE: we have to increase the refcount for constantHypothesis because we're moving a copy into `findBestHypothesis`
        //       and we want to keep using it later (when we're not using it anymore and not returning it, we have to free the function using `freeFunction`)
        constantHypothesis.keepFunctionAlive();
        SingleParameterHypothesis bestHypothesis = this->findBestHypothesis( modeledDataPointList,
                                                                             constantHypothesis );
        assert( constantHypothesis.getFunction() != NULL );
        assert( m_current_best_exponent >= Fraction( 0, 1 ) );

        // STEP 2: COMPUTE BOUNDS FROM BEST INTEGER POLYNOMIAL RESULTS
        Fraction center = m_current_best_exponent;
        Fraction left, right;
        if ( center.isZero() )
        {
            // special case: only need to try right side (minimum is between 0 and 1)
            // if m_left_exponent is 0, it will be skipped later
            left = Fraction( 0, 1 );
        }
        else
        {
            left = center - Fraction( 1, 1 );
        }

        right = center + Fraction( 1, 1 );
        assert( right >= Fraction( 0, 1 ) );
        assert( left >= Fraction( 0, 1 ) );

        assert( states.size() == slice_idx ); // assert correct index (before pushing)
        SearchState state = { left, center, right, bestHypothesis };
        states.push_back( state );

        if ( DEBUG_COMMENTS )
        {
            if ( slice_idx == ( MAX_LOG_EXPONENT + 1 ) )
            {
                init_debug_comment << std::endl << "poly = 0 -> log = [" << state.left << "," << state.center << "," << state.right << "]"
                                   << " (RSS: " << bestHypothesis.getRSS() << ", SMAPE: " << bestHypothesis.getSMAPE() << ")";
            }
            else
            {
                init_debug_comment << std::endl << "log = " << slice_idx << " -> poly = [" << state.left << "," << state.center << "," << state.right << "]"
                                   << " (RSS: " << bestHypothesis.getRSS() << ", SMAPE: " << bestHypothesis.getSMAPE() << ")";
            }
        }

        if ( bestHypothesis.getRSS() == 0 )
        {
            // TODO: fix potential memory leak
            DebugStream << "Returning function " << bestHypothesis.getFunction() << "\n";
            return bestHypothesis;
        }

        if ( globalBestHypothesisIndex < 0 || bestHypothesis.getRSS() < states[ globalBestHypothesisIndex ].hypothesis.getRSS() )
        {
            globalBestHypothesisIndex = slice_idx;
        }
    }

    if ( DEBUG_COMMENTS )
    {
        init_debug_comment << std::endl << "best index = " << globalBestHypothesisIndex;
        comments.push_back( new ModelComment( init_debug_comment.str() ) );
    }

    SingleParameterHypothesis finalBestHypothesis = states[ globalBestHypothesisIndex ].hypothesis;
    finalBestHypothesis.keepFunctionAlive();

    double currentAcceptanceTreshold = ACCEPTANCE_THRESHOLD;

    for ( int i = 0; i < 10; i++ )
    {
        std::stringstream step_debug_comment;
        if ( DEBUG_COMMENTS )
        {
            step_debug_comment << "Step " << ( i + 1 ) << ":";
        }

        std::vector<SearchState> previousStates = states;                    // create a copy of states before the next iteration, for comparisons
        //int                      previousGlobalBestHypothesisIndex = globalBestHypothesisIndex;
        globalBestHypothesisIndex = -1;

        for ( int slice_idx = 0; slice_idx <= ( MAX_LOG_EXPONENT + 1 ); slice_idx++ )
        {
            SearchState& state = states[ slice_idx ]; // this is safe because we don't push to this vector any more after the initialization

            if ( slice_idx == ( MAX_LOG_EXPONENT + 1 ) )
            {
                m_step                  = REFINING_STEP_REFINEMENT_LOG;
                m_current_poly_exponent = Fraction( 0, 1 );
            }
            else
            {
                m_step                 = REFINING_STEP_REFINEMENT_POLY;
                m_current_log_exponent = Fraction( slice_idx, 1 );
            }

            m_current_best_exponent = state.center;
            m_left_exponent         = state.left.mediant( state.center );
            m_right_exponent        = state.right.mediant( state.center );

            // Update state (NOTE: this also sets m_current_best_exponent)
            state.hypothesis = this->findBestHypothesis( modeledDataPointList, state.hypothesis );

            if ( m_current_best_exponent < state.center )
            {
                // zoom in to the left
                state.right  = state.center;
                state.center = m_current_best_exponent;
            }
            else if ( m_current_best_exponent > state.center )
            {
                // zoom in to the right
                state.left   = state.center;
                state.center = m_current_best_exponent;
            }
            else
            {
                // zoom in centrally
                state.left  = state.left.mediant( state.center );
                state.right = state.right.mediant( state.center );
            }

            if ( DEBUG_COMMENTS )
            {
                if ( slice_idx == ( MAX_LOG_EXPONENT + 1 ) )
                {
                    step_debug_comment << std::endl << "poly = 0 -> log = [" << state.left << "," << state.center << "," << state.right << "]"
                                       << " (RSS: " << state.hypothesis.getRSS() << ", SMAPE: " << state.hypothesis.getSMAPE() << ")";
                }
                else
                {
                    step_debug_comment << std::endl << "log = " << slice_idx << " -> poly = [" << state.left << "," << state.center << "," << state.right << "]"
                                       << " (RSS: " << state.hypothesis.getRSS() << ", SMAPE: " << state.hypothesis.getSMAPE() << ")";
                }
            }

            if ( globalBestHypothesisIndex < 0 || state.hypothesis.getRSS() < states[ globalBestHypothesisIndex ].hypothesis.getRSS() )
            {
                globalBestHypothesisIndex = slice_idx;
            }
        }

        step_debug_comment << std::endl << "best index = " << globalBestHypothesisIndex;

        if ( DEBUG_COMMENTS )
        {
            comments.push_back( new ModelComment( step_debug_comment.str() ) );
        }

        if ( states[ globalBestHypothesisIndex ].hypothesis.getRSS() == 0 )
        {
            // found perfect model
            finalBestHypothesis.freeFunction(); // free memory of previously best hypothesis
            finalBestHypothesis = states[ globalBestHypothesisIndex ].hypothesis;
            finalBestHypothesis.keepFunctionAlive();
            break; // stop iterative refinement
        }

        if ( finalBestHypothesis.getSMAPE() / states[ globalBestHypothesisIndex ].hypothesis.getSMAPE() >= currentAcceptanceTreshold ) // improvement more than 50%
        {
            finalBestHypothesis.freeFunction();                                                                                        // free memory of previously best hypothesis
            finalBestHypothesis = states[ globalBestHypothesisIndex ].hypothesis;
            finalBestHypothesis.keepFunctionAlive();                                                                                   // increment refcount so that this hypothesis' function will not be deleted in next iteration
            currentAcceptanceTreshold = ACCEPTANCE_THRESHOLD;
        }
        else
        {
            currentAcceptanceTreshold *= ACCEPTANCE_THRESHOLD;
        }

        bool continueSearch = false;
        for ( int slice_idx = 0; slice_idx <= ( MAX_LOG_EXPONENT + 1 ); slice_idx++ )
        {
            if ( TERMINATION_THRESHOLD * states[ slice_idx ].hypothesis.getSMAPE()  <= previousStates[ slice_idx ].hypothesis.getSMAPE() )
            {
                continueSearch = true;
                break;
            }
        }

        if ( !continueSearch )
        {
            break; // stop iterative refinement
        }
    } // end of iterative refinement loop

    // cleanup unused hypothesis functions from last iteration
    for ( int slice_idx = 0; slice_idx <= ( MAX_LOG_EXPONENT + 1 ); slice_idx++ )
    {
        if ( states[ slice_idx ].hypothesis.getFunction() != finalBestHypothesis.getFunction() )
        {
            states[ slice_idx ].hypothesis.freeFunction();
        }
    }

    // SEE IF IMPROVEMENT OVER CONSTANT MODEL IS GOOD ENOUGH
    if ( constantHypothesis.getFunction() != finalBestHypothesis.getFunction() )
    {
        if ( constantHypothesis.getSMAPE() / finalBestHypothesis.getSMAPE() < NONCONSTANCY_THRESHOLD ||
             finalBestHypothesis.calculateMaximalTermContribution( 0, modeledDataPointList ) < m_eps )
        {
            finalBestHypothesis.freeFunction();
            finalBestHypothesis = constantHypothesis;
        }
        else
        {
            constantHypothesis.freeFunction();
        }
    }

    if ( expectationFunction != NULL )
    {
        if ( expectationHypothesis.getRSS() < finalBestHypothesis.getRSS() )
        {
            comments.push_back( new ModelComment( BETTER_USER_EXPECTATION ) );
            // NOTE: We need to be sure that the function within this hypothesis is not the original instance provided by the user,
            //       because the model will take ownership of the function, which the user does not expect.
            finalBestHypothesis.freeFunction();
            return expectationHypothesis;
        }
        else
        {
            // delete our clone of the expectation function
            expectationHypothesis.freeFunction();
        }
    }

    if ( finalBestHypothesis.getSMAPE() > 2.0 )
    {
        comments.push_back( new ModelComment( BAD_MODEL ) );
    }

    return finalBestHypothesis;
}

bool
SingleParameterRefiningFunctionModeler::initSearchSpace( void )
{
    if ( m_step == REFINING_STEP_INITIAL_POLY )
    {
        if ( m_current_log_exponent.isZero() )
        {
            m_current_poly_exponent = Fraction( 1, 1 ); // start at 1 to prevent constant function
        }
        else
        {
            m_current_poly_exponent = Fraction( 0, 1 );
        }
    }
    else if ( m_step == REFINING_STEP_REFINEMENT_POLY )
    {
        if ( m_left_exponent.isZero() )
        {
            m_left_done             = true;
            m_current_poly_exponent = m_right_exponent;
        }
        else
        {
            m_left_done             = false;
            m_current_poly_exponent = m_left_exponent;
        }
    }
    else if ( m_step == REFINING_STEP_INITIAL_LOG )
    {
        if ( m_current_poly_exponent.isZero() )
        {
            m_current_log_exponent = Fraction( 1, 1 ); // start at 1 to prevent constant function
        }
        else
        {
            m_current_log_exponent = Fraction( 0, 1 );
        }
    }
    else if ( m_step == REFINING_STEP_REFINEMENT_LOG )
    {
        if ( m_left_exponent.isZero() )
        {
            m_left_done            = true;
            m_current_log_exponent = m_right_exponent;
        }
        else
        {
            m_left_done            = false;
            m_current_log_exponent = m_left_exponent;
        }
    }
    else
    {
        assert( false );
    }
    return true; // non-empty search space
}

bool
SingleParameterRefiningFunctionModeler::nextHypothesis( void )
{
    if ( m_step == REFINING_STEP_INITIAL_POLY )
    {
        m_current_poly_exponent = m_current_poly_exponent + Fraction( 1, 1 );
        if ( m_current_poly_exponent > Fraction( MAX_POLY_EXPONENT, 1 ) )
        {
            return false;
        }

        return true;
    }
    else if ( m_step == REFINING_STEP_REFINEMENT_POLY )
    {
        if ( m_left_done )
        {
            return false;
        }
        else
        {
            m_current_poly_exponent = m_right_exponent;
            m_left_done             = true;
            return true;
        }
    }
    else if ( m_step == REFINING_STEP_INITIAL_LOG )
    {
        m_current_log_exponent = m_current_log_exponent + Fraction( 1, 1 );
        if ( m_current_log_exponent > Fraction( MAX_LOG_EXPONENT, 1 ) )
        {
            return false;
        }

        return true;
    }
    else if ( m_step == REFINING_STEP_REFINEMENT_LOG )
    {
        if ( m_left_done )
        {
            return false;
        }
        else
        {
            m_current_log_exponent = m_right_exponent;
            m_left_done            = true;
            return true;
        }
    }
    else
    {
        assert( false ); // Not implemented
    }
}

SingleParameterFunction*
SingleParameterRefiningFunctionModeler::buildCurrentHypothesis( void )
{
    EXTRAP::CompoundTerm compoundTerm;

    if ( !m_current_poly_exponent.isZero() )
    {
        EXTRAP::SimpleTerm term;
        term.setFunctionType( polynomial );
        term.setExponent( m_current_poly_exponent.eval() );
        compoundTerm.addSimpleTerm( term );
    }
    if ( !m_current_log_exponent.isZero() )
    {
        EXTRAP::SimpleTerm term;
        term.setFunctionType( logarithm );
        term.setExponent( m_current_log_exponent.eval() );
        compoundTerm.addSimpleTerm( term );
    }
    compoundTerm.setCoefficient( 1 );
    SingleParameterFunction* simpleFunction = new SingleParameterFunction();
    simpleFunction->addCompoundTerm( compoundTerm );
    return simpleFunction;
}

bool
SingleParameterRefiningFunctionModeler::compareHypotheses( const SingleParameterHypothesis& old, const SingleParameterHypothesis& candidate, const std::vector<DataPoint>& modeledDataPointList  )
{
    // check if contribution of each term is big enough
    std::vector<EXTRAP::CompoundTerm> ct = candidate.getFunction()->getCompoundTerms();
    for ( int i = 0; i < ct.size(); i++ )
    {
        if ( ct[ i ].getCoefficient() == 0 || candidate.calculateMaximalTermContribution( i, modeledDataPointList ) < this->m_eps )
        {
            // This hypothesis is not worth considering, because one of the terms does not actually contribute to the
            // function value in a sufficient way. We have already seen another hypothesis which contains the remaining
            // terms, so we can ignore this one.
            return false;
        }
    }

    // use RSS here, and SMAPE only for decisions related to the threshold
    if ( candidate.getRSS() < old.getRSS() )
    {
        if ( m_step == REFINING_STEP_INITIAL_LOG || m_step == REFINING_STEP_REFINEMENT_LOG )
        {
            m_current_best_exponent = m_current_log_exponent;
        }
        else
        {
            m_current_best_exponent = m_current_poly_exponent;
        }
        return true;
    }
    else
    {
        return false;
    }
}
}; // Close namespace