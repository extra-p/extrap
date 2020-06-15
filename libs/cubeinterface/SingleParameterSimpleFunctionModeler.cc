#include "SingleParameterSimpleFunctionModeler.h"
#include "Utilities.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <cassert>
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
SingleParameterSimpleFunctionModeler::SingleParameterSimpleFunctionModeler()
{
    m_max_term_count = 1;
    //generateDefaultHypothesisBuildingBlocks();
}

void
SingleParameterSimpleFunctionModeler::addHypothesisBuildingBlock( CompoundTerm Term )
{
    this->m_hypotheses_building_blocks.push_back( Term );
    return;
}

void
SingleParameterSimpleFunctionModeler::generateHypothesisBuildingBlockSet
(
    const std::vector<double>& poly_exponents,
    const std::vector<double>& log_exponents
)
{
    this->m_hypotheses_building_blocks.clear();


    EXTRAP::SimpleTerm   logTerm, polyTerm;
    EXTRAP::FunctionType poly = EXTRAP::polynomial;
    EXTRAP::FunctionType log  = EXTRAP::logarithm;

    logTerm.setFunctionType( log );
    polyTerm.setFunctionType( poly );
    for ( int i = 0; i < log_exponents.size(); i++ )
    {
        logTerm.setExponent( log_exponents[ i ] );
        EXTRAP::CompoundTerm compoundTerm;
        compoundTerm.addSimpleTerm( logTerm );
        compoundTerm.setCoefficient( 1 );
        this->m_hypotheses_building_blocks.push_back( compoundTerm );
    }
    for ( int i = 0; i < poly_exponents.size(); i++ )
    {
        polyTerm.setExponent( poly_exponents[ i ] );
        EXTRAP::CompoundTerm compoundTerm;
        compoundTerm.addSimpleTerm( polyTerm );
        compoundTerm.setCoefficient( 1 );
        this->m_hypotheses_building_blocks.push_back( compoundTerm );
    }
    for ( int i = 0; i < log_exponents.size(); i++ )
    {
        logTerm.setExponent( log_exponents[ i ] );
        for ( int j = 0; j < poly_exponents.size(); j++ )
        {
            polyTerm.setExponent( poly_exponents[ j ] );
            EXTRAP::CompoundTerm compoundTerm;
            compoundTerm.addSimpleTerm( polyTerm );
            compoundTerm.addSimpleTerm( logTerm );
            compoundTerm.setCoefficient( 1 );
            this->m_hypotheses_building_blocks.push_back( compoundTerm );
        }
    }
    return;
}

void
SingleParameterSimpleFunctionModeler::generateDefaultHypothesisBuildingBlocks(bool allow_log)
{
    this->m_hypotheses_building_blocks.clear();

    /**
     * x^(a/b)*log(x)^(c)
     * fromLegacy(a,b,c)
     **/

    if(allow_log==true){
        //std::cout << "log terms allowed!" << std::endl;
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 0, 1, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 0, 1, 2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 4, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 3, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 4, 2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 3, 2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 2, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 2, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 2, 2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 2, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 3, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 2, 3, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 3, 4, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 4, 5, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 2, 3, 2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 3, 4, 2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 1, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 1, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 1, 2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 5, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 5, 4, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 4, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 4, 3, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 3, 2, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 3, 2, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 3, 2, 2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 5, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 7, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 2, 1, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 2, 1, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 2, 1, 2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 9, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 7, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 5, 2, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 5, 2, 1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 5, 2, 2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 8, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 11, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 3, 1, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 3, 1, 1 ) );

        //negative exponents poly and log
        /**
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -0, 1, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -0, 1, -2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 4, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 3, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 4, -2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 3, -2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 2, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 2, -2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -2, 3, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -3, 4, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -2, 3, -2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -3, 4, -2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 1, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 1, -2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -5, 4, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -4, 3, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -3, 2, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -3, 2, -2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -2, 1, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -2, 1, -2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -5, 2, -1 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -5, 2, -2 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -3, 1, -1 ) );
        **/
    }
    else
    {
        //std::cout << "no log terms allowed!" << std::endl;
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 2, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 2, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 3, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 4, 5, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 1, 1, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 5, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 4, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 3, 2, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 5, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 7, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 2, 1, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 9, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 7, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 5, 2, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 8, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 11, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( 3, 1, 0 ) );

        // negative exponents
        /**
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 2, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -2, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -3, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -4, 5, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -1, 1, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -5, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -4, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -3, 2, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -5, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -7, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -2, 1, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -9, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -7, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -5, 2, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -8, 3, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -11, 4, 0 ) );
        this->m_hypotheses_building_blocks.push_back( CompoundTerm::fromLegacy( -3, 1, 0 ) );
        **/
    }

    //DEBUG
    /*
    for (int i = 0; i < this->m_hypotheses_building_blocks.size(); i++)
    {
        CompoundTerm ct = this->m_hypotheses_building_blocks.at(i);
        std::cout << "term: " << ct.getAsString("x") << std::endl;
    }
    */

    return;
}

void
SingleParameterSimpleFunctionModeler::printHypothesisBuildingBlocks( void )
{
    std::stringstream lineBuffer;

    DebugStream << "Printing Hypotheses Building Blocks: " << std::endl;
    for ( int i = 0; i < this->m_hypotheses_building_blocks.size(); i++ )
    {
        lineBuffer.str( "" ); // clear line buffer
        lineBuffer << "Hypothesis Building Block [" << i << "]: 1";
        for ( int j = 0; j < this->m_hypotheses_building_blocks[ i ].getSimpleTerms().size(); j++ )
        {
            lineBuffer << this->m_hypotheses_building_blocks[ i ].getSimpleTerms()[ j ].getAsString( "p" );
        }
        DebugStream << "    " << lineBuffer.str() << std::endl;
    }
}

void
SingleParameterSimpleFunctionModeler::setMaxTermCount( int term_count )
{
    this->m_max_term_count = term_count;
}

int
SingleParameterSimpleFunctionModeler::getMaxTermCount( void ) const
{
    return this->m_max_term_count;
}

SingleParameterHypothesis
SingleParameterSimpleFunctionModeler::createModel( const Experiment* experiment, const ModelGeneratorOptions& options, const std::vector<DataPoint>& modeledDataPointList,
                                                   ModelCommentList&             comments,
                                                   const Function*               expectationFunction )
{

    //get bool value from modeler options...
    ModelGeneratorOptions m_options = options;
    bool allow_log = m_options.getUseLogTerms();

    //debug flag
    //std::cout << "allow log?:" << allow_log << std::endl;
    
    // create building blocks for modeler
    generateDefaultHypothesisBuildingBlocks(allow_log);

    this->m_current_term_count = 0;

    //Step 1. Compute constant model
    SingleParameterHypothesis constantHypothesis;

    {
        // NOTE: new scope so one can't refer to (potentially invalidated) constantFunction later
        SingleParameterFunction* constantFunction = this->createConstantModel( modeledDataPointList );
        // TODO: also use cross-validation at this point if enabled?
        constantHypothesis = SingleParameterHypothesis( constantFunction );
        constantHypothesis.computeCost( modeledDataPointList );
    }

    double constantCost = constantHypothesis.getRSS();
    DebugStream << "Computed Constant Model Cost: " << constantCost << std::endl;

    SingleParameterHypothesis bestHypothesis, newBestHypothesis;
    if ( constantCost == 0 )
    {
        // Optimization: return constant hypothesis if it is a perfect model
        bestHypothesis = constantHypothesis;
    }
    else
    {
        //Step 2. Iteratively create model with parameters
        this->m_current_term_count++;
        bestHypothesis = this->findBestHypothesis( modeledDataPointList, constantHypothesis );
        // NOTE: the function in constantHypothesis is now invalid (set to NULL)
        bestHypothesis.computeAdjustedRSquared( constantCost, modeledDataPointList );

        while ( this->m_current_term_count < this->m_max_term_count )
        {
            this->m_current_term_count++;

            bestHypothesis.keepFunctionAlive(); // must be deleted manually later
            newBestHypothesis = this->findBestHypothesis( modeledDataPointList, bestHypothesis );
            // NOTE: the function previously stored inside bestHypothesis might be deleted (if not explicitly kept alive)
            newBestHypothesis.computeAdjustedRSquared( constantCost, modeledDataPointList );
            DebugStream << "Comparing AR2, old: " << bestHypothesis.getAR2() << ", new: " << newBestHypothesis.getAR2() << std::endl;
            if ( newBestHypothesis.getAR2() < bestHypothesis.getAR2() )
            {
                assert( newBestHypothesis.getFunction() != bestHypothesis.getFunction() ); // these instances should never be the same
                newBestHypothesis.freeFunction();
                break;
            }
            else
            {
                if ( newBestHypothesis.getFunction() != bestHypothesis.getFunction() )
                {
                    bestHypothesis.freeFunction();
                }
                bestHypothesis = newBestHypothesis;
            }
        }
    }
    return bestHypothesis;
}


bool
SingleParameterSimpleFunctionModeler::initSearchSpace( void )
{
    // ensure that the configuration of the search space makes sense

    //Fix for debug!
    //std::cout << "building block size: " << this->m_hypotheses_building_blocks.size() << std::endl;

    assert( this->m_hypotheses_building_blocks.size() > 0 );
    assert( this->m_max_term_count > 0 );

    this->m_current_hypothesis = 0;
    this->m_current_hypothesis_building_block_vector.resize( this->m_current_term_count, 0 );
    for ( int i = this->m_current_hypothesis_building_block_vector.size() - 1; i >= 0; i-- )
    {
        this->m_current_hypothesis_building_block_vector[ this->m_current_hypothesis_building_block_vector.size() - 1 - i ] = i;
        this->m_current_hypothesis                                                                                         += i * std::pow( ( double )m_hypotheses_building_blocks.size(), ( int )this->m_current_hypothesis_building_block_vector.size() - 1 - i );
    }

    return !this->m_hypotheses_building_blocks.empty();
}

bool
SingleParameterSimpleFunctionModeler::nextHypothesis( void )
{
    bool not_found = true;

    while ( not_found )
    {
        not_found = false;

        this->m_current_hypothesis++;
        int index = this->m_current_hypothesis;

        if ( index >= std::pow( ( double )m_hypotheses_building_blocks.size(), ( int )m_current_hypothesis_building_block_vector.size() ) )
        {
            return false;
        }
        DebugStream << "Printing index " << index << std::endl;

        for ( int i = 0; i < this->m_current_hypothesis_building_block_vector.size(); i++ )
        {
            this->m_current_hypothesis_building_block_vector[ i ] = index % m_hypotheses_building_blocks.size();
            index                                                 = index / m_hypotheses_building_blocks.size();

            for ( int j = 0; j < i; j++ )
            {
                if ( this->m_current_hypothesis_building_block_vector[ i ] == this->m_current_hypothesis_building_block_vector[ j ] )
                {
                    not_found = true;
                }
            }
        }
        if ( !not_found )
        {
            return true;
        }
    }

    //if we are here, something horrible has happened.
    assert( false );
}

SingleParameterFunction*
SingleParameterSimpleFunctionModeler::buildCurrentHypothesis( void )
{
    SingleParameterFunction* simpleFunction = new SingleParameterFunction();
    for ( int i = 0; i < this->m_current_hypothesis_building_block_vector.size(); i++ )
    {
        simpleFunction->addCompoundTerm( this->m_hypotheses_building_blocks[ this->m_current_hypothesis_building_block_vector[ i ] ] );
    }
    return simpleFunction;
}

const std::vector<CompoundTerm>&
SingleParameterSimpleFunctionModeler::getBuildingBlocks() const
{
    return this->m_hypotheses_building_blocks;
}
}; // Close namespace