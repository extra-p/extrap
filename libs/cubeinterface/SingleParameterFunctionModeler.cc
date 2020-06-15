#include "SingleParameterModelGenerator.h"
#include "SingleParameterHypothesis.h"
#include "SingleParameterFunction.h"
#include "Utilities.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cassert>

namespace EXTRAP
{
SingleParameterFunctionModeler::SingleParameterFunctionModeler()
{
    m_crossvalidation = CROSSVALIDATION_NONE;
}

SingleParameterFunction*
SingleParameterFunctionModeler::createConstantModel( const std::vector<DataPoint>& modeledDataPointList )

{
    SingleParameterFunction* constantFunction = new SingleParameterFunction();

    DebugStream << "Creating constant model." << std::endl;

    double meanModel = 0;

    for ( int i = 0; i < modeledDataPointList.size(); i++ )
    {
        meanModel += modeledDataPointList[ i ].getValue() / ( double )modeledDataPointList.size();
    }

    constantFunction->setConstantCoefficient( meanModel );

    return constantFunction;
}

void
SingleParameterFunctionModeler::setCrossvalidationMethod( Crossvalidation cvMethod )
{
    this->m_crossvalidation = cvMethod;
}

Crossvalidation
SingleParameterFunctionModeler::getCrossvalidationMethod() const
{
    return this->m_crossvalidation;
}

SingleParameterHypothesis
SingleParameterFunctionModeler::findBestHypothesis( const std::vector<DataPoint>& modeledDataPointList, SingleParameterHypothesis& initialHypothesis )
{
    SingleParameterHypothesis bestHypothesis = initialHypothesis;
    this->initSearchSpace();

    const double CLEAN_CONSTANT_EPSILON = 1e-3;

    int hypothesesCount = 0; // count how many hypotheses we have checked

    do
    {
        SingleParameterFunction*  nextFunction   = this->buildCurrentHypothesis();
        SingleParameterHypothesis nextHypothesis = SingleParameterHypothesis( nextFunction );
        //std::cout << "Hypothesis: " << nextFunction->getAsString( "p" ) << std::endl;

        std::vector<DataPoint> temporaryDataPointList;
        switch ( this->m_crossvalidation )
        {
            case CROSSVALIDATION_NONE:
                nextHypothesis.estimateParameters( modeledDataPointList );
                nextHypothesis.cleanConstantCoefficient( CLEAN_CONSTANT_EPSILON, modeledDataPointList );
                nextHypothesis.computeCost( modeledDataPointList );
                //std::cout << "Estimated Parameters: " << nextFunction->getAsString( "p" ) << std::endl;
                break;
            case CROSSVALIDATION_LEAVE_ONE_OUT:
                for ( int i = 0; i < modeledDataPointList.size(); i++ )
                {
                    temporaryDataPointList.push_back( modeledDataPointList.at( i ) );
                }
                nextHypothesis.setRSS( 0 );
                nextHypothesis.setrRSS( 0 );
                nextHypothesis.setSMAPE( 0 );
                for ( int i = 0; i < modeledDataPointList.size(); i++ )
                {
                    temporaryDataPointList.erase( temporaryDataPointList.begin() + i );
                    nextHypothesis.estimateParameters( temporaryDataPointList );
                    nextHypothesis.cleanConstantCoefficient( CLEAN_CONSTANT_EPSILON, temporaryDataPointList );
                    nextHypothesis.computeCostLeaveOneOutCrossValidation( temporaryDataPointList, modeledDataPointList.at( i ) );
                    temporaryDataPointList.insert( temporaryDataPointList.begin() + i, modeledDataPointList.at( i ) );
                }
                temporaryDataPointList.clear();
                break;
            default:
                ErrorStream << "Invalid crossvalidation parameter. Using NO crossvalidation " << std::endl;
                nextHypothesis.estimateParameters( modeledDataPointList );
                nextHypothesis.cleanConstantCoefficient( CLEAN_CONSTANT_EPSILON, modeledDataPointList );
                nextHypothesis.computeCost( modeledDataPointList );
                //std::cout << "Estimated Parameters: " << nextFunction->getAsString( "p" ) << std::endl;
                break;
        }
        double cost = nextHypothesis.getRSS();
        //std::cout << "Model Cost: " << cost << std::endl;

        if ( !nextHypothesis.isValid() )
        {
            //std::cout << "WARNING: Model is invalid (due to numeric imprecision) and will be ignored." << std::endl;
            nextHypothesis.freeFunction();
        }
        else if ( compareHypotheses( bestHypothesis, nextHypothesis, modeledDataPointList ) )
        {
            bestHypothesis.freeFunction();
            bestHypothesis = nextHypothesis;
        }
        else
        {
            nextHypothesis.freeFunction();
        }
        hypothesesCount++;
    }
    while ( this->nextHypothesis() );

    //std::cout << "Evaluated " << hypothesesCount << " models, chosen best one: " << bestHypothesis.getFunction()->getAsString( "p" ) << std::endl;
    //std::cout << "Best Model Cost: " << bestHypothesis.getRSS() << std::endl;

    return bestHypothesis;
}

bool
SingleParameterFunctionModeler::compareHypotheses( const SingleParameterHypothesis& old, const SingleParameterHypothesis& candidate, const std::vector<DataPoint>& modeledDataPointList  )
{
    //this->m_eps = 0.0005;
    // check if contribution of each term is big enough
    std::vector<EXTRAP::CompoundTerm> ct = candidate.getFunction()->getCompoundTerms();
    for ( int i = 0; i < ct.size(); i++ )
    {
        if ( ct[ i ].getCoefficient() == 0 || candidate.calculateMaximalTermContribution( i, modeledDataPointList ) < this->m_eps )
        {
            // This hypothesis is not worth considering, because one of the terms does not actually contribute to the
            // function value in a sufficient way. We have already seen another hypothesis which contains the remaining
            // terms, so we can ignore this one.
            //std::cout << this->m_eps << "\n";
            //std::cout <<  candidate.calculateMaximalTermContribution( i, modeledDataPointList ) << "\n";
            return false;
        }
    }

    return candidate.getRSS() < old.getRSS();
}

/*static*/ void
SingleParameterFunctionModeler::getSingleParameterInterval( const std::vector<DataPoint>& modeledDataPointList, Value* low, Value* high )
{
    assert( modeledDataPointList.size() > 0 );
    assert( modeledDataPointList[ 0 ].getParameterList().size() == 1 );

    // extract value of first (and only) parameter
    *low  = modeledDataPointList[ 0 ].getParameterList().begin()->second;
    *high = *low;

    for ( int i = 1; i < modeledDataPointList.size(); i++ )
    {
        Value val = modeledDataPointList[ i ].getParameterList().begin()->second;
        if ( val < *low )
        {
            *low = val;
        }
        if ( val > *high )
        {
            *high = val;
        }
    }
}

void
SingleParameterFunctionModeler::setEpsilon( Value eps )
{
    this->m_eps = eps;
}

Value
SingleParameterFunctionModeler::getEpsilon( void ) const
{
    return this->m_eps;
}
}; // Close namespace