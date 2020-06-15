#include "MultiParameterHypothesis.h"
#include "Utilities.h"
#include <cmath>
#include <sstream>
#include <limits>
#include <cassert>

namespace EXTRAP
{
MultiParameterHypothesis::MultiParameterHypothesis( void ) : m_function( NULL ), m_function_refcount( NULL )
{
}

MultiParameterHypothesis::MultiParameterHypothesis( MultiParameterFunction* f )
{
    this->m_function                 = f;
    this->m_function_refcount        = new HypothesisFunctionRefcount();
    this->m_function_refcount->count = 1; // Initially referenced only inside this hypothesis
    this->m_function_refcount->func  = f;
    DebugStream << "### Initialized refcount to 1 for " << this->m_function << " at " << this->m_function_refcount << "\n";
}

MultiParameterFunction*
MultiParameterHypothesis::getFunction( void ) const
{
    return this->m_function;
}

double
MultiParameterHypothesis::getRSS( void ) const
{
    return this->m_RSS;
}

void
MultiParameterHypothesis::setRSS( double rss )
{
    this->m_RSS = rss;
}

double
MultiParameterHypothesis::getrRSS( void ) const
{
    return this->m_rRSS;
}

void
MultiParameterHypothesis::setrRSS( double rrss )
{
    this->m_rRSS = rrss;
}


double
MultiParameterHypothesis::getAR2( void ) const
{
    return this->m_AR2;
}

void
MultiParameterHypothesis::setAR2( double ar2 )
{
    this->m_AR2 = ar2;
}

double
MultiParameterHypothesis::getSMAPE( void ) const
{
    return this->m_SMAPE;
}

void
MultiParameterHypothesis::setSMAPE( double smape )
{
    this->m_SMAPE = smape;
}

double
MultiParameterHypothesis::getRE( void ) const
{
    return this->m_RE;
}

void
MultiParameterHypothesis::setRE( double re )
{
    this->m_RE = re;
}

std::vector<double>
MultiParameterHypothesis::getPredictedPoints( void )
{
    return this->m_predicted_points;
}

std::vector<double>
MultiParameterHypothesis::getActualPoints( void )
{
    return this->m_actual_points;
}

std::vector<double>
MultiParameterHypothesis::getPs( void )
{
    return this->m_ps;
}

std::vector<double>
MultiParameterHypothesis::getSizes( void )
{
    return this->m_sizes;
}

void
MultiParameterHypothesis::computeCost( const std::vector<DataPoint>& points )
{
    this->m_RSS  = 0;
    this->m_rRSS = 0;
    double smape  = 0;
    double re_sum = 0;
    std::vector<double> predicted_points;
    std::vector<double> actual_points;
    std::vector<double> ps;
    std::vector<double> sizes;

    for ( int i = 0; i < points.size(); i++ )
    {
        double predicted           = this->m_function->evaluate( points[ i ].getParameterList() );

        const ParameterValueList& list = points[ i ].getParameterList();
        Parameter p = Parameter("p");
        Parameter s = Parameter("size");
        ParameterValueList::const_iterator it = list.find(p);
        double processors = it->second;
        //std::cout << "Processors: " << processors << "\n";
        ParameterValueList::const_iterator it2 = list.find(s);
        double size = it2->second;
        //std::cout << "Size: " << size << "\n";


        predicted_points.push_back(predicted);
        double actual              = points[ i ].getValue();

        //std::cout << "predicted: " << predicted << "\n";
        //std::cout << "actual: " << actual << "\n";
        ps.push_back(processors);
        sizes.push_back(size);

        actual_points.push_back(actual);
        double difference          = predicted - actual;
        double absolute_difference = fabs( difference );
        double abssum              = fabs( actual ) + fabs( predicted );

        // calculate relative error
        double absolute_error = fabs( predicted - actual );
        //double tmp = std::max(absolute_error, actual);
        double relative_error = absolute_error / actual;
        re_sum = re_sum + relative_error;

        this->m_RSS += difference * difference;
        double relativeDifference = difference / points[ i ].getValue();
        this->m_rRSS += relativeDifference * relativeDifference;

        if ( abssum != 0.0 )
        {
            // This `if` condition prevents a division by zero, but it is correct: if sum is 0, both `actual` and `predicted`
            // must have been 0, and in that case the error at this point is 0, so we don't need to add anything.
            smape += fabs( difference ) / abssum * 2;
        }
    }
    //times 100 for percentage error
    this->m_RE    = re_sum / points.size();
    this->m_SMAPE = smape / points.size() * 100;
    this->m_predicted_points = predicted_points;
    this->m_actual_points = actual_points;
    this->m_ps = ps;
    this->m_sizes = sizes;
}

void
MultiParameterHypothesis::computeAdjustedRSquared( double TSS, const std::vector<DataPoint>& points )
{
    this->m_AR2 = 0.0;
    double adjR = 1.0 - ( this->m_RSS / TSS );

    double counter = 0;
    for ( int i = 0; i < this->m_function->getMultiParameterTerms().size(); i++ )
    {
        counter += m_function->getMultiParameterTerms()[ i ].getCompoundTermParameterPairs().size();
    }

    double degrees_freedom = points.size() - counter - 1;

    this->m_AR2 = ( 1.0 - ( 1.0 - adjR ) * ( points.size() - 1.0 ) / degrees_freedom );

    return;
}

void
MultiParameterHypothesis::computeCostLeaveOneOutCrossValidation( const std::vector<DataPoint>& points, DataPoint missing )
{
    double predicted  = this->m_function->evaluate( missing.getParameterList() );
    double actual     = missing.getValue();
    double difference = predicted - actual;
    this->m_RSS += difference * difference;
    double relativeDifference = difference / missing.getValue();
    this->m_rRSS += relativeDifference * relativeDifference;
    double abssum = fabs( actual ) + fabs( predicted );
    if ( abssum != 0.0 )
    {
        this->m_SMAPE += ( fabs( difference ) / abssum * 2 ) / points.size() * 100;
    }
}

void
MultiParameterHypothesis::estimateParameters( const std::vector<DataPoint>& points )
{
    assert( m_function != NULL );

    int hypothesisTotalTerms = this->m_function->getMultiParameterTerms().size() + 1;
    int experimentCount      = points.size();

    double*             M              = new double[ hypothesisTotalTerms * hypothesisTotalTerms ];
    double*             f              = new double[ hypothesisTotalTerms ];
    double*             rhs            = new double[ experimentCount ];
    double*             regularization = new double[ hypothesisTotalTerms ];
    std::vector<double> AT;
    AT.resize( hypothesisTotalTerms * experimentCount );

    for ( int i = 0; i < AT.size(); i++ )
    {
        AT[ i ] = 0;
    }
    for ( int i = 0; i < hypothesisTotalTerms; i++ )
    {
        int index = i * experimentCount;
        for ( int rhs_index = 0; rhs_index < experimentCount; rhs_index++ )
        {
            AT[ index ] = 1;

            double eval;
            if ( i > 0 )
            {
                MultiParameterTerm& multiParameterTerm = ( this->m_function->getMultiParameterTerms() )[ i - 1 ];

                multiParameterTerm.setCoefficient( 1 ); // reset coefficient

                for ( int j = 0; j < multiParameterTerm.getCompoundTermParameterPairs().size(); j++ )
                {
                    CompoundTerm t = ( multiParameterTerm.getCompoundTermParameterPairs() )[ j ].first;
                    t.setCoefficient( 1.0 );
                }

                eval = multiParameterTerm.evaluate( points[ rhs_index ].getParameterList() );
            }
            else
            {
                eval = 1;
            }
            AT[ index ] *= eval;
            index++;
            rhs[ rhs_index ] = points[ rhs_index ].getValue();
        }
    }

    for ( int i = 0; i < hypothesisTotalTerms; i++ )
    {
        double reg = 1;
        int    tmp = floor( log10( AT[ experimentCount * i + experimentCount - 1 ] ) );
        if ( tmp > 0 )
        {
            reg = pow( 10.0f, tmp );
        }
        regularization[ i ] = reg;
        for ( int j = 0; j < experimentCount; j++ )
        {
            int index = experimentCount * i + j;
            AT[ index ] *= reg;
        }
    }

    for ( int i = 0; i < hypothesisTotalTerms; i++ )
    {
        int row = i * experimentCount;
        for ( int j = 0; j < hypothesisTotalTerms; j++ )
        {
            int t_row = experimentCount * j;
            int index = i * hypothesisTotalTerms + j;
            M[ index ] = 0;
            for ( int k = 0; k < experimentCount; k++ )
            {
                M[ index ] += AT[ row + k ] * AT[ t_row + k ];
            }
        }
    }
    for ( int i = 0; i < hypothesisTotalTerms; i++ )
    {
        int row = i * experimentCount;
        f[ i ] = 0;
        for ( int k = 0; k < experimentCount; k++ )
        {
            f[ i ] += AT[ row + k ] * rhs[ k ];
        }
    }

    matrixInverse( M, hypothesisTotalTerms );

    for ( int i = 0; i < hypothesisTotalTerms; i++ )
    {
        int irow = i * hypothesisTotalTerms;
        if ( i == 0 )
        {
            this->m_function->setConstantCoefficient( 0 );
        }
        else
        {
            ( this->m_function->getMultiParameterTerms() )[ i - 1 ].setCoefficient( 0 );
        }

        for ( int j = 0; j < hypothesisTotalTerms; j++ )
        {
            if ( i == 0 )
            {
                this->m_function->setConstantCoefficient( this->m_function->getConstantCoefficient() +  ( M[ irow + j ] * f[ j ] ) );
            }
            else
            {
                ( this->m_function->getMultiParameterTerms() )[ i - 1 ].setCoefficient( ( this->m_function->getMultiParameterTerms() )[ i - 1 ].getCoefficient() + ( M[ irow + j ] * f[ j ] ) );
            }
        }
        if ( i == 0 )
        {
            this->m_function->setConstantCoefficient( this->m_function->getConstantCoefficient() * regularization[ i ] );
        }
        else
        {
            ( this->m_function->getMultiParameterTerms() )[ i - 1 ].setCoefficient( ( this->m_function->getMultiParameterTerms() )[ i - 1 ].getCoefficient() * regularization[ i ] );
        }
    }

    delete[] M;
    delete[] f;
    delete[] rhs;
    delete[] regularization;
    AT.clear();
    return;
}

bool
MultiParameterHypothesis::isValid( void )
{
    double cost = m_RSS;
    return !( cost != cost || fabs( cost ) == std::numeric_limits<double>::infinity() );
}

double
MultiParameterHypothesis::calculateMaximalTermContribution( int termIndex, const std::vector<DataPoint>& points ) const
{
    std::vector<EXTRAP::MultiParameterTerm> mt = m_function->getMultiParameterTerms();
    assert( termIndex < mt.size() );
    double max = 0;
    for ( int i = 0; i < points.size(); i++ )
    {
        double contribution = fabs( mt[ termIndex ].evaluate( points[ i ].getParameterList() ) / points[ i ].getValue() );
        if ( contribution > max )
        {
            max = contribution;
        }
    }
    return max;
}

void
MultiParameterHypothesis::cleanConstantCoefficient( double epsilon, const std::vector<DataPoint>& points )
{
    // This function is used to correct numerical imprecision in the matrix caculations,
    // when the constant coefficient should be zero but is instead very small.
    // We take into account the minimum data value to make sure that we don't "nullify"
    // actually relevant numbers.
    std::vector<DataPoint>::const_iterator it      = points.begin();
    double                                 minimum = it->getValue();
    ++it;
    for (; it != points.end(); ++it )
    {
        if ( it->getValue() < minimum )
        {
            minimum = it->getValue();
        }
    }

    if ( fabs( this->m_function->getConstantCoefficient() / minimum ) < epsilon )
    {
        this->m_function->setConstantCoefficient( 0 );
    }
}

void
MultiParameterHypothesis::keepFunctionAlive( void )
{
    assert( this->m_function_refcount != NULL );
    assert( this->m_function_refcount->func == this->m_function );
    this->m_function_refcount->count++;

    //DebugStream << "Incremented external function refcount (" << this->m_function << ") at " << this->m_function_refcount << " to " << this->m_function_refcount->count << "\n";
}

void
MultiParameterHypothesis::freeFunction( void )
{
    assert( this->m_function_refcount != NULL );
    assert( this->m_function_refcount->func == this->m_function );
    ( this->m_function_refcount->count )--;
    //DebugStream << "Decremented external function refcount (" << this->m_function << ") at " << this->m_function_refcount << " to " << this->m_function_refcount->count << "\n";
    if ( this->m_function_refcount->count == 0 )
    {
        DebugStream << "### Deleting function (" << this->m_function << ") with refcount at " << this->m_function_refcount << "\n";
        delete this->m_function;
        delete this->m_function_refcount;
        this->m_function          = NULL;
        this->m_function_refcount = NULL;
    }
}
};