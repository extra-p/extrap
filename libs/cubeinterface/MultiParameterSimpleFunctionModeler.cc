#include "MultiParameterSimpleFunctionModeler.h"
#include "SingleParameterHypothesis.h"
#include "ModelGeneratorOptions.h"
#include <sstream>
#include "Experiment.h"
#include "Coordinate.h"

namespace EXTRAP
{
const std::string BAD_MODEL               = "This model has a large error metric (SMAPE). Probably the measurements can not be explained by PMNF.";
const std::string BETTER_USER_EXPECTATION = "Expectation function specified by user is a better model than the best hypothesis.";

MultiParameterHypothesis
MultiParameterSimpleFunctionModeler::createModel( const Experiment*             experiment,
                                                  const ModelGeneratorOptions&  options,
                                                  const std::vector<DataPoint>& modeledDataPointList,
                                                  ModelCommentList&             comments,
                                                  const Function*               expectationFunction )
{
    double constantCost = 0;

    double meanModel = 0;

    for ( int i = 0; i < modeledDataPointList.size(); i++ )
    {
        meanModel += modeledDataPointList[ i ].getValue() / ( double )modeledDataPointList.size();
    }
    for ( int i = 0; i < modeledDataPointList.size(); i++ )
    {
        constantCost += ( modeledDataPointList[ i ].getValue() - meanModel ) * ( modeledDataPointList[ i ].getValue() - meanModel );
    }
    //constantCost
    //First the data points have to be split by parameter

    std::vector<std::vector<DataPoint> > singleParamDataPointLists;
    int                                  parameterNumbers;
    std::vector<EXTRAP::Parameter>       params, tmpparams;
    assert( modeledDataPointList.size() > 0 );
    parameterNumbers = modeledDataPointList[ 0 ].getParameterList().size();
    for ( int i = 1; i < modeledDataPointList.size(); i++ )
    {
        assert( parameterNumbers == modeledDataPointList[ i ].getParameterList().size() );
    }
    for ( EXTRAP::ParameterValueList::const_iterator it = modeledDataPointList[ 0 ].getParameterList().begin(); it != modeledDataPointList[ 0 ].getParameterList().end(); it++ )
    {
        const Parameter p = it->first;
        params.push_back( p );
        tmpparams.push_back( p );
        std::vector<DataPoint> singleParamDataPointList;
        std::vector<bool>      valueUsed( modeledDataPointList.size(), false );
        for ( int j = 0; j < modeledDataPointList.size(); j++ )
        {
            if ( valueUsed[ j ] == true )
            {
                continue;
            }
            Value v           = 0;
            int   sampleCount = 0;
            for ( int k = j; k < modeledDataPointList.size(); k++ )
            {
                if ( modeledDataPointList[ j ].getParameterValue( p ) == modeledDataPointList[ k ].getParameterValue( p ) )
                {
                    v += modeledDataPointList[ j ].getValue();
                    sampleCount++;
                    valueUsed[ k ] = true;
                }
            }
            v = v / ( double )sampleCount;
            ParameterValueList* temporaryPVL;
            temporaryPVL = new ParameterValueList();
            Interval interval;
            interval.end   = 0;
            interval.start = 0;
            temporaryPVL->insert( std::make_pair( p, modeledDataPointList[ j ].getParameterValue( p ) ) );
            const DataPoint t( temporaryPVL, sampleCount, v, interval );
            singleParamDataPointList.push_back( t );
        }
        singleParamDataPointLists.push_back( singleParamDataPointList );
    }
    //The datapoints are now split by parameter and sorted into the singleParamDataPoints vector
    std::vector<EXTRAP::SingleParameterHypothesis> bestSingleParameterHypotheses;
    std::vector<int>                               paramsToDelete;
    std::vector<int>                               paramsToKeep;
    ModelCommentList                               single_param_comments;
    ParameterList plist = experiment->getParameters();
    //Create single parameter hypothesis for all parameters
    for ( int i = 0; i < singleParamDataPointLists.size(); i++ )
    {
        //check if the parameter value is smaller than 1, then do not allow log terms...
        Parameter p = plist.at(i);
        bool allow_log = true;
        CoordinateList clist =  experiment->getCoordinates();
        for (int j = 0; j < clist.size(); j++){
            Coordinate* c = clist.at(j);
            std::string cstring = c->toString();
            std::string pstring = p.getName();
            int pos = cstring.find(pstring);
            std::string value = cstring.substr(pos,cstring.size());
            pos = value.find(",");
            value = value.substr(pos+1,value.size());
            pos = value.find(")");
            value = value.substr(0,pos);
            double v = -1;
            std::istringstream iss( value );
            iss >> v;
            if(v<1){
                allow_log = false;
                break;
            }
        }
        // adjust the modeler options accordingly
        ModelGeneratorOptions m_options = options;
        m_options.setUseLogTerms(allow_log);
        EXTRAP::SingleParameterHypothesis hyp = m_single_parameter_function_modeler->createModel( experiment, m_options, singleParamDataPointLists[ i ], single_param_comments, expectationFunction );

        if ( hyp.getFunction()->getCompoundTerms().size() > 0 )
        {
            bestSingleParameterHypotheses.push_back( hyp );
            paramsToKeep.push_back( i );
        }
        else
        {
            paramsToDelete.push_back( i );
        }
    }

    if ( paramsToDelete.size() == params.size() )
    {
        MultiParameterFunction* constantFunction = new MultiParameterFunction();
        constantFunction->setConstantCoefficient( meanModel );
        MultiParameterHypothesis constantHypothesis( constantFunction );
        constantHypothesis.setRSS( constantCost );
        constantHypothesis.setAR2( 0 );
        constantHypothesis.setrRSS( 0 );
        constantHypothesis.setSMAPE( 0 );
        return constantHypothesis;
    }
    else if ( ( params.size() - paramsToDelete.size() ) == 1 )
    {
        MultiParameterFunction* simpleFunction = new MultiParameterFunction();
        MultiParameterTerm      t;
        t.addCompoundTermParameterPair( bestSingleParameterHypotheses[ 0 ].getFunction()->getCompoundTerms()[ 0 ], params[ paramsToKeep[ 0 ] ] );
        t.setCoefficient( bestSingleParameterHypotheses[ 0 ].getFunction()->getCompoundTerms()[ 0 ].getCoefficient() );
        simpleFunction->addMultiParameterTerm( t );
        MultiParameterHypothesis simpleHypothesis( simpleFunction );
        simpleHypothesis.getFunction()->setConstantCoefficient( bestSingleParameterHypotheses[ 0 ].getFunction()->getConstantCoefficient() );
        simpleHypothesis.computeCost( modeledDataPointList );
        return simpleHypothesis;
    }

    //Remove unneccessary parameters
    for ( int i = paramsToDelete.size() - 1; i >= 0; i-- )
    {
        params.erase( params.begin() + paramsToDelete[ i ] );
    }

    std::vector<MultiParameterHypothesis> hypotheses;

    //add Hypotheses for 2 parameter models
    if ( params.size() == 2 )
    {
        //x*y*z
        EXTRAP::MultiParameterTerm            mult;
        for ( int i = 0; i < bestSingleParameterHypotheses.size(); i++ )
        {
            EXTRAP::CompoundTerm ct;
            ct = bestSingleParameterHypotheses[ i ].getFunction()->getCompoundTerms()[ 0 ];
            ct.setCoefficient( 1 );
            mult.addCompoundTermParameterPair( ct, params[ i ] );
        }

        // x+y+z
        std::vector<EXTRAP::MultiParameterTerm> add;
        for ( int i = 0; i < bestSingleParameterHypotheses.size(); i++ )
        {
            EXTRAP::MultiParameterTerm tmp;
            EXTRAP::CompoundTerm       ct;
            ct = bestSingleParameterHypotheses[ i ].getFunction()->getCompoundTerms()[ 0 ];
            ct.setCoefficient( 1 );
            tmp.addCompoundTermParameterPair( ct, params[ i ] );
            add.push_back( tmp );
        }

        EXTRAP::MultiParameterFunction* f1, * f2, * f3, * f4;
        f1 = new  EXTRAP::MultiParameterFunction();
        f2 = new  EXTRAP::MultiParameterFunction();
        f3 = new  EXTRAP::MultiParameterFunction();
        f4 = new  EXTRAP::MultiParameterFunction();
        f1->addMultiParameterTerm( mult );
        f2->addMultiParameterTerm( add[ 0 ] );
        f2->addMultiParameterTerm( mult );
        f3->addMultiParameterTerm( add[ 1 ] );
        f3->addMultiParameterTerm( mult );
        f4->addMultiParameterTerm( add[ 0 ] );
        f4->addMultiParameterTerm( add[ 1 ] );
        MultiParameterHypothesis              mph1( f1 );
        MultiParameterHypothesis              mph2( f2 );
        MultiParameterHypothesis              mph3( f3 );
        MultiParameterHypothesis              mph4( f4 );
        hypotheses.push_back( mph1 );   //a*b
        hypotheses.push_back( mph2 ); //a*b+a
        hypotheses.push_back( mph3 ); //a*b+b
        hypotheses.push_back( mph4 );   //a+b
    }

    //add Hypotheses for 3 parameter models
    if ( params.size() == 3 )
    {
        //x*y*z
        EXTRAP::MultiParameterTerm            mult;
        for ( int i = 0; i < bestSingleParameterHypotheses.size(); i++ )
        {
            EXTRAP::CompoundTerm ct;
            ct = bestSingleParameterHypotheses[ i ].getFunction()->getCompoundTerms()[ 0 ];
            ct.setCoefficient( 1 );
            mult.addCompoundTermParameterPair( ct, params[ i ] );
        }

        //x*y
        EXTRAP::MultiParameterTerm mult_x_y;
        EXTRAP::CompoundTerm ct;
        ct = bestSingleParameterHypotheses[ 0 ].getFunction()->getCompoundTerms()[ 0 ];
        ct.setCoefficient( 1 );
        mult_x_y.addCompoundTermParameterPair( ct, params[ 0 ] );
        EXTRAP::CompoundTerm ct2;
        ct2 = bestSingleParameterHypotheses[ 1 ].getFunction()->getCompoundTerms()[ 0 ];
        ct2.setCoefficient( 1 );
        mult_x_y.addCompoundTermParameterPair( ct2, params[ 1 ] );

        //y*z
        EXTRAP::MultiParameterTerm mult_y_z;
        EXTRAP::CompoundTerm ct3;
        ct3 = bestSingleParameterHypotheses[ 1 ].getFunction()->getCompoundTerms()[ 0 ];
        ct3.setCoefficient( 1 );
        mult_y_z.addCompoundTermParameterPair( ct3, params[ 1 ] );
        EXTRAP::CompoundTerm ct4;
        ct4 = bestSingleParameterHypotheses[ 2 ].getFunction()->getCompoundTerms()[ 0 ];
        ct4.setCoefficient( 1 );
        mult_y_z.addCompoundTermParameterPair( ct4, params[ 2 ] );

        //x*z
        EXTRAP::MultiParameterTerm mult_x_z;
        EXTRAP::CompoundTerm ct5;
        ct5 = bestSingleParameterHypotheses[ 0 ].getFunction()->getCompoundTerms()[ 0 ];
        ct5.setCoefficient( 1 );
        mult_x_z.addCompoundTermParameterPair( ct5, params[ 0 ] );
        EXTRAP::CompoundTerm ct6;
        ct6 = bestSingleParameterHypotheses[ 2 ].getFunction()->getCompoundTerms()[ 0 ];
        ct6.setCoefficient( 1 );
        mult_x_z.addCompoundTermParameterPair( ct6, params[ 2 ] );

        // x+y+z
        std::vector<EXTRAP::MultiParameterTerm> add;
        for ( int i = 0; i < bestSingleParameterHypotheses.size(); i++ )
        {
            EXTRAP::MultiParameterTerm tmp;
            EXTRAP::CompoundTerm       ct;
            ct = bestSingleParameterHypotheses[ i ].getFunction()->getCompoundTerms()[ 0 ];
            ct.setCoefficient( 1 );
            tmp.addCompoundTermParameterPair( ct, params[ i ] );
            add.push_back( tmp );
        }

        EXTRAP::MultiParameterFunction* f0;
        EXTRAP::MultiParameterFunction* f1;
        EXTRAP::MultiParameterFunction* f2;
        EXTRAP::MultiParameterFunction* f3;
        EXTRAP::MultiParameterFunction* f4;
        EXTRAP::MultiParameterFunction* f5;
        EXTRAP::MultiParameterFunction* f6;
        EXTRAP::MultiParameterFunction* f7;
        EXTRAP::MultiParameterFunction* f8;
        EXTRAP::MultiParameterFunction* f9;
        EXTRAP::MultiParameterFunction* f10;
        EXTRAP::MultiParameterFunction* f11;
        EXTRAP::MultiParameterFunction* f12;
        EXTRAP::MultiParameterFunction* f13;
        EXTRAP::MultiParameterFunction* f14;
        EXTRAP::MultiParameterFunction* f15;
        EXTRAP::MultiParameterFunction* f16;
        EXTRAP::MultiParameterFunction* f17;
        EXTRAP::MultiParameterFunction* f18;
        EXTRAP::MultiParameterFunction* f19;
        EXTRAP::MultiParameterFunction* f20;
        EXTRAP::MultiParameterFunction* f21;
        EXTRAP::MultiParameterFunction* f22;

        f0 = new EXTRAP::MultiParameterFunction();
        f1 = new EXTRAP::MultiParameterFunction();
        f2 = new EXTRAP::MultiParameterFunction();
        f3 = new EXTRAP::MultiParameterFunction();
        f4 = new EXTRAP::MultiParameterFunction();
        f5 = new EXTRAP::MultiParameterFunction();
        f6 = new EXTRAP::MultiParameterFunction();
        f7 = new EXTRAP::MultiParameterFunction();
        f8 = new EXTRAP::MultiParameterFunction();
        f9 = new EXTRAP::MultiParameterFunction();
        f10 = new EXTRAP::MultiParameterFunction();
        f11 = new EXTRAP::MultiParameterFunction();
        f12 = new EXTRAP::MultiParameterFunction();
        f13 = new EXTRAP::MultiParameterFunction();
        f14 = new EXTRAP::MultiParameterFunction();
        f15 = new EXTRAP::MultiParameterFunction();
        f16 = new EXTRAP::MultiParameterFunction();
        f17 = new EXTRAP::MultiParameterFunction();
        f18 = new EXTRAP::MultiParameterFunction();
        f19 = new EXTRAP::MultiParameterFunction();
        f20 = new EXTRAP::MultiParameterFunction();
        f21 = new EXTRAP::MultiParameterFunction();
        f22 = new EXTRAP::MultiParameterFunction();

        // x*y*z
        f1->addMultiParameterTerm( mult );
        MultiParameterHypothesis mph1( f1 );

        // x+y+z
        f2->addMultiParameterTerm( add[ 0 ] );
        f2->addMultiParameterTerm( add[ 1 ] );
        f2->addMultiParameterTerm( add[ 2 ] );
        MultiParameterHypothesis mph2( f2 );

        // x*y*z+x
        f3->addMultiParameterTerm( mult );
        f3->addMultiParameterTerm( add[ 0 ] );
        MultiParameterHypothesis mph3( f3 );

        // x*y*z+y
        f4->addMultiParameterTerm( mult );
        f4->addMultiParameterTerm( add[ 1 ] );
        MultiParameterHypothesis mph4( f4 );

        // x*y*z+z
        f5->addMultiParameterTerm( mult );
        f5->addMultiParameterTerm( add[ 2 ] );
        MultiParameterHypothesis mph5( f5 );

        // x*y*z+x*y
        f6->addMultiParameterTerm( mult );
        f6->addMultiParameterTerm( mult_x_y );
        MultiParameterHypothesis mph6( f6 );

        // x*y*z+y*z
        f7->addMultiParameterTerm( mult );
        f7->addMultiParameterTerm( mult_y_z );
        MultiParameterHypothesis mph7( f7 );

        // x*y*z+x*z
        f8->addMultiParameterTerm( mult );
        f8->addMultiParameterTerm( mult_x_z );
        MultiParameterHypothesis mph8( f8 );

        // x*y*z+x*y+z
        f9->addMultiParameterTerm( mult );
        f9->addMultiParameterTerm( mult_x_y );
        f9->addMultiParameterTerm( add[ 2 ] );
        MultiParameterHypothesis mph9( f9 );

        // x*y*z+y*z+x
        f10->addMultiParameterTerm( mult );
        f10->addMultiParameterTerm( mult_y_z );
        f10->addMultiParameterTerm( add[ 0 ] );
        MultiParameterHypothesis mph10( f10 );

        // x*y*z+x*z+y
        f0->addMultiParameterTerm( mult );
        f0->addMultiParameterTerm( mult_x_z );
        f0->addMultiParameterTerm( add[ 1 ] );
        MultiParameterHypothesis mph0( f0 );

        // x*y*z+x+y
        f11->addMultiParameterTerm( mult );
        f11->addMultiParameterTerm( add[ 0 ] );
        f11->addMultiParameterTerm( add[ 1 ] );
        MultiParameterHypothesis mph11( f11 );

        // x*y*z+x+z
        f21->addMultiParameterTerm( mult );
        f21->addMultiParameterTerm( add[ 0 ] );
        f21->addMultiParameterTerm( add[ 2 ] );
        MultiParameterHypothesis mph21( f21 );

        // x*y*z+y+z
        f22->addMultiParameterTerm( mult );
        f22->addMultiParameterTerm( add[ 1 ] );
        f22->addMultiParameterTerm( add[ 2 ] );
        MultiParameterHypothesis mph22( f22 );

        // x*y+z
        f12->addMultiParameterTerm( mult_x_y );
        f12->addMultiParameterTerm( add[ 2 ] );
        MultiParameterHypothesis mph12( f12 );

        // x*y+z+y
        f13->addMultiParameterTerm( mult_x_y );
        f13->addMultiParameterTerm( add[ 2 ] );
        f13->addMultiParameterTerm( add[ 1 ] );
        MultiParameterHypothesis mph13( f13 );

        // x*y+z+x
        f14->addMultiParameterTerm( mult_x_y );
        f14->addMultiParameterTerm( add[ 2 ] );
        f14->addMultiParameterTerm( add[ 0 ] );
        MultiParameterHypothesis mph14( f14 );

        // x*z+y
        f15->addMultiParameterTerm( mult_x_z );
        f15->addMultiParameterTerm( add[ 1 ] );
        MultiParameterHypothesis mph15( f15 );

        // x*z+y+x
        f16->addMultiParameterTerm( mult_x_z );
        f16->addMultiParameterTerm( add[ 1 ] );
        f16->addMultiParameterTerm( add[ 0 ] );
        MultiParameterHypothesis mph16( f16 );

        // x*z+y+z
        f17->addMultiParameterTerm( mult_x_z );
        f17->addMultiParameterTerm( add[ 1 ] );
        f17->addMultiParameterTerm( add[ 2 ] );
        MultiParameterHypothesis mph17( f17 );

        // y*z+x
        f18->addMultiParameterTerm( mult_y_z );
        f18->addMultiParameterTerm( add[ 0 ] );
        MultiParameterHypothesis mph18( f18 );

        // y*z+x+y
        f19->addMultiParameterTerm( mult_y_z );
        f19->addMultiParameterTerm( add[ 0 ] );
        f19->addMultiParameterTerm( add[ 1 ] );
        MultiParameterHypothesis mph19( f19 );

        // y*z+x+z
        f20->addMultiParameterTerm( mult_y_z );
        f20->addMultiParameterTerm( add[ 0 ] );
        f20->addMultiParameterTerm( add[ 2 ] );
        MultiParameterHypothesis mph20( f20 );

        hypotheses.push_back( mph0 );
        hypotheses.push_back( mph1 );
        hypotheses.push_back( mph2 );
        hypotheses.push_back( mph3 );
        hypotheses.push_back( mph4 );
        hypotheses.push_back( mph5 );
        hypotheses.push_back( mph6 );
        hypotheses.push_back( mph7 );
        hypotheses.push_back( mph8 );
        hypotheses.push_back( mph9 );
        hypotheses.push_back( mph10 );
        hypotheses.push_back( mph11 );
        hypotheses.push_back( mph12 );
        hypotheses.push_back( mph13 );
        hypotheses.push_back( mph14 );
        hypotheses.push_back( mph15 );
        hypotheses.push_back( mph16 );
        hypotheses.push_back( mph17 );
        hypotheses.push_back( mph18 );
        hypotheses.push_back( mph19 );
        hypotheses.push_back( mph20 );
        hypotheses.push_back( mph21 );
        hypotheses.push_back( mph22 );
    }

    /**
    //add Hypotheses for 4 parameter models
    if ( params.size() == 4 )
    {
        EXTRAP::MultiParameterFunction* f9, * f10;
        f9 = new EXTRAP::MultiParameterFunction();
        f10 = new EXTRAP::MultiParameterFunction();
        f9->addMultiParameterTerm( mult );
        MultiParameterHypothesis mph7( f9 );
        f10->addMultiParameterTerm( add[ 0 ] );
        f10->addMultiParameterTerm( add[ 1 ] );
        f10->addMultiParameterTerm( add[ 2 ] );
        f10->addMultiParameterTerm( add[ 3 ] );
        MultiParameterHypothesis mph8( f10 );
        hypotheses.push_back( mph7 );   //a*b*c*d
        hypotheses.push_back( mph8 );   //a+b+c+d
    }

    //add Hypotheses for 5 parameter models
    if ( params.size() == 5 )
    {
        EXTRAP::MultiParameterFunction* f11, * f12;
        f11 = new EXTRAP::MultiParameterFunction();
        f12 = new EXTRAP::MultiParameterFunction();
        f11->addMultiParameterTerm( mult );
        MultiParameterHypothesis mph9( f11 );
        f12->addMultiParameterTerm( add[ 0 ] );
        f12->addMultiParameterTerm( add[ 1 ] );
        f12->addMultiParameterTerm( add[ 2 ] );
        f12->addMultiParameterTerm( add[ 3 ] );
        f12->addMultiParameterTerm( add[ 4 ] );
        MultiParameterHypothesis mph10( f12 );
        hypotheses.push_back( mph9 );   //a*b*c*d*e
        hypotheses.push_back( mph10 );   //a+b+c+d+e
    }
    **/
    
    MultiParameterHypothesis bestHypothesis = hypotheses[ 0 ];
    bestHypothesis.estimateParameters( modeledDataPointList );
    bestHypothesis.computeCost( modeledDataPointList );
    bestHypothesis.computeAdjustedRSquared( constantCost, modeledDataPointList );

    //
    // Uncomment the line below to get detailed printouts of all the hypotheses
    //
    std::cout << "hypothesis 0 : " << bestHypothesis.getFunction()->getAsString( params ) << " --- smape: " << bestHypothesis.getSMAPE() << " --- rss: " << bestHypothesis.getRE() << " --- ar2: " << bestHypothesis.getAR2() << std::endl;

    for ( int i = 1; i < hypotheses.size(); i++ )
    {
        hypotheses[ i ].estimateParameters( modeledDataPointList );
        hypotheses[ i ].computeCost( modeledDataPointList );
        hypotheses[ i ].computeAdjustedRSquared( constantCost, modeledDataPointList );

        //
        // Uncomment the line below to get detailed printouts of all the hypotheses
        //
        std::cout << "hypothesis " << i << " : " << hypotheses[ i ].getFunction()->getAsString( params ) << " --- smape: " << hypotheses[ i ].getSMAPE() << " --- rss: " << hypotheses[ i ].getRE() << " --- ar2: " << hypotheses[ i ].getAR2() << std::endl;

        if ( hypotheses[ i ].getSMAPE() < bestHypothesis.getSMAPE() )
        {
            delete ( bestHypothesis.getFunction() );
            bestHypothesis = hypotheses[ i ];
        }
        else
        {
            delete ( hypotheses[ i ].getFunction() );
        }

        //
        // Alternative hypotheses selection with relative error
        //
        //if ( hypotheses[ i ].getRE() < bestHypothesis.getRE() )
        //{
        //    delete ( bestHypothesis.getFunction() );
        //    bestHypothesis = hypotheses[ i ];
        //}
        //else
        //{
        //    delete ( hypotheses[ i ].getFunction() );
        //}
    }

    std::cout << "works until here..." << std::endl;

    std::cout << "besthypothesis : " << bestHypothesis.getFunction()->getAsString( params ) << " --- smape: " << bestHypothesis.getSMAPE() << " --- rss: " << bestHypothesis.getRE() << " --- ar2: " << bestHypothesis.getAR2() << std::endl;


    return bestHypothesis;
}
}; // Close namespace