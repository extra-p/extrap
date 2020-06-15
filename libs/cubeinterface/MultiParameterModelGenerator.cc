#include "MultiParameterModelGenerator.h"
#include "MultiParameterSimpleFunctionModeler.h"
#include "MultiParameterHypothesis.h"
#include "MultiParameterFunction.h"
#include "Utilities.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cassert>
#include "ModelGeneratorOptions.h"
#include "DataPoint.h"

namespace EXTRAP
{
MultiParameterModelGenerator::MultiParameterModelGenerator()
{
}

Model*
MultiParameterModelGenerator::createModel( const Experiment* experiment, const ModelGeneratorOptions& options, const ExperimentPointList& modeledPointList,
                                           const Function*            expectationFunction )
{
    FunctionPair* confidence_interval = new FunctionPair();
    FunctionPair* error_cone          = new FunctionPair();
    FunctionPair* noise_error         = new FunctionPair();

    confidence_interval->upper = new Function();
    confidence_interval->lower = new Function();
    error_cone->upper          = new Function();
    error_cone->lower          = new Function();
    noise_error->upper         = new Function();
    noise_error->lower         = new Function();

    this->m_options = options;

    //TODO: All of this should be done in the specific Modeler Classes, as the code is specific to each modeler type
    // Build list of data points from experiment points
    std::vector<DataPoint> dataPoints;
    dataPoints.reserve( modeledPointList.size() );
    for ( std::vector<ExperimentPoint*>::const_iterator it = modeledPointList.begin(); it != modeledPointList.end(); ++it )
    {
        ExperimentPoint* p = *it;
        if ( this->m_options.getGenerateModelOptions() == GENERATE_MODEL_MEAN )
        {
            dataPoints.push_back( DataPoint( &( p->getCoordinate() ), p->getSampleCount(), p->getMean(), p->getMeanCI() ) );
        }
        else if ( this->m_options.getGenerateModelOptions() == GENERATE_MODEL_MEDIAN )
        {
            dataPoints.push_back( DataPoint( &( p->getCoordinate() ), p->getSampleCount(), p->getMedian(), p->getMedianCI() ) );
        }
        else
        {
            assert( false );
        }
    }

    ModelCommentList comments = ModelCommentList();

    MultiParameterHypothesis bestHypothesis = getFunctionModeler().createModel( experiment, this->getModelGeneratorOptions(), dataPoints, comments, expectationFunction );

    // The resulting model will take ownership of the function
    return new Model( ( Function* )bestHypothesis.getFunction(),
                      this,
                      confidence_interval,
                      error_cone,
                      noise_error,
                      comments,
                      bestHypothesis.getRSS(),
                      bestHypothesis.getAR2(),
                      bestHypothesis.getSMAPE(),
                      bestHypothesis.getRE(),
                      bestHypothesis.getPredictedPoints(),
                      bestHypothesis.getActualPoints(),
                      bestHypothesis.getPs(),
                      bestHypothesis.getSizes() );
}

void
MultiParameterModelGenerator::setModelGeneratorOptions( ModelGeneratorOptions options )
{
    this->m_options = options;
}

ModelGeneratorOptions
MultiParameterModelGenerator::getModelGeneratorOptions( void ) const
{
    return this->m_options;
}

bool
MultiParameterModelGenerator::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  m_user_name ) );

    //get all the options
    ModelGeneratorOptions                options                = this->getModelGeneratorOptions();
    GenerateModelOptions                 generate_model_options = options.getGenerateModelOptions();
    int                                  min_number_points      = options.getMinNumberPoints();
    SparseModelerMultiParameterStrategy  multi_point_strategy   = options.getMultiPointsStrategy();
    int                                  number_add_points      = options.getNumberAddPoints();
    SparseModelerSingleParameterStrategy single_point_strategy  = options.getSinglePointsStrategy();
    bool                                 use_add_points         = options.getUseAddPoints();

    std::string generate_strategy = "";
    std::string single_strategy   = "";
    std::string multi_strategy    = "";

    //convert generate model options to string
    switch ( generate_model_options )
    {
        case GENERATE_MODEL_MEAN:
            generate_strategy = "GENERATE_MODEL_MEAN";
            break;

        case GENERATE_MODEL_MEDIAN:
            generate_strategy = "GENERATE_MODEL_MEDIAN";
            break;

        default:
            ErrorStream << "Invalid Options in the modeler found. Exiting..." << std::endl;
            return false;
    }

    //convert single parameter strategy to string
    switch ( single_point_strategy )
    {
        case FIRST_POINTS_FOUND:
            single_strategy = "FIRST_POINTS_FOUND";
            break;

        case MAX_NUMBER_POINTS:
            single_strategy = "MAX_NUMBER_POINTS";
            break;

        case MOST_EXPENSIVE_POINTS:
            single_strategy = "MOST_EXPENSIVE_POINTS";
            break;

        case CHEAPEST_POINTS:
            single_strategy = "CHEAPEST_POINTS";
            break;

        default:
            ErrorStream << "Invalid Options in the modeler found. Exiting..." << std::endl;
            return false;
    }

    //convert multi parameter strategy to string
    switch ( multi_point_strategy )
    {
        case INCREASING_COST:
            multi_strategy = "INCREASING_COST";
            break;

        case DECREASING_COST:
            multi_strategy = "DECREASING_COST";
            break;

        default:
            ErrorStream << "Invalid Options in the modeler found. Exiting..." << std::endl;
            return false;
    }

    //convert bool to int values
    int add_points = use_add_points;

    //write the options
    SAFE_RETURN( ioHelper->writeString(  generate_strategy ) );
    SAFE_RETURN( ioHelper->writeInt( min_number_points ) );
    SAFE_RETURN( ioHelper->writeString(  single_strategy ) );
    SAFE_RETURN( ioHelper->writeInt( add_points ) );
    SAFE_RETURN( ioHelper->writeInt( number_add_points ) );
    SAFE_RETURN( ioHelper->writeString(  multi_strategy ) );

    return true;
}

void
MultiParameterModelGenerator::deserialize( IoHelper* ioHelper )
{
    std::string userName = ioHelper->readString();
    this->setUserName( userName );

    //init variables for options
    std::string generate_strategy = "";
    std::string single_strategy   = "";
    std::string multi_strategy    = "";
    int         add_points        = 0;
    int         min_number_points = 0;
    int         number_add_points = 0;

    //read the options
    generate_strategy = ioHelper->readString();
    min_number_points = ioHelper->readInt();
    single_strategy   = ioHelper->readString();
    add_points        = ioHelper->readInt();
    number_add_points = ioHelper->readInt();
    multi_strategy    = ioHelper->readString();

    //convert ints to bool values
    bool use_add_points = add_points;

    //init enums
    GenerateModelOptions                 generate_model_options;
    SparseModelerSingleParameterStrategy single_point_strategy;
    SparseModelerMultiParameterStrategy  multi_point_strategy;

    //convert generate model options to enum
    if ( generate_strategy == "GENERATE_MODEL_MEAN" )
    {
        generate_model_options = GENERATE_MODEL_MEAN;
    }
    else if ( generate_strategy == "GENERATE_MODEL_MEDIAN" )
    {
        generate_model_options = GENERATE_MODEL_MEDIAN;
    }
    else
    {
        ErrorStream << "Invalid ModelOptions found in File." << std::endl;
    }

    //convert single parameter strategy to enum
    if ( single_strategy == "FIRST_POINTS_FOUND" )
    {
        single_point_strategy = FIRST_POINTS_FOUND;
    }
    else if ( single_strategy == "MAX_NUMBER_POINTS" )
    {
        single_point_strategy = MAX_NUMBER_POINTS;
    }
    else if ( single_strategy == "MOST_EXPENSIVE_POINTS" )
    {
        single_point_strategy = MOST_EXPENSIVE_POINTS;
    }
    else if ( single_strategy == "CHEAPEST_POINTS" )
    {
        single_point_strategy = CHEAPEST_POINTS;
    }
    else
    {
        ErrorStream << "Invalid ModelOptions found in File." << std::endl;
    }

    //convert multi parameter strategy to enum
    if ( multi_strategy == "INCREASING_COST" )
    {
        multi_point_strategy = INCREASING_COST;
    }
    else if ( multi_strategy == "DECREASING_COST" )
    {
        multi_point_strategy = DECREASING_COST;
    }
    else
    {
        ErrorStream << "Invalid ModelOptions found in File." << std::endl;
    }

    //set the model options
    this->m_options.setGenerateModelOptions( generate_model_options );
    this->m_options.setMinNumberPoints( min_number_points );
    this->m_options.setMultiPointsStrategy( multi_point_strategy );
    this->m_options.setNumberAddPoints( number_add_points );
    this->m_options.setSinglePointsStrategy( single_point_strategy );
    this->m_options.setUseAddPoints( use_add_points );
}
}; // Close namespace