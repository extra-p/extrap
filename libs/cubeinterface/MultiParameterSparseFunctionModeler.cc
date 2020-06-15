#include <stdio.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <cassert>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <map>
#include "MultiParameterSparseFunctionModeler.h"
#include "IncrementalPoint.h"
#include "MultiParameterHypothesis.h"
#include "Experiment.h"
#include "SingleParameterSimpleModelGenerator.h"
#include "Utilities.h"
#include "IoHelper.h"
#include "Coordinate.h"
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
MultiParameterHypothesis
MultiParameterSparseFunctionModeler::createModel( const Experiment* experiment, const ModelGeneratorOptions& options, const std::vector<DataPoint>& modeledDataPointList,
                                                  ModelCommentList&             comments,
                                                  const Function*               expectationFunction )
{

    //DEBUG
    //analyze the measurement points
    std::cout << "1.) Number of available Data Points: " << modeledDataPointList.size() << std::endl;
    ParameterList plist = experiment->getParameters();
    Parameter     p1    = plist.at( 0 );
    Parameter     p2    = plist.at( 1 );
    //Parameter     p3    = plist.at( 2 );
    //Parameter     p4    = plist.at( 3 );

    //options for the sparse modeler
    ModelGeneratorOptions m_options = options;

    //checkinng for auto select option
    bool auto_select = m_options.getUseAutoSelect();

    /*
    if(auto_select==true){

        //DEBUG
        std::cout << "using auto select option" << std::endl;

        //checking the number of min points for the single parameter experiments and the selection strategy
        int                                  min_measurement_points = m_options.getMinNumberPoints();

        //container for the single parameter coordinate lists
        std::vector<CoordinateList> coordinate_container_list;
        coordinate_container_list = findExpensiveMeasurementPoints( experiment, min_measurement_points, modeledDataPointList );
        
        std::vector<Experiment*>              experiments          = createAllExperiments( coordinate_container_list, experiment, modeledDataPointList );
        std::vector<Experiment*>              experiments2         = modelAllExperiments( experiments, m_options );
        std::vector<SingleParameterFunction*> functions            = getAllExperimentFunctions( experiments2 );
        CoordinateList                        base_cord_list       = getBaseCoordinates( coordinate_container_list );
        std::vector<DataPoint>                base_data_point_list = getBaseDataPoints( base_cord_list, experiment, modeledDataPointList );

        //Calculate the number of extra points that are available
        int                                 number_base_data_points      = base_cord_list.size();
        int                                 number_available_data_points = modeledDataPointList.size();
        int                                 max_add_data_points          = number_available_data_points - number_base_data_points;
        if(max_add_data_points > 0){
            m_options.setUseAddPoints(true);
            m_options.setNumberAddPoints(max_add_data_points);
        }
        if ( m_options.getUseAddPoints() == false )
        {
            //use only the base points from the single parameter experiments
            MultiParameterHypothesis bestHypothesis = findBestMultiParameterHypothesis( base_data_point_list, experiment, functions );
            return bestHypothesis;
        }
        else
        {
            //use additional data points for modeling   
            std::vector<DataPoint>   additional_data_point_list = getAdditionalDataPoints( base_cord_list, experiment, modeledDataPointList );
            std::vector<DataPoint>   sorted_data_points         = sortByCostInc( experiment, additional_data_point_list );
            std::vector<DataPoint>   data_points                = addAdditionalDataPoints( base_data_point_list, sorted_data_points, m_options.getNumberAddPoints() );    
            MultiParameterHypothesis bestHypothesis             = findBestMultiParameterHypothesis( data_points, experiment, functions );
            return bestHypothesis;
        }
    }
    */
    //else
    //{
    //checking the number of min points for the single parameter experiments and the selection strategy
    int                                  min_measurement_points = m_options.getMinNumberPoints();
    SparseModelerSingleParameterStrategy single_point_strategy  = m_options.getSinglePointsStrategy();

    //container for the single parameter coordinate lists
    std::vector<CoordinateList> coordinate_container_list;

    if ( single_point_strategy == FIRST_POINTS_FOUND )
    {
        coordinate_container_list = findFirstMeasurementPoints( experiment, min_measurement_points );
    }
    else if ( single_point_strategy == MAX_NUMBER_POINTS )
    {
        coordinate_container_list = findMaxMeasurementPoints( experiment, min_measurement_points );
    }
    else if ( single_point_strategy == MOST_EXPENSIVE_POINTS )
    {
        coordinate_container_list = findExpensiveMeasurementPoints( experiment, min_measurement_points, modeledDataPointList );
    }
    else if ( single_point_strategy == CHEAPEST_POINTS )
    {
        coordinate_container_list = findCheapMeasurementPoints( experiment, min_measurement_points, modeledDataPointList );
    }
    else
    {
        ErrorStream << "The selection strategy for the points you are trying to use does not exist. Using first points found strategy instead." << std::endl;
        coordinate_container_list = findFirstMeasurementPoints( experiment, min_measurement_points );
    }

    std::vector<Experiment*>              experiments          = createAllExperiments( coordinate_container_list, experiment, modeledDataPointList );
    std::vector<Experiment*>              experiments2         = modelAllExperiments( experiments, m_options );
    std::vector<SingleParameterFunction*> functions            = getAllExperimentFunctions( experiments2 );
    CoordinateList                        base_cord_list       = getBaseCoordinates( coordinate_container_list );
    std::vector<DataPoint>                base_data_point_list = getBaseDataPoints( base_cord_list, experiment, modeledDataPointList );

    //Calculate the number of extra points that are available
    int                                 number_base_data_points      = base_cord_list.size();
    int                                 number_available_data_points = modeledDataPointList.size();
    //DEBUG
    std::cout << "Number of available data points is:" << number_available_data_points << std::endl;
    int                                 max_add_data_points          = number_available_data_points - number_base_data_points;
    //DEBUG
    std::cout << "Number of additionally available data points is:" << max_add_data_points << std::endl;
    int                                 number_add_data_points       = m_options.getNumberAddPoints();
    bool                                use_add_data_points          = m_options.getUseAddPoints();
    //DEBUG
    std::cout << "Number of additionally used data points is:" << number_add_data_points << std::endl;
    SparseModelerMultiParameterStrategy multi_point_strategy         = m_options.getMultiPointsStrategy();

    //checking the number of data points to add additionally
    if ( number_add_data_points < 0 || number_add_data_points > max_add_data_points )
    {
        //ErrorStream << "The number of data points you choose to add is invalid. Maximum possible is " << max_add_data_points << ". Minimum required is 1. Using only baseline points instead." << std::endl;
        use_add_data_points = false;
    }

    if ( use_add_data_points == false )
    {
        //DEBUG
        std::cout << "2.) Using only base points for modeling!" << std::endl;
        std::cout << "3.) Using the following points for modeling:" << std::endl;
        for ( int i = 0; i < base_data_point_list.size(); i++ )
        {
            DataPoint dp  = base_data_point_list.at( i );
            double    v   = dp.getValue();
            double    p1v = dp.getParameterValue( p1 );
            double    p2v = dp.getParameterValue( p2 );  
            //double    p3v = dp.getParameterValue( p3 );
            //double    p4v = dp.getParameterValue( p4 );

            std::cout << "(" << p1.getName() << "," << p1v << ")(" << p2.getName() << "," << p2v << ") = " << v << std::endl;
            //std::cout << "(" << p1.getName() << "," << p1v << ")(" << p2.getName() << "," << p2v << ")(" << p3.getName() << "," << p3v << ") = " << v << std::endl;
            //std::cout << "(" << p1.getName() << "," << p1v << ")(" << p2.getName() << "," << p2v << ")(" << p3.getName() << "," << p3v << ")(" << p4.getName() << "," << p4v << ") = " << v << std::endl;
        }

        //use only the base points from the single parameter experiments
        MultiParameterHypothesis bestHypothesis = findBestMultiParameterHypothesis( base_data_point_list, experiment, functions );
        return bestHypothesis;
    }
    else
    {
        std::cout << "2.) Using additional points for modeling!" << std::endl;

        //use additional data points for modeling
        if ( multi_point_strategy == INCREASING_COST )
        {
            std::vector<DataPoint>   additional_data_point_list = getAdditionalDataPoints( base_cord_list, experiment, modeledDataPointList );
            std::vector<DataPoint>   sorted_data_points         = sortByCostInc( experiment, additional_data_point_list );
            std::vector<DataPoint>   data_points                = addAdditionalDataPoints( base_data_point_list, sorted_data_points, number_add_data_points );
            
            //DEBUG
            std::cout << "3.) Using the following points for modeling:" << std::endl;
            for ( int i = 0; i < data_points.size(); i++ )
            {
                DataPoint dp  = data_points.at( i );
                double    v   = dp.getValue();
                double    p1v = dp.getParameterValue( p1 );
                double    p2v = dp.getParameterValue( p2 );  
                //double    p3v = dp.getParameterValue( p3 );
                //double    p4v = dp.getParameterValue( p4 );

                std::cout << "(" << p1.getName() << "," << p1v << ")(" << p2.getName() << "," << p2v << ") = " << v << std::endl;
                //std::cout << "(" << p1.getName() << "," << p1v << ")(" << p2.getName() << "," << p2v << ")(" << p3.getName() << "," << p3v << ") = " << v << std::endl;
                //std::cout << "(" << p1.getName() << "," << p1v << ")(" << p2.getName() << "," << p2v << ")(" << p3.getName() << "," << p3v << ")(" << p4.getName() << "," << p4v << ") = " << v << std::endl;
            }
            
            MultiParameterHypothesis bestHypothesis             = findBestMultiParameterHypothesis( data_points, experiment, functions );
            return bestHypothesis;
        }
        else if ( multi_point_strategy == DECREASING_COST )
        {
            std::vector<DataPoint>   additional_data_point_list = getAdditionalDataPoints( base_cord_list, experiment, modeledDataPointList );
            std::vector<DataPoint>   sorted_data_points         = sortByCostDec( experiment, additional_data_point_list );
            std::vector<DataPoint>   data_points                = addAdditionalDataPoints( base_data_point_list, sorted_data_points, number_add_data_points );
            MultiParameterHypothesis bestHypothesis             = findBestMultiParameterHypothesis( data_points, experiment, functions );
            return bestHypothesis;
        }
        else
        {
            ErrorStream << "The selection strategy for the points you are trying to use does not exist. Using increasing cost strategy instead." << std::endl;
            std::vector<DataPoint>   additional_data_point_list = getAdditionalDataPoints( base_cord_list, experiment, modeledDataPointList );
            std::vector<DataPoint>   sorted_data_points         = sortByCostInc( experiment, additional_data_point_list );
            std::vector<DataPoint>   data_points                = addAdditionalDataPoints( base_data_point_list, sorted_data_points, number_add_data_points );
            MultiParameterHypothesis bestHypothesis             = findBestMultiParameterHypothesis( data_points, experiment, functions );
            return bestHypothesis;
        }
    }
    //}
}

std::vector<DataPoint>
MultiParameterSparseFunctionModeler::addAdditionalDataPoints( std::vector<DataPoint>& data_points, const std::vector<DataPoint>& modeledDataPointList, int number_add_data_points )
{
    for ( int i = 0; i < number_add_data_points; i++ )
    {
        DataPoint dp = modeledDataPointList.at( i );
        data_points.push_back( dp );
    }
    return data_points;
}

std::vector<DataPoint>
MultiParameterSparseFunctionModeler::sortByCostDec( const Experiment* experiment, const std::vector<DataPoint>& modeledDataPointList )
{
    std::vector<DataPoint> modeledDataPointList_copy = modeledDataPointList;
    std::vector<DataPoint> sorted_data_points;
    ParameterList          plist       = experiment->getParameters();
    Parameter              p1          = plist.at( 0 );

    for ( int i = 0; i < modeledDataPointList.size(); i++ )
    {
        double cost = 0;
        int    id   = 0;
        for ( int j = 0; j < modeledDataPointList_copy.size(); j++ )
        {
            DataPoint dp        = modeledDataPointList_copy.at( j );
            double    processes = dp.getParameterValue( p1 );
            double    time      = dp.getValue();
            if ( j == 0 )
            {
                cost = processes * time;
            }
            double cost_n = processes * time;
            if ( cost_n >= cost )
            {
                cost = cost_n;
                id   = j;
            }
        }
        DataPoint point_to_add = modeledDataPointList_copy.at( id );
        modeledDataPointList_copy.erase( modeledDataPointList_copy.begin() + id );
        sorted_data_points.push_back( point_to_add );
    }
    return sorted_data_points;
}

std::vector<DataPoint>
MultiParameterSparseFunctionModeler::sortByCostInc( const Experiment* experiment, const std::vector<DataPoint>& modeledDataPointList )
{
    std::vector<DataPoint> modeledDataPointList_copy = modeledDataPointList;
    std::vector<DataPoint> sorted_data_points;
    ParameterList          plist       = experiment->getParameters();
    Parameter              p1          = plist.at( 0 );

    for ( int i = 0; i < modeledDataPointList.size(); i++ )
    {
        double cost = 0;
        int    id   = 0;
        for ( int j = 0; j < modeledDataPointList_copy.size(); j++ )
        {
            DataPoint dp        = modeledDataPointList_copy.at( j );
            double    processes = dp.getParameterValue( p1 );
            double    time      = dp.getValue();
            if ( j == 0 )
            {
                cost = processes * time;
            }
            double cost_n = processes * time;
            if ( cost_n < cost )
            {
                cost = cost_n;
                id   = j;
            }
        }
        DataPoint point_to_add = modeledDataPointList_copy.at( id );
        modeledDataPointList_copy.erase( modeledDataPointList_copy.begin() + id );
        sorted_data_points.push_back( point_to_add );
    }
    return sorted_data_points;
}

std::vector<DataPoint>
MultiParameterSparseFunctionModeler::getAdditionalDataPoints( CoordinateList base_cord_list, const Experiment* experiment, const std::vector<DataPoint>& modeledDataPointList )
{
    std::vector<DataPoint> additional_data_point_list;
    ParameterList          plist = experiment->getParameters();

    for ( int i = 0; i < modeledDataPointList.size(); i++ )
    {
        DataPoint          dp  = modeledDataPointList.at( i );
        std::string cord_string = "";

        for (int j = 0; j < plist.size(); j++)
        {
            Parameter              parameter    = plist.at( j );
            double             parameter_value = dp.getParameterValue( parameter );
            std::ostringstream strs;
            strs << parameter_value;
            std::string        parameter_value_string = strs.str();
            cord_string = cord_string + "(" + parameter.getName() + "," + parameter_value_string + ")";
        }

        bool        is_existing = false;
        for ( int j = 0; j < base_cord_list.size(); j++ )
        {
            Coordinate* c        = base_cord_list.at( j );
            std::string c_string = c->toString();

            //bring the coordinate string in the right order
            //since someone thought it is a good idea to use a map that sorts alpabetically...
            std::vector<std::string> string_parts;
            std::string working_copy = c_string;

            for (int k = 0; k < plist.size(); k++)
            {
                Parameter parameter = plist.at( k );
                std::string parameter_string = parameter.getName();
                int pos = working_copy.find(parameter_string);
                std::string temp = working_copy.substr(pos-1,working_copy.size());
                pos = temp.find(")");
                temp = temp.substr(0,pos+1);
                string_parts.push_back(temp);
            }

            std::string sorted_coordinate_string = "";

            for (int k = 0; k < plist.size(); k++)
            {
                sorted_coordinate_string = sorted_coordinate_string + string_parts[k];
            }
            
            if ( cord_string == sorted_coordinate_string )
            {
                is_existing = true;
                break;
            }
        }
        if ( is_existing == false )
        {
            additional_data_point_list.push_back( dp );
        }
    }
    return additional_data_point_list;
}

std::vector<DataPoint>
MultiParameterSparseFunctionModeler::getBaseDataPoints( CoordinateList base_cord_list, const Experiment* experiment, const std::vector<DataPoint>& modeledDataPointList )
{
    std::vector<DataPoint> base_data_point_list;
    ParameterList          plist = experiment->getParameters();
    
    for ( int i = 0; i < modeledDataPointList.size(); i++ )
    {
        DataPoint          dp  = modeledDataPointList.at( i );
        std::string cord_string = "";

        for (int j = 0; j < plist.size(); j++)
        {
            Parameter              parameter    = plist.at( j );
            double             parameter_value = dp.getParameterValue( parameter );
            std::ostringstream strs;
            strs << parameter_value;
            std::string        parameter_value_string = strs.str();
            cord_string = cord_string + "(" + parameter.getName() + "," + parameter_value_string + ")";
        }
        
        bool        is_existing = false;
        for ( int j = 0; j < base_cord_list.size(); j++ )
        {
            Coordinate* c        = base_cord_list.at( j );
            std::string c_string = c->toString();

            //bring the coordinate string in the right order
            //since someone thought it is a good idea to use a map that sorts alpabetically...
            std::vector<std::string> string_parts;
            std::string working_copy = c_string;

            for (int k = 0; k < plist.size(); k++)
            {
                Parameter parameter = plist.at( k );
                std::string parameter_string = parameter.getName();
                int pos = working_copy.find(parameter_string);
                std::string temp = working_copy.substr(pos-1,working_copy.size());
                pos = temp.find(")");
                temp = temp.substr(0,pos+1);
                string_parts.push_back(temp);
            }

            std::string sorted_coordinate_string = "";

            for (int k = 0; k < plist.size(); k++)
            {
                sorted_coordinate_string = sorted_coordinate_string + string_parts[k];
            }

            if ( cord_string == sorted_coordinate_string )
            {
                is_existing = true;
                break;
            }
        }
        if ( is_existing == true )
        {
            base_data_point_list.push_back( dp );
        }
    }
    return base_data_point_list;
}

CoordinateList
MultiParameterSparseFunctionModeler::getBaseCoordinates( std::vector<CoordinateList> coordinate_container_list )
{
    CoordinateList base_cord_list;
    for ( int i = 0; i < coordinate_container_list.size(); i++ )
    {
        CoordinateList temp = coordinate_container_list.at( i );
        for ( int j = 0; j < temp.size(); j++ )
        {
            Coordinate* c = temp.at( j );
            if ( base_cord_list.size() == 0 )
            {
                base_cord_list.push_back( c );
            }
            else
            {
                std::string c_string    = c->toString();
                bool        is_existing = false;
                for ( int z = 0; z < base_cord_list.size(); z++ )
                {
                    Coordinate* c2        = base_cord_list.at( z );
                    std::string c2_string = c2->toString();
                    if ( c_string == c2_string )
                    {
                        is_existing = true;
                        break;
                    }
                }
                if ( is_existing == false )
                {
                    base_cord_list.push_back( c );
                }
            }
        }
    }
    return base_cord_list;
}

MultiParameterHypothesis
MultiParameterSparseFunctionModeler::findBestMultiParameterHypothesis( const std::vector<DataPoint>& modeledDataPointList, const Experiment* parent_experiment, std::vector<SingleParameterFunction*> functions )
{
    bool DEBUG = true;

    double constantCost = 0;
    double meanModel    = 0;

    for ( int i = 0; i < modeledDataPointList.size(); i++ )
    {
        meanModel += modeledDataPointList[ i ].getValue() / ( double )modeledDataPointList.size();
    }
    for ( int i = 0; i < modeledDataPointList.size(); i++ )
    {
        constantCost += ( modeledDataPointList[ i ].getValue() - meanModel ) * ( modeledDataPointList[ i ].getValue() - meanModel );
    }

    std::vector<EXTRAP::Parameter> plist = parent_experiment->getParameters();

    // get compound terms
    std::vector<CompoundTerm> compound_terms;
    std::vector<int> paramsToDelete;
    std::vector<int> paramsToKeep;
    ParameterList param_list;
    for ( int i = 0; i < functions.size(); i++ )
    {
        std::vector<CompoundTerm> ct_list = functions[ i ]->getCompoundTerms();
        if ( ct_list.size() > 0 )
        {
            CompoundTerm ct = ct_list[0];
            paramsToKeep.push_back( i );
            compound_terms.push_back( ct );
        }
        else
        {
            paramsToDelete.push_back( i );
        }
    }
    // in case function is constant
    if ( paramsToDelete.size() == plist.size() )
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
    // in case is simple function with one parameter
    else if ( ( plist.size() - paramsToDelete.size() ) == 1 )
    {
        MultiParameterFunction* simpleFunction = new MultiParameterFunction();
        MultiParameterTerm t;
        t.addCompoundTermParameterPair( compound_terms[0], plist[ paramsToKeep[ 0 ] ] );
        t.setCoefficient( compound_terms[0].getCoefficient() );
        simpleFunction->addMultiParameterTerm( t );
        MultiParameterHypothesis simpleHypothesis( simpleFunction );
        simpleHypothesis.getFunction()->setConstantCoefficient( functions[ paramsToKeep[ 0 ] ]->getConstantCoefficient() );
        simpleHypothesis.computeCost( modeledDataPointList );
        return simpleHypothesis;
    }
    //Remove unneccessary parameters
    for ( int i = paramsToDelete.size() - 1; i >= 0; i-- )
    {
        plist.erase( plist.begin() + paramsToDelete[ i ] );
    }

    // vector for hypotheses
    std::vector<MultiParameterHypothesis> hypotheses;

    //add Hypotheses for 2 parameter models
    if ( plist.size() == 2 )
    {
        //x*y
        EXTRAP::MultiParameterTerm            mult;
        for ( int i = 0; i < compound_terms.size(); i++ )
        {
            EXTRAP::CompoundTerm ct;
            ct = compound_terms[i];
            ct.setCoefficient( 1 );
            mult.addCompoundTermParameterPair( ct, plist[ i ] );
        }

        //x+y
        std::vector<EXTRAP::MultiParameterTerm> add;
        for ( int i = 0; i < compound_terms.size(); i++ )
        {
            EXTRAP::MultiParameterTerm mpt;
            EXTRAP::CompoundTerm ct = compound_terms[i];
            ct.setCoefficient( 1 );
            mpt.addCompoundTermParameterPair( ct, plist[ i ] );
            mpt.setCoefficient( 1 );
            add.push_back( mpt );
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
    if ( plist.size() == 3 )
    {
        //x*y*z
        EXTRAP::MultiParameterTerm            mult;
        for ( int i = 0; i < compound_terms.size(); i++ )
        {
            EXTRAP::CompoundTerm ct;
            ct = compound_terms[i];
            ct.setCoefficient( 1 );
            mult.addCompoundTermParameterPair( ct, plist[ i ] );
        }

        //x*y
        EXTRAP::MultiParameterTerm mult_x_y;
        EXTRAP::CompoundTerm ct0 = compound_terms[0];
        ct0.setCoefficient( 1 );
        mult_x_y.addCompoundTermParameterPair( ct0, plist[ 0 ] );
        EXTRAP::CompoundTerm ct1 = compound_terms[1];
        ct1.setCoefficient( 1 );
        mult_x_y.addCompoundTermParameterPair( ct1, plist[ 1 ] );

        //y*z
        EXTRAP::MultiParameterTerm mult_y_z;
        EXTRAP::CompoundTerm ct2 = compound_terms[1];
        ct2.setCoefficient( 1 );
        mult_y_z.addCompoundTermParameterPair( ct2, plist[ 1 ] );
        EXTRAP::CompoundTerm ct3 = compound_terms[2];
        ct3.setCoefficient( 1 );
        mult_y_z.addCompoundTermParameterPair( ct3, plist[ 2 ] );

        //x*z
        EXTRAP::MultiParameterTerm mult_x_z;
        EXTRAP::CompoundTerm ct4 = compound_terms[0];
        ct4.setCoefficient( 1 );
        mult_x_z.addCompoundTermParameterPair( ct4, plist[ 0 ] );
        EXTRAP::CompoundTerm ct5 = compound_terms[2];
        ct5.setCoefficient( 1 );
        mult_x_z.addCompoundTermParameterPair( ct5, plist[ 2 ] );

        //x+y+z
        std::vector<EXTRAP::MultiParameterTerm> add;
        for ( int i = 0; i < compound_terms.size(); i++ )
        {
            EXTRAP::MultiParameterTerm mpt;
            EXTRAP::CompoundTerm ct = compound_terms[i];
            ct.setCoefficient( 1 );
            mpt.addCompoundTermParameterPair( ct, plist[ i ] );
            mpt.setCoefficient( 1 );
            add.push_back( mpt );
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

    //select one function as the bestHypothesis for the start
    MultiParameterHypothesis bestHypothesis = hypotheses[ 0 ];
    bestHypothesis.estimateParameters( modeledDataPointList );
    bestHypothesis.computeCost( modeledDataPointList );
    bestHypothesis.computeAdjustedRSquared( constantCost, modeledDataPointList );

    //debug print out for the hypothesis
    if ( DEBUG == true )
    {
        std::cout << "hypothesis 0 : " << bestHypothesis.getFunction()->getAsString( plist ) << " --- smape: " << bestHypothesis.getSMAPE() << " --- ar2: " << bestHypothesis.getAR2() << " --- rss: " << bestHypothesis.getRSS() << " --- rrss: " << bestHypothesis.getrRSS() << " --- re: " << bestHypothesis.getRE() << std::endl;
    }

    //find the best hypothesis
    for ( int i = 1; i < hypotheses.size(); i++ )
    {
        hypotheses[ i ].estimateParameters( modeledDataPointList );
        hypotheses[ i ].computeCost( modeledDataPointList );
        hypotheses[ i ].computeAdjustedRSquared( constantCost, modeledDataPointList );

        //debug print out for the hypothesis
        if ( DEBUG == true )
        {
            std::cout << "hypothesis " << i << " : " << hypotheses[ i ].getFunction()->getAsString( plist ) << " --- smape: " << hypotheses[ i ].getSMAPE() << " --- ar2: " << hypotheses[ i ].getAR2() << " --- rss: " << hypotheses[ i ].getRSS() << " --- rrss: " << hypotheses[ i ].getrRSS() << " --- re: " << hypotheses[ i ].getRE() << std::endl;
        }

        if ( hypotheses[ i ].getSMAPE() < bestHypothesis.getSMAPE() )
        {
            delete ( bestHypothesis.getFunction() );
            bestHypothesis = hypotheses[ i ];
        }
        else
        {
            delete ( hypotheses[ i ].getFunction() );
        }
        /**
         * alternative selection with relative error
           if ( hypotheses[ i ].getRE() < bestHypothesis.getRE() )
           {
            delete ( bestHypothesis.getFunction() );
            bestHypothesis = hypotheses[ i ];
           }
           else
           {
            delete ( hypotheses[ i ].getFunction() );
           }
         **/
    }
    return bestHypothesis;
}

std::vector<SingleParameterFunction*>
MultiParameterSparseFunctionModeler::getAllExperimentFunctions( std::vector<Experiment*> experiments )
{
    std::vector<SingleParameterFunction*> functions;
    for ( int i = 0; i < experiments.size(); i++ )
    {
        Experiment*              experiment     = experiments.at( i );
        const ParameterList&     parameterNames = experiment->getParameters();
        Callpath*                callpath       = experiment->getCallpath( 0 );
        Metric*                  metric         = experiment->getMetric( 0 );
        const ModelList&         model_list     = experiment->getModels( *metric, *callpath );
        SingleParameterFunction* function       = dynamic_cast<SingleParameterFunction*>( model_list[ 0 ]->getModelFunction() );
        functions.push_back( function );
    }
    return functions;
}

std::vector<Experiment*>
MultiParameterSparseFunctionModeler::modelAllExperiments( std::vector<Experiment*> experiments, ModelGeneratorOptions options )
{
    std::vector<Experiment*> exps;
    for ( int i = 0; i < experiments.size(); i++ )
    {
        Experiment* exp = experiments.at( i );
        exp = addSingleParameterModeler( exp, options );
        exps.push_back( exp );
    }
    return exps;
}

Experiment*
MultiParameterSparseFunctionModeler::addSingleParameterModeler( Experiment* exp, ModelGeneratorOptions options )
{
    Experiment*                          experiment       = exp;
    MetricList                           metrics          = experiment->getMetrics();
    CallpathList                         callpaths        = experiment->getAllCallpaths();
    ModelGenerator*                      model_generator  = NULL;
    SingleParameterSimpleModelGenerator* single_generator = new SingleParameterSimpleModelGenerator();
    model_generator = single_generator;
    // check if the parameter value is smaller than 1, then do not allow log terms...
    CoordinateList cordlist = experiment->getCoordinates();
    bool allow_log = true;
    for ( int i = 0; i < cordlist.size(); i++ )
    {
        std::string tmp = cordlist.at( i )->toString();
        int pos = tmp.find(",");
        std::string value = tmp.substr(pos+1,tmp.size());
        value = value.substr(0,value.size()-1);
        double v = 0;
        std::istringstream iss( value );
        iss >> v;
        if(v<1){
            allow_log = false;
            break;
        }
    }
    // tell the modeler if he should use log terms for modeling
    options.setUseLogTerms(allow_log);
    experiment->addModelGenerator( model_generator );
    experiment->modelAll( *model_generator, experiment, options );
    return experiment;
}

std::vector<Experiment*>
MultiParameterSparseFunctionModeler::createAllExperiments( std::vector<CoordinateList> coordinate_container_list, const Experiment* parent_experiment, const std::vector<DataPoint>& modeledDataPointList )
{
    std::vector<Experiment*> experiments;
    if ( coordinate_container_list.size() > 0 )
    {
        CoordinateList coordinate_container;
        Parameter      parameter;
        Metric*        metric = NULL;
        Region*        region = NULL;
        Callpath*      callpath;
        Experiment*    single_param_exp;
        for ( int i = 0; i < coordinate_container_list.size(); i++ )
        {
            coordinate_container = coordinate_container_list.at( i );
            parameter            = parent_experiment->getParameter( i );
            metric               = parent_experiment->getMetric( 0 );
            region               = parent_experiment->getRegion( 0 );
            callpath             = parent_experiment->getCallpath( 0 );
            single_param_exp     = createSingleParameterExperiment( parent_experiment, coordinate_container, parameter, metric, region, callpath, modeledDataPointList );
            experiments.push_back( single_param_exp );
        }
    }
    else
    {
        ErrorStream << "List of measurement points is empty. Can not create experiments for parameters." << std::endl;
    }
    return experiments;
}

Experiment*
MultiParameterSparseFunctionModeler::createSingleParameterExperiment( const Experiment* original_experiment, CoordinateList coordinate_container, Parameter parameter, Metric* metric, Region* region, Callpath* callpath, const std::vector<DataPoint>& modeledDataPointList )
{
    bool DEBUG = false;

    //add experiment objects
    Experiment* single_param_exp = new Experiment();
    single_param_exp->addParameter( parameter );
    single_param_exp->addMetric( metric );
    single_param_exp->addRegion( region );
    single_param_exp->addCallpath( callpath );
    if ( DEBUG == true )
    {
        std::cout << "Measurement Points for Single Parameter Experiment: " << parameter.getName() << "\n";
    }

    /**
     * add coordinates to the experiment
     * must be casted from n parameter cords to 1 parameter cords
     * keep a copy of them for later to add the data values
     **/
    std::vector<std::string> save_cords;
    for ( int i = 0; i < coordinate_container.size(); i++ )
    {
        Coordinate* cord        = coordinate_container.at( i );
        std::string cord_string = cord->toString();
        save_cords.push_back( cord_string );
        int pos = 0;
        pos = cord_string.find( parameter.getName() );
        std::string parameter_value = "";
        parameter_value = cord_string.substr( pos, cord_string.size() );
        pos             = parameter_value.find( ")" );
        parameter_value = parameter_value.substr( 0, pos );
        pos             = parameter_value.find( "," );
        parameter_value = parameter_value.substr( pos + 1, parameter_value.size() );
        double x;
        std::istringstream iss( parameter_value );
        iss >> x;
        Coordinate* c = new Coordinate();
        c->insert( std::pair<Parameter, Value>( parameter, x ) );
        single_param_exp->addCoordinate( c );
        if ( DEBUG == true )
        {
            std::cout << c->toString() << "\n";
        }
    }
    if ( DEBUG == true )
    {
        std::cout << "\n";
    }

    CoordinateList pointslist = single_param_exp->getCoordinates();
    // array that keeps all the data points, separated by the measurement point id of the point list which stores the measurement points
    IncrementalPoint data_points[ pointslist.size() ];

    //do the same thing, but with the modelDataPoints and not with line passing from the file...

    for ( int j = 0; j < save_cords.size(); j++ )
    {
        std::string coordinate_string = save_cords.at( j );

        //bring the coordinate string in the right order
        //since someone thought it is a good idea to use a map that sorts alpabetically...
        std::vector<std::string> string_parts;
        ParameterList p_list =  original_experiment->getParameters();
        std::string working_copy = coordinate_string;

        for (int k = 0; k < p_list.size(); k++)
        {
            Parameter parameter = p_list.at( k );
            std::string parameter_string = parameter.getName();
            int pos = working_copy.find(parameter_string);
            std::string temp = working_copy.substr(pos-1,working_copy.size());
            pos = temp.find(")");
            temp = temp.substr(0,pos+1);
            string_parts.push_back(temp);
        }

        std::string sorted_coordinate_string = "";

        for (int k = 0; k < p_list.size(); k++)
        {
            sorted_coordinate_string = sorted_coordinate_string + string_parts[k];
        }

        for ( int i = 0; i < modeledDataPointList.size(); i++ )
        {
            DataPoint data_point = modeledDataPointList.at( i );
            ParameterList param_list =  original_experiment->getParameters();
            double    x   = data_point.getValue();
            std::string p = "";

            for (int k = 0; k < param_list.size(); k++)
            {
                Parameter parameter  = param_list.at( k );
                double parameter_value = data_point.getParameterValue( parameter );
                std::ostringstream strs1;
                strs1 << parameter_value;
                std::string parameter_value_string = strs1.str();
                p = p + "(" + parameter.getName() + "," + parameter_value_string + ")";
            }

            if ( sorted_coordinate_string == p )
            {
                data_points[ j ].addValue( x );
            }
        }
    }

    // add all the collected measurements in form of the data points to the experiment
    for ( int i = 0; i < pointslist.size(); i++ )
    {
        single_param_exp->addDataPoint( data_points[ i ].getExperimentPoint( pointslist[ i ],
                                                                             metric,
                                                                             callpath ),
                                        *metric,
                                        *callpath );
    }

    return single_param_exp;
}

std::vector<CoordinateList>
MultiParameterSparseFunctionModeler::findFirstMeasurementPoints( const Experiment* parent_experiment, int min_measurement_points )
{
    bool DEBUG = false;

    std::vector<CoordinateList> coordinate_container_list;

    // check the number of parameters
    ParameterList parameters = parent_experiment->getParameters();

    if ( DEBUG == true )
    {
        std::cout << "Number of parameters: " << parameters.size() << ".\n";
        for ( int i = 0; i < parameters.size(); i++ )
        {
            std::cout << "parameter " << i << ": " << parameters.at( i ).getName() << std::endl;
        }
    }

    if ( parameters.size() == 1 )
    {
        ErrorStream << "Experiment contains not enough parameters. Minimum for this modeling approach is 2." << std::endl;
    }

    else if ( parameters.size() == 2 || parameters.size() == 3 || parameters.size() == 4 || parameters.size() == 5 )
    {
        CoordinateList coordinates_tmp = parent_experiment->getCoordinates();
        CoordinateList coordinates;

        //analyze the coordinate list, to filter duplicates in case there are some
        for ( int i = 0; i < coordinates_tmp.size(); i++ )
        {
            if ( i == 0 )
            {
                coordinates.push_back( coordinates_tmp.at( i ) );
            }
            else
            {
                bool found = false;
                for ( int j = 0; j < coordinates.size(); j++ )
                {
                    if ( coordinates.at( j )->toString() == coordinates_tmp.at( i )->toString() )
                    {
                        found = true;
                        break;
                    }
                }
                if ( found == false )
                {
                    coordinates.push_back( coordinates_tmp.at( i ) );
                }
            }
        }

        // debug print for the coordinates input
        if ( DEBUG == true )
        {
            std::cout << "Number of Coordinates: " << coordinates.size() << ".\n";
            for ( int i = 0; i < coordinates.size(); i++ )
            {
                std::cout << "Coordinate " << i << ": " << coordinates.at( i )->toString() << std::endl;
            }
        }

        for ( int i = 0; i < parameters.size(); i++ )
        {
            Parameter      parameter = parameters[ i ];
            CoordinateList coordinate_container;
            bool           done = false;
            for ( int j = 0; j < coordinates.size(); j++ )
            {
                Coordinate*         reference_coordinate = coordinates[ j ];
                std::vector<double> parameter_value_list = getParameterValues( parameters, reference_coordinate, i );

                for ( int z = 0; z < coordinates.size(); z++ )
                {
                    Coordinate*         coordinate            = coordinates[ z ];
                    std::vector<double> parameter_value_list2 = getParameterValues( parameters, coordinate, i );
                    bool                equal                 = compareParameterValues( parameter_value_list, parameter_value_list2 );
                    if ( equal == true )
                    {
                        coordinate_container.push_back( coordinate );
                    }
                }
                if ( coordinate_container.size() < min_measurement_points )
                {
                    coordinate_container.clear();
                }
                else
                {
                    done = true;
                    break;
                }
            }
            if ( done == true )
            {
                coordinate_container_list.push_back( coordinate_container );
            }
            else
            {
                ErrorStream << "Not enough measurement points for parameter " << parameter.getName() << "." << std::endl;
            }
        }

        // print the result of the search
        if ( DEBUG == true )
        {
            for ( int i = 0; i < coordinate_container_list.size(); i++ )
            {
                Parameter parameter = parameters[ i ];
                std::cout << "Measurement Points for parameter: " << parameter.getName() << ".\n";
                CoordinateList cord_list = coordinate_container_list.at( i );
                for ( int j = 0; j < cord_list.size(); j++ )
                {
                    Coordinate* reference_coordinate = cord_list[ j ];
                    std::string temp                 = reference_coordinate->toString();
                    std::cout << temp << "\n";
                }
                std::cout << "\n";
            }
        }
    }
    else
    {
        ErrorStream << "Experiment contains a number of parameters that is currently not supported." << std::endl;
    }
    return coordinate_container_list;
}

std::vector<CoordinateList>
MultiParameterSparseFunctionModeler::findMaxMeasurementPoints( const Experiment* parent_experiment, int min_measurement_points )
{
    bool DEBUG = false;

    //the output list, will contain the final selected coordinates
    std::vector<CoordinateList> coordinate_container_list;

    //check the number of parameters
    ParameterList parameters = parent_experiment->getParameters();

    //create empty lists for the coordinate selection process
    std::vector<std::vector<CoordinateList> > container;
    for ( int i = 0; i < parameters.size(); i++ )
    {
        std::vector<CoordinateList> tmp_container_list;
        container.push_back( tmp_container_list );
    }

    if ( DEBUG == true )
    {
        std::cout << "Number of parameters: " << parameters.size() << ".\n";
        for ( int i = 0; i < parameters.size(); i++ )
        {
            std::cout << "parameter " << i << ": " << parameters.at( i ).getName() << std::endl;
        }
    }

    if ( parameters.size() == 1 )
    {
        ErrorStream << "Experiment contains not enough parameters. Minimum for this modeling approach is 2." << std::endl;
    }

    //Sparse Modeler only support 2 parameters
    else if ( parameters.size() == 2 || parameters.size() == 3 || parameters.size() == 4 || parameters.size() == 5 )
    {
        CoordinateList coordinates_tmp = parent_experiment->getCoordinates();
        CoordinateList coordinates;

        //analyze the coordinate list, to filter duplicates in case there are some
        for ( int i = 0; i < coordinates_tmp.size(); i++ )
        {
            if ( i == 0 )
            {
                coordinates.push_back( coordinates_tmp.at( i ) );
            }
            else
            {
                bool found = false;
                for ( int j = 0; j < coordinates.size(); j++ )
                {
                    if ( coordinates.at( j )->toString() == coordinates_tmp.at( i )->toString() )
                    {
                        found = true;
                        break;
                    }
                }
                if ( found == false )
                {
                    coordinates.push_back( coordinates_tmp.at( i ) );
                }
            }
        }

        // debug print for the coordinates input
        if ( DEBUG == true )
        {
            std::cout << "Number of Coordinates: " << coordinates.size() << ".\n";
            for ( int i = 0; i < coordinates.size(); i++ )
            {
                std::cout << "Coordinate " << i << ": " << coordinates.at( i )->toString() << std::endl;
            }
        }

        //find point rows that have at least min points
        for ( int i = 0; i < parameters.size(); i++ )
        {
            Parameter      parameter = parameters[ i ];
            CoordinateList coordinate_container;
            for ( int j = 0; j < coordinates.size(); j++ )
            {
                Coordinate*         reference_coordinate = coordinates[ j ];
                std::vector<double> parameter_value_list = getParameterValues( parameters, reference_coordinate, i );
                for ( int z = 0; z < coordinates.size(); z++ )
                {
                    Coordinate*         coordinate            = coordinates[ z ];
                    std::vector<double> parameter_value_list2 = getParameterValues( parameters, coordinate, i );
                    bool                equal                 = compareParameterValues( parameter_value_list, parameter_value_list2 );
                    if ( equal == true )
                    {
                        coordinate_container.push_back( coordinate );
                    }
                }
                if ( coordinate_container.size() < min_measurement_points )
                {
                    coordinate_container.clear();
                }
                else
                {
                    //save the point row that was found
                    container.at( i ).push_back( coordinate_container );
                    coordinate_container.clear();
                }
            }
        }

        //remove the duplicates in the list
        std::vector<std::vector<CoordinateList> > copy_container;
        for ( int i = 0; i < container.size(); i++ )
        {
            std::vector<CoordinateList> tmp_container_list = container.at( i );
            std::vector<CoordinateList> copy_container_list;
            for ( int j = 0; j < tmp_container_list.size(); j++ )
            {
                if ( copy_container_list.size() == 0 )
                {
                    copy_container_list.push_back( tmp_container_list.at( j ) );
                }
                else
                {
                    bool           in_list   = false;
                    CoordinateList cord_list = tmp_container_list.at( j );
                    for ( int k = 0; k < copy_container_list.size(); k++ )
                    {
                        bool           are_equal  = true;
                        CoordinateList cord_list2 = copy_container_list.at( k );
                        for ( int l = 0; l < cord_list2.size(); l++ )
                        {
                            Coordinate* c1 = cord_list.at( l );
                            Coordinate* c2 = cord_list2.at( l );
                            if ( c1->toString() != c2->toString() )
                            {
                                are_equal = false;
                                break;
                            }
                        }
                        if ( are_equal == true )
                        {
                            in_list = true;
                            break;
                        }
                    }
                    if ( in_list == false )
                    {
                        copy_container_list.push_back( tmp_container_list.at( j ) );
                    }
                }
            }
            copy_container.push_back( copy_container_list );
        }

        //find the longest row for each parameter
        for ( int i = 0; i < copy_container.size(); i++ )
        {
            int                         id                 = 0;
            int                         list_size          = 0;
            std::vector<CoordinateList> tmp_container_list = copy_container.at( i );
            for ( int j = 0; j < tmp_container_list.size(); j++ )
            {
                CoordinateList tmp_list = tmp_container_list.at( j );
                if ( tmp_list.size() >= list_size )
                {
                    list_size = tmp_list.size();
                    id        = j;
                }
            }
            coordinate_container_list.push_back( tmp_container_list.at( id ) );
        }

        // print the result of the search
        if ( DEBUG == true )
        {
            for ( int i = 0; i < coordinate_container_list.size(); i++ )
            {
                Parameter parameter = parameters[ i ];
                std::cout << "Measurement Points for parameter: " << parameter.getName() << ".\n";
                CoordinateList cord_list = coordinate_container_list.at( i );
                for ( int j = 0; j < cord_list.size(); j++ )
                {
                    Coordinate* reference_coordinate = cord_list[ j ];
                    std::string temp                 = reference_coordinate->toString();
                    std::cout << temp << "\n";
                }
                std::cout << "\n";
            }
        }
    }
    else
    {
        ErrorStream << "Experiment contains a number of parameters that is currently not supported." << std::endl;
    }
    return coordinate_container_list;
}

std::vector<CoordinateList>
MultiParameterSparseFunctionModeler::findExpensiveMeasurementPoints( const Experiment* parent_experiment, int min_measurement_points, const std::vector<DataPoint>& modeledDataPointList )
{
    bool DEBUG = false;

    //working copy of the data points list
    std::vector<DataPoint> copy_modeledDataPointList = modeledDataPointList;

    //the output list, will contain the final selected coordinates
    std::vector<CoordinateList> coordinate_container_list;

    //check the number of parameters
    ParameterList parameters = parent_experiment->getParameters();

    //create empty lists for the coordinate selection process
    std::vector<std::vector<CoordinateList> > container;
    for ( int i = 0; i < parameters.size(); i++ )
    {
        std::vector<CoordinateList> tmp_container_list;
        container.push_back( tmp_container_list );
    }

    if ( DEBUG == true )
    {
        std::cout << "Number of parameters: " << parameters.size() << ".\n";
        for ( int i = 0; i < parameters.size(); i++ )
        {
            std::cout << "parameter " << i << ": " << parameters.at( i ).getName() << std::endl;
        }
    }

    if ( parameters.size() == 1 )
    {
        ErrorStream << "Experiment contains not enough parameters. Minimum for this modeling approach is 2." << std::endl;
    }

    else if ( parameters.size() == 2 || parameters.size() == 3 || parameters.size() == 4 || parameters.size() == 5 )
    {
        CoordinateList coordinates_tmp = parent_experiment->getCoordinates();
        CoordinateList coordinates;

        //analyze the coordinate list, to filter duplicates in case there are some
        for ( int i = 0; i < coordinates_tmp.size(); i++ )
        {
            if ( i == 0 )
            {
                coordinates.push_back( coordinates_tmp.at( i ) );
            }
            else
            {
                bool found = false;
                for ( int j = 0; j < coordinates.size(); j++ )
                {
                    if ( coordinates.at( j )->toString() == coordinates_tmp.at( i )->toString() )
                    {
                        found = true;
                        break;
                    }
                }
                if ( found == false )
                {
                    coordinates.push_back( coordinates_tmp.at( i ) );
                }
            }
        }

        // debug print for the coordinates input
        if ( DEBUG == true )
        {
            std::cout << "Number of Coordinates: " << coordinates.size() << ".\n";
            for ( int i = 0; i < coordinates.size(); i++ )
            {
                std::cout << "Coordinate " << i << ": " << coordinates.at( i )->toString() << std::endl;
            }
        }

        //find point rows that have at least min points
        for ( int i = 0; i < parameters.size(); i++ )
        {
            Parameter      parameter = parameters[ i ];
            CoordinateList coordinate_container;
            for ( int j = 0; j < coordinates.size(); j++ )
            {
                Coordinate*         reference_coordinate = coordinates[ j ];
                std::vector<double> parameter_value_list = getParameterValues( parameters, reference_coordinate, i );
                for ( int z = 0; z < coordinates.size(); z++ )
                {
                    Coordinate*         coordinate            = coordinates[ z ];
                    std::vector<double> parameter_value_list2 = getParameterValues( parameters, coordinate, i );
                    bool                equal                 = compareParameterValues( parameter_value_list, parameter_value_list2 );
                    if ( equal == true )
                    {
                        coordinate_container.push_back( coordinate );
                    }
                }
                if ( coordinate_container.size() < min_measurement_points )
                {
                    coordinate_container.clear();
                }
                else
                {
                    //save the point row that was found
                    container.at( i ).push_back( coordinate_container );
                    coordinate_container.clear();
                }
            }
        }

        //remove the duplicates in the list
        std::vector<std::vector<CoordinateList> > copy_container;
        for ( int i = 0; i < container.size(); i++ )
        {
            std::vector<CoordinateList> tmp_container_list = container.at( i );
            std::vector<CoordinateList> copy_container_list;
            for ( int j = 0; j < tmp_container_list.size(); j++ )
            {
                if ( copy_container_list.size() == 0 )
                {
                    copy_container_list.push_back( tmp_container_list.at( j ) );
                }
                else
                {
                    bool           in_list   = false;
                    CoordinateList cord_list = tmp_container_list.at( j );
                    for ( int k = 0; k < copy_container_list.size(); k++ )
                    {
                        bool           are_equal  = true;
                        CoordinateList cord_list2 = copy_container_list.at( k );
                        for ( int l = 0; l < cord_list2.size(); l++ )
                        {
                            Coordinate* c1 = cord_list.at( l );
                            Coordinate* c2 = cord_list2.at( l );
                            if ( c1->toString() != c2->toString() )
                            {
                                are_equal = false;
                                break;
                            }
                        }
                        if ( are_equal == true )
                        {
                            in_list = true;
                            break;
                        }
                    }
                    if ( in_list == false )
                    {
                        copy_container_list.push_back( tmp_container_list.at( j ) );
                    }
                }
            }
            copy_container.push_back( copy_container_list );
        }

        //find the most expensive row for each parameter
        for ( int i = 0; i < copy_container.size(); i++ )
        {
            int                         id                 = 0;
            std::vector<CoordinateList> tmp_container_list = copy_container.at( i );
            double                      max_sum_cost       = 0;
            for ( int j = 0; j < tmp_container_list.size(); j++ )
            {
                double         sum_cost = 0;
                CoordinateList tmp_list = tmp_container_list.at( j );
                for ( int k = 0; k < tmp_list.size(); k++ )
                {
                    Coordinate* c        = tmp_list.at( k );
                    std::string c_string = c->toString();
                    //find the data point for this coordinate
                    double time = 0;
                    double p    = 0;
                    for ( int l = 0; l < copy_modeledDataPointList.size(); l++ )
                    {
                        DataPoint          dp  = copy_modeledDataPointList.at( l );
                        double             v   = dp.getValue();
                        std::string cord_string = "";

                        Parameter parameter0 = parameters.at( 0 );
                        double processes = dp.getParameterValue( parameter0 );

                        for (int o = 0; o < parameters.size(); o++)
                        {
                            Parameter parameter = parameters.at( o );
                            double parameter_value = dp.getParameterValue( parameter );
                            std::ostringstream strs;
                            strs << parameter_value;
                            std::string parameter_value_string = strs.str();
                            cord_string = cord_string + "(" + parameter.getName() + "," + parameter_value_string + ")";
                        }
                        
                        //works only when: cost = p * time
                        if ( c_string == cord_string )
                        {
                            p    = processes;
                            time = v;
                            break;
                        }
                    }
                    //calculate cost for this coordinate
                    double cost = p * time;
                    sum_cost = sum_cost + cost;
                }

                //check if the current sum_cost is higher than the previous one
                if ( sum_cost >= max_sum_cost )
                {
                    max_sum_cost = sum_cost;
                    id           = j;
                }
            }
            //add the best solution found
            coordinate_container_list.push_back( tmp_container_list.at( id ) );
        }

        // print the result of the search
        if ( DEBUG == true )
        {
            for ( int i = 0; i < coordinate_container_list.size(); i++ )
            {
                Parameter parameter = parameters[ i ];
                std::cout << "Measurement Points for parameter: " << parameter.getName() << ".\n";
                CoordinateList cord_list = coordinate_container_list.at( i );
                for ( int j = 0; j < cord_list.size(); j++ )
                {
                    Coordinate* reference_coordinate = cord_list[ j ];
                    std::string temp                 = reference_coordinate->toString();
                    std::cout << temp << "\n";
                }
                std::cout << "\n";
            }
        }
    }
    else
    {
        ErrorStream << "Experiment contains a number of parameters that is currently not supported." << std::endl;
    }
    return coordinate_container_list;
}

std::vector<CoordinateList>
MultiParameterSparseFunctionModeler::findCheapMeasurementPoints( const Experiment* parent_experiment, int min_measurement_points, const std::vector<DataPoint>& modeledDataPointList )
{
    bool DEBUG = false;

    //working copy of the data points list
    std::vector<DataPoint> copy_modeledDataPointList = modeledDataPointList;

    //the output list, will contain the final selected coordinates
    std::vector<CoordinateList> coordinate_container_list;

    //check the number of parameters
    ParameterList parameters = parent_experiment->getParameters();

    //create empty lists for the coordinate selection process
    std::vector<std::vector<CoordinateList> > container;
    for ( int i = 0; i < parameters.size(); i++ )
    {
        std::vector<CoordinateList> tmp_container_list;
        container.push_back( tmp_container_list );
    }

    if ( DEBUG == true )
    {
        std::cout << "Number of parameters: " << parameters.size() << ".\n";
        for ( int i = 0; i < parameters.size(); i++ )
        {
            std::cout << "parameter " << i << ": " << parameters.at( i ).getName() << std::endl;
        }
    }

    if ( parameters.size() == 1 )
    {
        ErrorStream << "Experiment contains not enough parameters. Minimum for this modeling approach is 2." << std::endl;
    }

    else if ( parameters.size() == 2 || parameters.size() == 3 || parameters.size() == 4 || parameters.size() == 5 )
    {
        CoordinateList coordinates_tmp = parent_experiment->getCoordinates();
        CoordinateList coordinates;

        //analyze the coordinate list, to filter duplicates in case there are some
        for ( int i = 0; i < coordinates_tmp.size(); i++ )
        {
            if ( i == 0 )
            {
                coordinates.push_back( coordinates_tmp.at( i ) );
            }
            else
            {
                bool found = false;
                for ( int j = 0; j < coordinates.size(); j++ )
                {
                    if ( coordinates.at( j )->toString() == coordinates_tmp.at( i )->toString() )
                    {
                        found = true;
                        break;
                    }
                }
                if ( found == false )
                {
                    coordinates.push_back( coordinates_tmp.at( i ) );
                }
            }
        }

        // debug print for the coordinates input
        if ( DEBUG == true )
        {
            std::cout << "Number of Coordinates: " << coordinates.size() << ".\n";
            for ( int i = 0; i < coordinates.size(); i++ )
            {
                std::cout << "Coordinate " << i << ": " << coordinates.at( i )->toString() << std::endl;
            }
        }

        //find point rows that have at least min points
        for ( int i = 0; i < parameters.size(); i++ )
        {
            Parameter      parameter = parameters[ i ];
            CoordinateList coordinate_container;
            for ( int j = 0; j < coordinates.size(); j++ )
            {
                Coordinate*         reference_coordinate = coordinates[ j ];
                std::vector<double> parameter_value_list = getParameterValues( parameters, reference_coordinate, i );
                for ( int z = 0; z < coordinates.size(); z++ )
                {
                    Coordinate*         coordinate            = coordinates[ z ];
                    std::vector<double> parameter_value_list2 = getParameterValues( parameters, coordinate, i );
                    bool                equal                 = compareParameterValues( parameter_value_list, parameter_value_list2 );
                    if ( equal == true )
                    {
                        coordinate_container.push_back( coordinate );
                    }
                }
                if ( coordinate_container.size() < min_measurement_points )
                {
                    coordinate_container.clear();
                }
                else
                {
                    //save the point row that was found
                    container.at( i ).push_back( coordinate_container );
                    coordinate_container.clear();
                }
            }
        }

        //remove the duplicates in the list
        std::vector<std::vector<CoordinateList> > copy_container;
        for ( int i = 0; i < container.size(); i++ )
        {
            std::vector<CoordinateList> tmp_container_list = container.at( i );
            std::vector<CoordinateList> copy_container_list;
            for ( int j = 0; j < tmp_container_list.size(); j++ )
            {
                if ( copy_container_list.size() == 0 )
                {
                    copy_container_list.push_back( tmp_container_list.at( j ) );
                }
                else
                {
                    bool           in_list   = false;
                    CoordinateList cord_list = tmp_container_list.at( j );
                    for ( int k = 0; k < copy_container_list.size(); k++ )
                    {
                        bool           are_equal  = true;
                        CoordinateList cord_list2 = copy_container_list.at( k );
                        for ( int l = 0; l < cord_list2.size(); l++ )
                        {
                            Coordinate* c1 = cord_list.at( l );
                            Coordinate* c2 = cord_list2.at( l );
                            if ( c1->toString() != c2->toString() )
                            {
                                are_equal = false;
                                break;
                            }
                        }
                        if ( are_equal == true )
                        {
                            in_list = true;
                            break;
                        }
                    }
                    if ( in_list == false )
                    {
                        copy_container_list.push_back( tmp_container_list.at( j ) );
                    }
                }
            }
            copy_container.push_back( copy_container_list );
        }

        //find the cheapest row for each parameter
        for ( int i = 0; i < copy_container.size(); i++ )
        {
            int                         id                 = 0;
            std::vector<CoordinateList> tmp_container_list = copy_container.at( i );
            double                      min_sum_cost       = 0;
            for ( int j = 0; j < tmp_container_list.size(); j++ )
            {
                double         sum_cost = 0;
                CoordinateList tmp_list = tmp_container_list.at( j );
                for ( int k = 0; k < tmp_list.size(); k++ )
                {
                    Coordinate* c        = tmp_list.at( k );
                    std::string c_string = c->toString();
                    //find the data point for this coordinate
                    double time = 0;
                    double p    = 0;
                    for ( int l = 0; l < copy_modeledDataPointList.size(); l++ )
                    {
                        DataPoint          dp  = copy_modeledDataPointList.at( l );
                        double             v   = dp.getValue();
                        std::string cord_string = "";

                        Parameter parameter0 = parameters.at( 0 );
                        double processes = dp.getParameterValue( parameter0 );

                        for (int o = 0; o < parameters.size(); o++)
                        {
                            Parameter parameter = parameters.at( o );
                            double parameter_value = dp.getParameterValue( parameter );
                            std::ostringstream strs;
                            strs << parameter_value;
                            std::string parameter_value_string = strs.str();
                            cord_string = cord_string + "(" + parameter.getName() + "," + parameter_value_string + ")";
                        }
                        
                        //works only when: cost = p * time
                        if ( c_string == cord_string )
                        {
                            p    = processes;
                            time = v;
                            break;
                        }
                    }
                    //calculate cost for this coordinate
                    double cost = p * time;
                    sum_cost = sum_cost + cost;
                }

                //for the first iteration
                if (min_sum_cost == 0) {
                    min_sum_cost = sum_cost;
                    id           = j;
                }
                //check if the current sum_cost is lower than the previous one
                else if ( sum_cost < min_sum_cost )
                {
                    min_sum_cost = sum_cost;
                    id           = j;
                }
            }
            //add the best solution found
            coordinate_container_list.push_back( tmp_container_list.at( id ) );
        }

        // print the result of the search
        if ( DEBUG == true )
        {
            for ( int i = 0; i < coordinate_container_list.size(); i++ )
            {
                Parameter parameter = parameters[ i ];
                std::cout << "Measurement Points for parameter: " << parameter.getName() << ".\n";
                CoordinateList cord_list = coordinate_container_list.at( i );
                for ( int j = 0; j < cord_list.size(); j++ )
                {
                    Coordinate* reference_coordinate = cord_list[ j ];
                    std::string temp                 = reference_coordinate->toString();
                    std::cout << temp << "\n";
                }
                std::cout << "\n";
            }
        }
    }
    else
    {
        ErrorStream << "Experiment contains a number of parameters that is currently not supported." << std::endl;
    }
    return coordinate_container_list;
}

std::vector<double>
MultiParameterSparseFunctionModeler::getParameterValues( ParameterList parameters, Coordinate* reference_coordinate, int parameter_id )
{
    std::vector<double> parameter_value_list;
    for ( int i = 0; i < parameters.size(); i++ )
    {
        if ( i != parameter_id )
        {
            Parameter parameter       = parameters[ i ];
            Value     parameter_value = reference_coordinate->at( parameter.getName() );
            parameter_value_list.push_back( parameter_value );
        }
    }
    return parameter_value_list;
}

bool
MultiParameterSparseFunctionModeler::compareParameterValues( std::vector<double> parameter_value_list1, std::vector<double> parameter_value_list2 )
{
    if ( parameter_value_list1.size() != parameter_value_list2.size() )
    {
        return false;
    }
    for ( int i = 0; i < parameter_value_list1.size(); i++ )
    {
        if ( parameter_value_list1.at( i ) != parameter_value_list2.at( i ) )
        {
            return false;
        }
    }
    return true;
}

bool
MultiParameterSparseFunctionModeler::analyzeMeasurementPoint( ParameterList parameters, std::vector<double> parameter_value_list )
{
    int n = parameters.size() - 1;

    std::cout << "n" << n << std::endl;

    if ( n == 2 )
    {
        std::cout << parameter_value_list.at( 0 ) << std::endl;
        std::cout << parameter_value_list.at( 1 ) << std::endl;

        if ( parameter_value_list.at( 0 ) == parameter_value_list.at( 1 ) )
        {
            return true;
        }
    }
    else
    {
        double value = parameter_value_list.at( 0 );
        for ( int i = 1; i < n; i++ )
        {
            if ( value != parameter_value_list.at( i ) )
            {
                return false;
            }
        }
    }
    return true;
}
}; // Close namespace