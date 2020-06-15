#include "IncrementalPoint.h"
#include "Model.h"
#include "SingleParameterSimpleModelGenerator.h"
#include "SingleParameterRefiningModelGenerator.h"
#include "MultiParameterSimpleModelGenerator.h"
#include "MultiParameterSparseModelGenerator.h"
#include "Utilities.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <sstream>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace EXTRAP
{
Experiment::Experiment( void )
{
}

Experiment::Experiment( const Experiment& e )
{
}

Experiment&
Experiment::operator=( const Experiment& e )
{
    return *this;
}

Experiment::~Experiment( void )
{
    // Delete metric definitions
    for ( MetricList::iterator i = m_metrics.begin();
          i != m_metrics.end();
          i++ )
    {
        delete( *i );
    }

    // Delete region definitions
    for ( RegionList::iterator i = m_regions.begin();
          i != m_regions.end();
          i++ )
    {
        delete( *i );
    }

    // Delete callpath definitions
    for ( CallpathList::iterator i = m_all_callpaths.begin();
          i != m_all_callpaths.end();
          i++ )
    {
        delete( *i );
    }

    // Delete coordinate collection
    for ( CoordinateList::iterator i = m_coordinates.begin();
          i != m_coordinates.end();
          i++ )
    {
        delete( *i );
    }

    // Delete all associated models
    for ( ModelMatrix::iterator i = m_models.begin();
          i != m_models.end();
          i++ )
    {
        for ( std::vector<ModelList>::iterator j = ( *i ).begin();
              j != ( *i ).end();
              j++ )
        {
            for ( ModelList::iterator k = ( *j ).begin();
                  k != ( *j ).end();
                  k++ )
            {
                ModelCommentList comments = ( *k )->getComments();
                for ( ModelCommentList::const_iterator it = comments.begin(); it != comments.end(); it++ )
                {
                    model_comment_reference_counter[ *it ]--;
                    if ( model_comment_reference_counter[ *it ] == 0 )
                    {
                        this->m_comments.erase( this->m_comments.begin() + ( *it )->getId() );
                        delete( *it );
                    }
                }
                delete( *k );
            }
        }
    }
    assert( this->m_comments.size() == 0 );
    // Delete all data points
    for ( ExperimentPointMatrix::iterator i = m_points.begin();
          i != m_points.end();
          i++ )
    {
        for ( std::vector<ExperimentPointList>::iterator j = ( *i ).begin();
              j != ( *i ).end();
              j++ )
        {
            for ( ExperimentPointList::iterator k = ( *j ).begin();
                  k != ( *j ).end();
                  k++ )
            {
                delete( *k );
            }
        }
    }

    // Delete all model comments
    for ( ModelCommentList::iterator i = m_comments.begin();
          i != m_comments.end();
          i++ )
    {
        ;   //delete( *i );
    }

    // Delete all model generators
    for ( ModelGeneratorList::iterator i = m_generators.begin();
          i != m_generators.end();
          i++ )
    {
        delete( *i );
    }
}

const MetricList&
Experiment::getMetrics( void ) const
{
    return this->m_metrics;
}

const RegionList&
Experiment::getRegions( void ) const
{
    return m_regions;
}

const CallpathList&
Experiment::getRootCallpaths( void ) const
{
    return m_root_callpaths;
}

const CallpathList&
Experiment::getAllCallpaths( void ) const
{
    return m_all_callpaths;
}

const ParameterList&
Experiment::getParameters( void ) const
{
    return m_parameters;
}

const CoordinateList&
Experiment::getCoordinates( void ) const
{
    return m_coordinates;
}

const ModelList&
Experiment::getModels( const Metric&   metric,
                       const Callpath& callpath ) const
{
    // Use at() because it performs boundary checking.
    // The [] operator does not.
    if ( m_models.size() > metric.getId() &&
         m_models.at( metric.getId() ).size() > callpath.getId() )
    {
        return m_models.at( metric.getId() ).at( callpath.getId() );
    }
    return m_empty_model_list;
}

const ExperimentPointList&
Experiment::getPoints( const Metric&   metric,
                       const Callpath& callpath ) const
{
    // Use at() because it performs boundary checking.
    // The [] operator does not.
    return m_points.at( metric.getId() ).at( callpath.getId() );
}

const ModelGeneratorList&
Experiment::getModelGenerators( void ) const
{
    return m_generators;
}

const ModelCommentList&
Experiment::getModelComments( void ) const
{
    return m_comments;
}

Parameter
Experiment::getParameter( int id ) const
{
    assert( id >= 0 && id < m_parameters.size() );
    return m_parameters[ id ];
}

Metric*
Experiment::getMetric( int id ) const
{
    assert( id >= 0 && id < m_metrics.size() );
    return m_metrics[ id ];
}

Region*
Experiment::getRegion( int id ) const
{
    assert( id >= 0 && id < m_regions.size() );
    return m_regions[ id ];
}

Callpath*
Experiment::getCallpath( int id ) const
{
    assert( id >= 0 && id < m_all_callpaths.size() );
    return m_all_callpaths[ id ];
}

const Coordinate*
Experiment::getCoordinate( int id ) const
{
    assert( id >= 0 && id < m_coordinates.size() );
    return m_coordinates[ id ];
}

const Coordinate*
Experiment::getCoordinateByIndices( std::vector<Value> indices ) const
{
    if ( indices.size() != this->m_parameters.size() )
    {
        ErrorStream << "Number of Parameters (" << this->m_parameters.size() << ") does not match the number of indices given (" << indices.size() << ")" << std::endl;
        return NULL;
    }
    for ( CoordinateList::const_iterator it = this->m_coordinates.begin(); it != this->m_coordinates.end(); it++ )
    {
        for ( int i = 0; i < indices.size(); i++ )
        {
            if ( ( *it )->find( this->m_parameters[ i ] )->second != indices[ i ] )
            {
                // if index does not fit check next coordinate
                break;
            }
            else if ( i == indices.size() - 1 )
            {
                //At this point we have checked all indexes for the coordinate and have found the coordinate.
                return *it;
            }
        }
    }
    ErrorStream << "Could not find a Coordinate for the given indices." << std::endl;
    return NULL;
}

const std::set<Value>
Experiment::getValuesForParameter( const Parameter& parameter ) const
{
    std::set<Value> valueSet;
    for ( CoordinateList::const_iterator coordinate = m_coordinates.begin(); coordinate != m_coordinates.end(); coordinate++ )
    {
        if ( ( *coordinate )->find( parameter ) != ( *coordinate )->end() )
        {
            valueSet.insert( ( *coordinate )->find( parameter )->second );
        }
    }
    return valueSet;
}

ModelGenerator*
Experiment::getModelGenerator( int id ) const
{
    assert( id >= 0 && id < m_generators.size() );
    return m_generators[ id ];
}

ModelComment*
Experiment::getModelComment( int id ) const
{
    //test
    //original code!!!
    //assert( id >= 0 && id < m_comments.size() );
    //return m_comments[ id ];

    //ErrorStream << "Comment ID: " << id << std::endl;
    if ( id >= 0 && id < m_comments.size() )
    {
        return m_comments[ id ];
    }
    else
    {
        ModelComment* new_comment = new ModelComment( "empty" );
        return new_comment;
    }
}

void
Experiment::addParameter( Parameter& newParameter )
{
    newParameter.setId( m_parameters.size() );
    m_parameters.push_back( newParameter );
}

void
Experiment::addMetric( Metric* newMetric )
{
    if ( newMetric == NULL )
    {
        return;
    }
    newMetric->setId( m_metrics.size() );
    m_metrics.push_back( newMetric );
    std::sort( this->m_metrics.begin(), this->m_metrics.end(), less );
    for ( int i = 0; i < this->m_metrics.size(); i++ )
    {
        this->m_metrics[ i ]->setId( i );
    }
}

void
Experiment::addRegion( Region* newRegion )
{
    if ( newRegion == NULL )
    {
        return;
    }
    newRegion->setId( m_regions.size() );
    m_regions.push_back( newRegion );
}

void
Experiment::addCoordinate( Coordinate* newCoordinate )
{
    if ( newCoordinate == NULL )
    {
        return;
    }
    newCoordinate->setId( m_coordinates.size() );
    m_coordinates.push_back( newCoordinate );
    std::sort( m_coordinates.begin(), m_coordinates.end(), lessCoordinate );
    for ( int i = 0; i < m_coordinates.size(); i++ )
    {
        m_coordinates[ i ]->setId( i );
    }
}

void
Experiment::addCallpath( Callpath* newCallpath )
{
    if ( newCallpath == NULL )
    {
        return;
    }
    m_all_callpaths.push_back( newCallpath );
    if ( newCallpath->getParent() == NULL )
    {
        m_root_callpaths.push_back( newCallpath );
    }
    std::sort( m_all_callpaths.begin(), m_all_callpaths.end(), lessCallpath );
    for ( int i = 0; i < m_all_callpaths.size(); i++ )
    {
        m_all_callpaths[ i ]->setId( i );
    }
}

void
Experiment::addDataPoint( ExperimentPoint* newDataPoint,
                          const Metric&    metric,
                          const Callpath&  callpath )
{
    if ( newDataPoint == NULL )
    {
        std::cout << "Method addDataPoint in class Experiment says: experiment point is null!" << std::endl;
        return;
    }
    for ( int i = m_points.size(); i <= metric.getId(); i++ )
    {
        m_points.push_back( std::vector<ExperimentPointList>() );
    }
    for ( int i = m_points[ metric.getId() ].size(); i <= callpath.getId(); i++ )
    {
        m_points[ metric.getId() ].push_back( ExperimentPointList() );
    }
    m_points[ metric.getId() ][ callpath.getId() ].push_back( newDataPoint );
    std::sort( m_points[ metric.getId() ][ callpath.getId() ].begin(), m_points[ metric.getId() ][ callpath.getId() ].end(), lessExperimentPoint );
}

void
Experiment::addModel( Model*          newModel,
                      const Metric&   metric,
                      const Callpath& callpath )
{
    if ( newModel == NULL )
    {
        return;
    }
    for ( int i = m_models.size(); i <= metric.getId(); i++ )
    {
        m_models.push_back( std::vector<ModelList>() );
    }
    for ( int i = m_models[ metric.getId() ].size(); i <= callpath.getId(); i++ )
    {
        m_models[ metric.getId() ].push_back( ModelList() );
    }
    ModelCommentList commentList = newModel->getComments();
    for ( int i = 0; i < commentList.size(); i++ )
    {
        addModelComment( commentList[ i ]->getMessage() );
    }

    m_models[ metric.getId() ][ callpath.getId() ].push_back( newModel );
}

void
Experiment::addModelGenerator( ModelGenerator* newGenerator )
{
    if ( newGenerator == NULL )
    {
        return;
    }
    newGenerator->setId( m_generators.size() );
    m_generators.push_back( newGenerator );
}

void
Experiment::addModelComment( ModelComment* newComment )
{
    if ( newComment == NULL )
    {
        return;
    }
    newComment->setId( m_comments.size() );
    model_comment_reference_counter[ newComment ] = 1;
    m_comments.push_back( newComment );
}

ModelComment*
Experiment::addModelComment( const std::string& message )
{
    for ( ModelCommentList::iterator it = m_comments.begin();
          it != m_comments.end();
          it++ )
    {
        if ( ( *it )->getMessage() == message )
        {
            model_comment_reference_counter[ *it ]++;
            return *it;
        }
    }
    ModelComment* newComment = new ModelComment( message );
    addModelComment( newComment );
    return newComment;
}

void
Experiment::modelAll( ModelGenerator& generator, Experiment* experiment, ModelGeneratorOptions options )
{
    for ( MetricList::iterator metric_it = m_metrics.begin();
          metric_it != m_metrics.end();
          metric_it++ )
    {
        Metric* metric = *metric_it;
        for ( CallpathList::iterator callpath_it = m_all_callpaths.begin();
              callpath_it != m_all_callpaths.end();
              callpath_it++ )
        {
            Callpath*                  callpath    = *callpath_it;
            const ExperimentPointList& data_points = getPoints( *metric, *callpath );
            Model*                     model       = generator.createModel( experiment, options, data_points );
            
            assert( model->getModelFunction() );
            addModel( model, *metric, *callpath );
        }
    }
}

void
Experiment::deleteModel( int             modelIndex,
                         const Metric&   metric,
                         const Callpath& callpath )
{
    ModelList&       models   = ( ModelList& )getModels( metric, callpath );
    Model*           model    = models[ modelIndex ];
    ModelCommentList comments = model->getComments();
    for ( ModelCommentList::const_iterator it = comments.begin(); it != comments.end(); it++ )
    {
        model_comment_reference_counter[ *it ]--;
        if ( model_comment_reference_counter[ *it ] == 0 )
        {
            this->m_comments.erase( this->m_comments.begin() + ( *it )->getId() );
            delete( *it );
        }
    }
    models.erase( models.begin() + modelIndex );
    delete( model );
}

void
Experiment::deleteModel( int modelIndex )
{
    for ( MetricList::iterator metric_it = m_metrics.begin();
          metric_it != m_metrics.end();
          metric_it++ )
    {
        Metric* metric = *metric_it;
        for ( CallpathList::iterator callpath_it = m_all_callpaths.begin();
              callpath_it != m_all_callpaths.end();
              callpath_it++ )
        {
            deleteModel( modelIndex,
                         *metric,
                         **callpath_it );
        }
    }
}

bool
Experiment::writeExtrapFile( const std::string& filename ) const
{
    std::ofstream dataFile( filename.c_str(), std::ios::binary );
    if ( !serialize( dataFile ) )
    {
        ErrorStream << "Could not write File successfully" << std::endl;
        dataFile.close();
        remove( filename.c_str() );
        return false;
    }
    else
    {
        dataFile.close();
        return true;
    }
}

Experiment*
Experiment::openExtrapFile( const std::string& filename )
{
    std::ifstream dataFile( filename.c_str(), std::ios::binary );
    Experiment*   new_experiment;
    if ( dataFile.good() )
    {
        new_experiment = deserialize( dataFile );
    }
    else
    {
        ErrorStream << "File " << filename << " does not exist" << std::endl;
        return NULL;
    }
    if ( new_experiment == NULL )
    {
        ErrorStream << "Could not read File successfully" << std::endl;
    }
    dataFile.close();
    /**
     * The validation needs to be deactivated, otherwise the printer tool will throw an error when reading a extrap file that was created from the json input function with less coordinates.
       if ( new_experiment == NULL || !new_experiment->validate_experiment() )
       {
        return NULL;
       }
       else
       {
        return new_experiment;
       }
     **/
    return new_experiment;
}

Experiment*
Experiment::openHDF5File( const std::string& filename )
{
    #ifdef USE_HDF5
    HDF5Reader  reader( filename );
    Experiment* new_experiment = reader.read();
    if ( new_experiment == NULL || !new_experiment->validate_experiment() )
    {
        return NULL;
    }
    else
    {
        return new_experiment;
    }
    #else
    {
        ErrorStream << "ExtraP was installed without HDF5 support" << std::endl;
    }
    return NULL;
    #endif
}

Experiment*
Experiment::openJsonInput( const std::string& filename )
{
    std::ifstream dataFile( filename.c_str() );
    std::string   lineString;
    std::string   fieldName;
    if ( dataFile.is_open() == false )
    {
        ErrorStream << "Unable to open the input file " << filename << std::endl;

        return NULL;
    }
    Experiment*              new_experiment = new Experiment();
    Metric*                  new_metric     = NULL;
    Parameter                parameter;
    CoordinateList           pointslist;
    Region*                  region = NULL;
    Callpath*                callpath;
    std::vector<std::string> lines;

    int currentLine = 0;
    while ( getline( dataFile, lineString, '\n' ) )
    {
        currentLine++;

        // ignore empty lines
        if ( lineString.empty() )
        {
            continue;
        }

        // ignore comments
        if ( lineString.c_str()[ 0 ] == '#' )
        {
            continue;
        }

        // copy the current line and store for later use
        lines.push_back( lineString );

        // parse parameter and add it to the experiment
        int         pos = 0;
        std::string p   = lineString.substr( 11, lineString.size() );
        pos = p.find( "}" );
        p   = p.substr( 0, pos );
        std::vector<std::string> ps;
        std::vector<std::string> ys;
        std::string              delimiter = ",";
        std::string              token;
        while ( ( pos = p.find( delimiter ) ) != std::string::npos )
        {
            token = p.substr( 0, pos );
            ps.push_back( token );
            ys.push_back( token );
            p.erase( 0, pos + delimiter.length() );
        }
        ps.push_back( p );
        ys.push_back( p );
        for ( int i = 0; i < ps.size(); i++ )
        {
            std::string tp = ps.at( i );
            tp         = tp.substr( 1, tp.size() );
            pos        = tp.find( ":" );
            tp         = tp.substr( 0, pos - 1 );
            ps.at( i ) = tp;
        }
        if ( currentLine == 1 )
        {
            for ( int i = 0; i < ps.size(); i++ )
            {
                parameter = Parameter( ps.at( i ) );
                //debug print for parameter detection
                //std::cout << parameter.getName() << std::endl;
                new_experiment->addParameter( parameter );
            }
        }

        // parse the metric and add it to the experiment
        std::string m = lineString;
        pos = m.find( "metric" );
        m   = m.substr( pos + 9, m.size() );
        pos = m.find( "," );
        m   = m.substr( 0, pos - 1 );
        //debug print for the metric detection
        //std::cout << "metric: " << m << std::endl;
        if ( currentLine == 1 )
        {
            new_metric = new Metric( m, "unknown" );
            new_experiment->addMetric( new_metric );
        }

        // parse the region and callpath and add it to the experiment
        if ( currentLine == 1 )
        {
            region = new Region( "reg", filename, 1 );
            new_experiment->addRegion( region );
            callpath = new Callpath( region, NULL );
            new_experiment->addCallpath( callpath );
        }

        // parse the measurement points and add them to the experiment
        for ( int i = 0; i < ys.size(); i++ )
        {
            std::string ty = ys.at( i );
            pos        = ty.find( ":" );
            ty         = ty.substr( pos + 1, ty.size() );
            ys.at( i ) = ty;
        }
        // check if the measurement point was already added previously
        //read the first line
        if ( pointslist.size() == 0 )
        {
            Coordinate* pv_list = new Coordinate();
            for ( int i = 0; i < ys.size(); i++ )
            {
                double z;
                std::istringstream iss( ys.at( i ) );
                iss >> z;
                parameter = Parameter( ps.at( i ) );
                ( *pv_list )[ parameter ] = z;
            }
            pointslist.push_back( pv_list );
            new_experiment->addCoordinate( pv_list );
        }
        //read the other lines
        else
        {
            Coordinate* pv_list = new Coordinate();
            for ( int i = 0; i < ys.size(); i++ )
            {
                double z;
                std::istringstream iss( ys.at( i ) );
                iss >> z;
                parameter                 = Parameter( ps.at( i ) );
                ( *pv_list )[ parameter ] = z;
            }
            bool exists = false;
            for ( int i = 0; i < pointslist.size(); i++ )
            {
                Coordinate* pv_list2 = pointslist.at( i );
                bool        equal    = true;
                for ( int j = 0; j < ys.size(); j++ )
                {
                    parameter = Parameter( ps.at( j ) );
                    double pv_list2_value = ( *pv_list2 )[ parameter ];
                    double pv_list_value  = ( *pv_list )[ parameter ];
                    if ( pv_list2_value != pv_list_value )
                    {
                        equal = false;
                        break;
                    }
                }
                if ( equal == true )
                {
                    exists = true;
                    break;
                }
            }
            if ( exists == false )
            {
                pointslist.push_back( pv_list );
                new_experiment->addCoordinate( pv_list );
            }
        }
    }

    //add measurements
    IncrementalPoint         data_point;
    std::vector<std::string> base_mp_list;

    for ( int i = 0; i < lines.size(); i++ )
    {
        std::string line    = lines.at( i );
        std::string base_mp = line;
        int         pos     = 0;

        //get basemp
        pos     = base_mp.find( "params" );
        pos     = pos + 8;
        base_mp = base_mp.substr( pos, base_mp.size() );
        pos     = 0;
        pos     = base_mp.find( "}" );
        pos     = pos + 1;
        base_mp = base_mp.substr( 0, pos );
        pos     = 0;

        //check if mp was already added to exp
        bool in_list = false;
        for ( int j = 0; j < base_mp_list.size(); j++ )
        {
            if ( base_mp == base_mp_list.at( j ) )
            {
                in_list = true;
                break;
            }
        }
        if ( in_list == true )
        {
            continue;
        }

        data_point.clear();

        for ( int j = 0; j < lines.size(); j++ )
        {
            std::string line        = lines.at( j );
            std::string new_mp      = line;
            std::string measurement = line;

            //get new mp
            pos    = new_mp.find( "params" );
            pos    = pos + 8;
            new_mp = new_mp.substr( pos, new_mp.size() );
            pos    = 0;
            pos    = new_mp.find( "}" );
            pos    = pos + 1;
            new_mp = new_mp.substr( 0, pos );
            pos    = 0;

            //check if base mp and new mp are equal
            if ( new_mp == base_mp )
            {
                //get measurement
                pos         = measurement.find( "value" );
                pos         = pos + 7;
                measurement = measurement.substr( pos, measurement.size() );
                measurement = measurement.substr( 0, measurement.size() - 1 );
                double m;
                std::istringstream iss( measurement );
                iss >> m;
                pos = 0;
                //debug print out for the read in of the measured value at coord
                //std::cout << "Measured Value: " << m << "\n";
                //add measurement to incremental point
                data_point.addValue( m );
            }
        }

        //cast base mp to extrap cord format
        std::string base_mp_extrap = base_mp;

        //remove {}
        base_mp_extrap = base_mp_extrap.substr( 1, base_mp_extrap.size() );
        base_mp_extrap = base_mp_extrap.substr( 0, base_mp_extrap.size() - 1 );

        //remove "
        base_mp_extrap.erase( std::remove( base_mp_extrap.begin(), base_mp_extrap.end(), '"' ), base_mp_extrap.end() );

        //add ()
        base_mp_extrap.insert( 0, "(" );
        base_mp_extrap.append( ")" );
        pos = 0;
        //get the number of , in the string
        int number_of_commas = std::count( base_mp_extrap.begin(), base_mp_extrap.end(), ',' );

        if ( number_of_commas == 0 )
        {
            replace( base_mp_extrap.begin(), base_mp_extrap.end(), ':', ',' );
        }
        else
        {
            for ( int counter = 0; counter < number_of_commas; counter++ )
            {
                //remove ,
                pos = base_mp_extrap.find( "," );
                base_mp_extrap.erase( pos, 1 );
                //add ()
                base_mp_extrap.insert( pos, "(" );
                base_mp_extrap.insert( pos, ")" );
                pos = 0;
            }
            //change : to ,
            replace( base_mp_extrap.begin(), base_mp_extrap.end(), ':', ',' );
        }

        //find base mp in cord list
        int id = -1;
        for ( int j = 0; j < pointslist.size(); j++ )
        {
            Coordinate* c = pointslist[ j ];
            std::string s = c->toString();

            //change the base mp order to match the one of the coordinates...

            //split the basemp into param and value and save them in a list
            std::string              tmp = base_mp_extrap;
            std::vector<std::string> tmp_list;
            std::vector<std::string> param_list;
            std::vector<std::string> value_list;
            int                      number_of_commas = std::count( tmp.begin(), tmp.end(), ',' );
            int                      pos              = 0;

            for ( int counter = 0; counter < number_of_commas; counter++ )
            {
                pos = tmp.find( ")" );
                tmp_list.push_back( tmp.substr( 0, pos ) );
                tmp = tmp.substr( pos + 1, tmp.size() );
                pos = 0;
            }

            for ( int counter = 0; counter < tmp_list.size(); counter++ )
            {
                tmp_list[ counter ] = tmp_list[ counter ].substr( 1, tmp_list[ counter ].size() );
            }

            for ( int counter = 0; counter < tmp_list.size(); counter++ )
            {
                pos = tmp_list[ counter ].find( "," );
                std::string temp_string = tmp_list[ counter ];
                temp_string = temp_string.substr( 0, pos );
                param_list.push_back( temp_string );
                pos = 0;
            }

            for ( int counter = 0; counter < tmp_list.size(); counter++ )
            {
                pos = tmp_list[ counter ].find( "," );
                std::string temp_string = tmp_list[ counter ];
                temp_string = temp_string.substr( pos + 1, temp_string.size() );
                value_list.push_back( temp_string );
                pos = 0;
            }

            //split the coord into param and value and save them in a list
            std::string              tmp2 = s;
            std::vector<std::string> tmp_list2;
            std::vector<std::string> param_list2;
            std::vector<std::string> value_list2;
            number_of_commas = std::count( tmp2.begin(), tmp2.end(), ',' );
            pos              = 0;

            for ( int counter = 0; counter < number_of_commas; counter++ )
            {
                pos = tmp2.find( ")" );
                tmp_list2.push_back( tmp2.substr( 0, pos ) );
                tmp2 = tmp2.substr( pos + 1, tmp2.size() );
                pos  = 0;
            }

            for ( int counter = 0; counter < tmp_list2.size(); counter++ )
            {
                tmp_list2[ counter ] = tmp_list2[ counter ].substr( 1, tmp_list2[ counter ].size() );
            }

            for ( int counter = 0; counter < tmp_list2.size(); counter++ )
            {
                pos = tmp_list2[ counter ].find( "," );
                std::string temp_string = tmp_list2[ counter ];
                temp_string = temp_string.substr( 0, pos );
                param_list2.push_back( temp_string );
                pos = 0;
            }

            for ( int counter = 0; counter < tmp_list2.size(); counter++ )
            {
                pos = tmp_list2[ counter ].find( "," );
                std::string temp_string = tmp_list2[ counter ];
                temp_string = temp_string.substr( pos + 1, temp_string.size() );
                value_list2.push_back( temp_string );
                pos = 0;
            }

            //reconstruct the base mp with the correct order...
            std::string base_mp_extrap2 = "";
            for ( int counter = 0; counter < param_list2.size(); counter++ )
            {
                std::string par = param_list2.at( counter );
                int         id  = 0;

                //search for the id in the base mp param list
                for ( int counter2 = 0; counter2 < param_list.size(); counter2++ )
                {
                    if ( param_list[ counter2 ] == par )
                    {
                        id = counter2;
                        break;
                    }
                }

                base_mp_extrap2 = base_mp_extrap2 + "(" + param_list[ id ] + "," + value_list[ id ] + ")";
            }
            if ( s == base_mp_extrap2 )
            {
                id = j;
                break;
            }
        }

        if ( id != -1 )
        {
            //add incremental point to exp
            new_experiment->addDataPoint( data_point.getExperimentPoint( pointslist[ id ], new_metric, callpath ), *new_metric, *callpath );
        }

        //remember that this mp was already added to the exp
        base_mp_list.push_back( base_mp );
    }

    dataFile.close();

    return new_experiment;
}

Experiment*
Experiment::openTextInput( const std::string& filename )
{
    std::ifstream dataFile( filename.c_str() );
    std::string   lineString;
    std::string   fieldName;
    if ( dataFile.is_open() == false )
    {
        ErrorStream << "Unable to open the test data file " << filename << std::endl;

        return NULL;
    }
    Experiment*      new_experiment = new Experiment();
    Metric*          new_metric     = NULL;
    Parameter        parameter;
    CoordinateList   pointslist;
    Region*          new_region = NULL;
    Region*          old_region = NULL;
    Callpath*        callpath;
    IncrementalPoint data_point;
    int              iterator = 0;

    int currentLine = 0;
    while ( getline( dataFile, lineString, '\n' ) )
    {
        // get a line and store in lineString
        currentLine++;

        // ignore empty lines
        if ( lineString.empty() )
        {
            continue;
        }
        // ignore comments
        if ( lineString.c_str()[ 0 ] == '#' )
        {
            continue;
        }

        // get name of field
        std::istringstream iss( lineString );
        iss >> fieldName;
        std::string tmp;

        if ( fieldName == "METRIC" )
        {
            tmp.clear();
            while ( iss >> tmp )
            {
                new_metric = new Metric( tmp, "unknown" );
                new_experiment->addMetric( new_metric );
                tmp.clear();
            }
        }
        else if ( fieldName == "REGION" )
        {
            tmp.clear();
            while ( iss >> tmp )
            {
                new_region = new Region( tmp, filename, 1 );
                new_experiment->addRegion( new_region );
                callpath = new Callpath( new_region, NULL );
                new_experiment->addCallpath( callpath );
                tmp.clear();
            }
        }
        else if ( fieldName == "DATA" )
        {
            if ( new_metric == NULL )
            {
                new_metric = new Metric( "Test", "unknown" );
                new_experiment->addMetric( new_metric );
            }
            data_point.clear();
            while ( !iss.eof() )
            {
                double measured_point;
                while ( iss >> measured_point )
                {
                    data_point.addValue( measured_point );
                }
            }
            if ( new_region == NULL )
            {
                ErrorStream << "In line "
                            << currentLine
                            << ": experiment line missing "
                            << std::endl;

                return NULL;
            }
            if ( ( iterator == 0 ) && ( old_region == new_region ) && ( old_region != NULL ) )
            {
                ErrorStream << "In line "
                            << currentLine
                            << ": too many data points lines before new experiment  "
                            << std::endl;

                return NULL;
            }
            if ( ( iterator != 0 ) && ( old_region != new_region ) && ( old_region != NULL ) )
            {
                ErrorStream << "In line "
                            << currentLine
                            << ": too few data points lines before new experiment "
                            << std::endl;

                return NULL;
            }
            if ( iterator >= pointslist.size() )
            {
                ErrorStream << "In line "
                            << currentLine
                            << ": measurement points should be defined "
                            << std::endl;

                return NULL;
            }
            new_experiment->addDataPoint( data_point.getExperimentPoint( pointslist[ iterator ],
                                                                         new_metric,
                                                                         callpath ),
                                          *new_metric,
                                          *callpath );
            iterator++;
            if ( iterator == pointslist.size() )
            {
                iterator = 0;
            }
            old_region = new_region;
        }

        else if ( fieldName == "PARAMETER" )
        {
            while ( !iss.eof() )
            {
                tmp.clear();
                while ( iss >> tmp )
                {
                    parameter = Parameter( tmp );
                    tmp.clear();
                    new_experiment->addParameter( parameter );
                }
            }
            if ( new_experiment->getParameters().size() > 3 )
            {
                ErrorStream << " Only 3 parameters are currently supported." << std::endl;
                return NULL;
            }
        }

        else if ( fieldName == "POINTS" )
        {
            const ParameterList& params_list = new_experiment->getParameters();
            double               value;
            char                 c;
            int                  input_str_size = iss.str().size();
            while ( !iss.fail() && iss.tellg() < input_str_size )
            {
                Coordinate* pv_list = new Coordinate();

                iss >> c;
                for ( unsigned int i = 0; i < params_list.size(); ++i )
                {
                    iss >> value;
                    ( *pv_list )[ params_list[ i ] ] = value;
                }
                iss >> c;
                pointslist.push_back( pv_list );
                new_experiment->addCoordinate( pv_list );
            }
        }
        else
        {
            ErrorStream << fieldName
                        << " not a valid test file row start value.";

            return NULL;
        }
    }

    dataFile.close();

    if ( !new_experiment->validate_experiment() )
    {
        return NULL;
    }
    else
    {
        return new_experiment;
    }
}

Experiment*
Experiment::createModelGenerator( Experiment* experiment )
{
    MetricList      metrics         = experiment->getMetrics();
    CallpathList    callpaths       = experiment->getAllCallpaths();
    ModelGenerator* model_generator = NULL;
    unsigned int    nparams         = experiment->getParameters().size();
    bool allow_log_terms = true;

    if ( nparams == 1 )
    {
        SingleParameterSimpleModelGenerator* single_generator =
            new SingleParameterSimpleModelGenerator();
        single_generator->setEpsilon( 0.05 );
        single_generator->setMaxTermCount( 1 );
        model_generator = single_generator;
    }
    else
    {
        MultiParameterSimpleModelGenerator* mparam_generator =
            new MultiParameterSimpleModelGenerator();
        model_generator = mparam_generator;
    }

    //check if the parameter value is smaller than 1, then do not allow log terms...
    CoordinateList cordlist = experiment->getCoordinates();
    bool allow_log = true;
    for ( int i = 0; i < cordlist.size(); i++ )
    {
        std::string tmp = cordlist.at( i )->toString();
        //std::cout << "cord: " << tmp << std::endl;
        int pos = tmp.find(")");
        std::string value1 = tmp.substr(0,pos+1);
        //std::cout << "value1: " << value1 << std::endl;
        tmp = tmp.substr(pos+1,tmp.length());
        pos = tmp.find(")");
        std::string value2 = tmp.substr(0,pos+1);
        //std::cout << "value2: " << value2 << std::endl;
        tmp = tmp.substr(pos+1,tmp.length());
        std::string value3 = tmp;
        //std::cout << "value3: " << value3 << std::endl;
        pos = value1.find(",");
        value1 = value1.substr(pos+1,value1.length());
        value1 = value1.substr(0, value1.length()-1);
        double v1 = 0;
        std::istringstream iss( value1 );
        iss >> v1;
        //std::cout << "v1: " << v1 << std::endl;
        pos = value2.find(",");
        value2 = value2.substr(pos+1,value2.length());
        value2 = value2.substr(0, value2.length()-1);
        double v2 = 0;
        std::istringstream iss2( value2 );
        iss2 >> v2;
        //std::cout << "v2: " << v2 << std::endl;
        pos = value3.find(",");
        value3 = value3.substr(pos+1,value3.length());
        value3 = value3.substr(0, value3.length()-1);
        double v3 = 0;
        std::istringstream iss3( value3 );
        iss3 >> v3;
        //std::cout << "v3: " << v3 << std::endl;
        if(v1<1){
            allow_log = false;
            break;
        }
        else if(v2<1){
            allow_log = false;
            break;
        }
        else if(v3<1){
            allow_log = false;
            break;
        }
    }
    allow_log_terms = allow_log;

    //set the modeler options
    ModelGeneratorOptions options;
    options.setGenerateModelOptions( GENERATE_MODEL_MEAN );
    options.setMinNumberPoints( 5 );
    options.setMultiPointsStrategy( INCREASING_COST );
    options.setNumberAddPoints( 0 );
    options.setSinglePointsStrategy( FIRST_POINTS_FOUND );
    options.setUseAddPoints( false );
    options.setUseAutoSelect( false );
    options.setUseLogTerms(allow_log_terms);

    experiment->addModelGenerator( model_generator );
    experiment->modelAll( *model_generator, experiment, options );
    return experiment;
}

bool
Experiment::serialize( std::ofstream& stream ) const
{
    IoHelper ioHelper = IoHelper( &stream );
    //Write Prefix and Version Specifier
    SAFE_RETURN( ioHelper.writeString( EXTRAP_CLASSIFIER ) );
    SAFE_RETURN( ioHelper.writeString( VERSION_QUALIFIER ) );
    //Write Parameters
    for ( ParameterList::const_iterator it = this->m_parameters.begin(); it != this->m_parameters.end(); it++ )
    {
        Parameter p = *it;
        SAFE_RETURN(  p.serialize(  &ioHelper ) );
    }
    for ( MetricList::const_iterator it = this->m_metrics.begin(); it != this->m_metrics.end(); it++ )
    {
        Metric* m = *it;
        SAFE_RETURN(  m->serialize(  &ioHelper ) );
    }
    for ( RegionList::const_iterator it = this->m_regions.begin(); it != this->m_regions.end(); it++ )
    {
        Region* r = *it;
        SAFE_RETURN(  r->serialize(  &ioHelper ) );
    }
    for ( CallpathList::const_iterator it = this->m_all_callpaths.begin(); it != this->m_all_callpaths.end(); it++ )
    {
        Callpath* cp = *it;
        SAFE_RETURN(  cp->serialize(  &ioHelper ) );
    }
    for ( CoordinateList::const_iterator it = this->m_coordinates.begin(); it != this->m_coordinates.end(); it++ )
    {
        SAFE_RETURN(  ( *it )->serialize(  &ioHelper ) );
    }
    for ( ModelCommentList::const_iterator it = this->m_comments.begin(); it != this->m_comments.end(); it++ )
    {
        SAFE_RETURN(  ( *it )->serialize(  &ioHelper ) );
    }
    for ( ModelGeneratorList::const_iterator it = this->m_generators.begin(); it != this->m_generators.end(); it++ )
    {
        SAFE_RETURN(  ( *it )->serialize(  &ioHelper ) );
    }
    for ( MetricList::const_iterator it = this->m_metrics.begin(); it != this->m_metrics.end(); it++ )
    {
        for ( CallpathList::const_iterator it2 = this->m_all_callpaths.begin(); it2 != this->m_all_callpaths.end(); it2++ )
        {
            ModelList list = this->getModels( **it, **it2 );
            for ( ModelList::iterator it3 = list.begin(); it3 != list.end(); it3++ )
            {
                SAFE_RETURN(  ( *it3 )->serialize( *it, *it2, &ioHelper ) );
            }
        }
    }
    int points = 0;
    for ( MetricList::const_iterator it = this->m_metrics.begin(); it != this->m_metrics.end(); it++ )
    {
        for ( CallpathList::const_iterator it2 = this->m_all_callpaths.begin(); it2 != this->m_all_callpaths.end(); it2++ )
        {
            ExperimentPointList list = this->getPoints( **it, **it2 );
            points += list.size();
            for ( ExperimentPointList::const_iterator it3 = list.begin(); it3 != list.end(); it3++ )
            {
                SAFE_RETURN(  ( *it3 )->serialize(  &ioHelper ) );
            }
        }
    }
    return true;
}

Experiment*
Experiment::deserialize( std::ifstream& stream )
{
    IoHelper    ioHelper = IoHelper( &stream );
    Experiment* exp      = new Experiment();
    //Write Prefix and Version Specifier
    std::string qualifier = ioHelper.readString();
    if ( 0 != qualifier.compare( EXTRAP_CLASSIFIER ) )
    {
        ErrorStream << "This is not an EXTRAP Experiment File. Qualifier was " << qualifier << std::endl;
        return NULL;
    }
    std::string versionNumber = ioHelper.readString();
    char        c;
    std::string prefix = ioHelper.readString();
    while ( stream.good() )
    {
        DebugStream << "Deserialize " << prefix << std::endl;
        if ( 0 == prefix.compare( Parameter::PARAMETER_PREFIX ) )
        {
            Parameter p = Parameter::deserialize( &ioHelper );
            exp->addParameter( p );
        }
        else if ( 0 == prefix.compare( Metric::METRIC_PREFIX ) )
        {
            Metric* m = Metric::deserialize( &ioHelper );
            SAFE_RETURN_NULL(  m );
            exp->addMetric( m );
        }
        else if ( 0 == prefix.compare( Region::REGION_PREFIX ) )
        {
            Region* r = Region::deserialize( &ioHelper );
            SAFE_RETURN_NULL(  r );
            exp->addRegion( r );
        }
        else if ( 0 == prefix.compare( Callpath::CALLPATH_PREFIX ) )
        {
            Callpath* cp = Callpath::deserialize( exp, &ioHelper );
            exp->addCallpath( cp );
        }
        else if ( 0 == prefix.compare( Coordinate::COORDINATE_PREFIX ) )
        {
            Coordinate* c = Coordinate::deserialize( exp, &ioHelper );
            SAFE_RETURN_NULL(  c );
            exp->addCoordinate( c );
        }
        else if ( 0 == prefix.compare( ModelComment::MODELCOMMENT_PREFIX ) )
        {
            ModelComment* comment = ModelComment::deserialize( &ioHelper );
            SAFE_RETURN_NULL(  comment );
            exp->addModelComment( comment );
        }
        else if ( 0 == prefix.compare( SingleParameterSimpleModelGenerator::SINGLEPARAMETERSIMPLEMODELGENERATOR_PREFIX ) )
        {
            ModelGenerator* generator = SingleParameterSimpleModelGenerator::deserialize( &ioHelper );
            SAFE_RETURN_NULL(  generator );
            exp->addModelGenerator( generator );
        }
        else if ( 0 == prefix.compare( SingleParameterRefiningModelGenerator::SINGLEPARAMETERREFININGMODELGENERATOR_PREFIX ) )
        {
            ModelGenerator* generator = SingleParameterRefiningModelGenerator::deserialize( &ioHelper );
            SAFE_RETURN_NULL(  generator );
            exp->addModelGenerator( generator );
        }
        else if ( 0 == prefix.compare( MultiParameterSimpleModelGenerator::MULTIPARAMETERSIMPLEMODELGENERATOR_PREFIX ) )
        {
            ModelGenerator* generator = MultiParameterSimpleModelGenerator::deserialize( &ioHelper );
            SAFE_RETURN_NULL(  generator );
            exp->addModelGenerator( generator );
        }
        else if ( 0 == prefix.compare( MultiParameterSparseModelGenerator::MULTIPARAMETERSPARSEMODELGENERATOR_PREFIX ) )
        {
            ModelGenerator* generator = MultiParameterSparseModelGenerator::deserialize( &ioHelper );
            SAFE_RETURN_NULL(  generator );
            exp->addModelGenerator( generator );
        }
        else if ( 0 == prefix.compare( ExperimentPoint::EXPERIMENTPOINT_PREFIX ) )
        {
            ExperimentPoint* point = ExperimentPoint::deserialize( exp, &ioHelper );
            SAFE_RETURN_NULL(  point );
            exp->addDataPoint( point,
                               *point->getMetric(),
                               *point->getCallpath() );
        }
        else if ( 0 == prefix.compare( Model::MODEL_PREFIX ) )
        {
            int64_t metric_id;
            int64_t callpath_id;
            Model*  model = Model::deserialize( exp, &metric_id, &callpath_id, &ioHelper );
            SAFE_RETURN_NULL(  model );
            exp->addModel( model,
                           *exp->getMetric( metric_id ),
                           *exp->getCallpath( callpath_id ) );
        }
        else
        {
            ErrorStream << "Unknown object: " << prefix << ". Can not load experiment." << std::endl;
            return NULL;
        }

        prefix = ioHelper.readString();
    }
    /**
     * needs to be deactivated, otherwise experiment files created by the json input method with less coordinates will not load properly in the printer tool.
       if ( !exp->validate_experiment() )
       {
        return NULL;
       }
       else
       {
        return exp;
       }
     **/
    //new code
    return exp;
}

bool
equal( const Experiment* lhs, const Experiment* rhs )
{
    if ( lhs == rhs )
    {
        return true;
    }
    else if ( lhs == NULL || rhs == NULL )
    {
        return false;
    }
    bool                         result        = true;
    const std::vector<Parameter> parametersLhs = lhs->getParameters();
    const std::vector<Parameter> parametersRhs = rhs->getParameters();
    if ( parametersLhs.size() != parametersRhs.size() )
    {
        ErrorStream << "#Parameters does not match" << std::endl;
        return false;
    }
    int length = parametersLhs.size();
    for ( int i = 0; i < length; i++ )
    {
        if ( !equal( &parametersLhs[ i ], &parametersRhs[ i ] ) )
        {
            ErrorStream << "Parameter does not match" << std::endl;
            return false;
        }
    }
    const std::vector<Metric*> metricsLhs = lhs->getMetrics();
    const std::vector<Metric*> metricsRhs = rhs->getMetrics();
    if ( metricsLhs.size() != metricsRhs.size() )
    {
        ErrorStream << "#Metrics does not match" << std::endl;
        return false;
    }
    int lengthMetrics = metricsLhs.size();
    for ( int i = 0; i < lengthMetrics; i++ )
    {
        if ( !equal( metricsLhs[ i ], metricsRhs[ i ] ) )
        {
            ErrorStream << "Metric does not match" << std::endl;
            return false;
        }
    }
    const std::vector<Callpath*> callpathsLhs = lhs->getAllCallpaths();
    const std::vector<Callpath*> callpathsRhs = rhs->getAllCallpaths();
    if ( callpathsLhs.size() != callpathsRhs.size() )
    {
        ErrorStream << "#Callpaths does not match" << std::endl;
        return false;
    }
    int lengthCallpaths = callpathsLhs.size();
    for ( int j = 0; j < lengthCallpaths; j++ )
    {
        if ( !equal( callpathsLhs[ j ], callpathsRhs[ j ] ) )
        {
            ErrorStream << "Callpath does not match" << std::endl;
            return false;
        }
    }
    for ( int i = 0; i < lengthMetrics; i++ )
    {
        for ( int j = 0; j < lengthCallpaths; j++ )
        {
            ExperimentPointList pointsLhs = lhs->getPoints( *metricsLhs[ i ], *callpathsLhs[ j ] );
            ExperimentPointList pointsRhs = rhs->getPoints( *metricsRhs[ i ], *callpathsRhs[ j ] );
            if ( pointsLhs.size() != pointsRhs.size() )
            {
                ErrorStream << "ExperimentPointList size does not match: Left: " << pointsLhs.size() << " Right: " << pointsRhs.size() << std::endl;
                return false;
            }
            int lengthPoints = pointsLhs.size();
            for ( int k = 0; k < lengthPoints; k++ )
            {
                if ( !equal( pointsLhs[ k ], pointsRhs[ k ] ) )
                {
                    ErrorStream << "ExperimentPoint Not Ok" << std::endl;
                    return false;
                }
            }
            ModelList modelsLhs = lhs->getModels( *metricsLhs[ i ], *callpathsLhs[ j ] );
            ModelList modelsRhs = rhs->getModels( *metricsRhs[ i ], *callpathsRhs[ j ] );
            if ( modelsLhs.size() != modelsRhs.size() )
            {
                return false;
            }
            int lengthModels = modelsLhs.size();
            for ( int l = 0; l < lengthModels; l++ )
            {
                if ( !equal( modelsLhs[ l ], modelsRhs[ l ] ) )
                {
                    ErrorStream << "Model not ok" << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}
bool
Experiment::validate_experiment()
{
    const ParameterList params = this->getParameters();
    if ( params.size() > 3 )
    {
        ErrorStream << "Experiment has more than 3 Parameters which is not supported" << std::endl;
        return false;
    }
    int numValues = 1;
    for ( ParameterList::const_iterator param = params.begin(); param != params.end(); param++ )
    {
        numValues *= this->getValuesForParameter( *param ).size();
    }
    CallpathList allCallpaths = this->getAllCallpaths();
    MetricList   allMetrics   = this->getMetrics();
    for ( CallpathList::const_iterator cp = allCallpaths.begin(); cp != allCallpaths.end(); cp++ )
    {
        for ( MetricList::const_iterator metric = allMetrics.begin(); metric != allMetrics.end(); metric++ )
        {
            int numExperimentpoints = this->getPoints( **metric, **cp ).size();
            if ( numExperimentpoints != ( numValues ) )
            {
                ErrorStream << "Incomplete Number of ExperimentPoints for Callpath " << ( *cp )->getFullName() << " and Metric " << ( *metric )->getName() << std::endl;
                ErrorStream << "Is: " << numExperimentpoints << " but should be: " <<  numValues << std::endl;
                return false;
            }
        }
    }
    return true;
}
};