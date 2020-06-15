
#include <string>
#include <iostream>
#include <dirent.h>

#include "CubeInterface.h"

#include <Cube.h>
#include <CubeMetric.h>
#include <CubeCartesian.h>
#include <CubeScaleFuncValue.h>
#include <CubeServices.h>

#include "CubeMapping.h"
#include "Parameter.h"
#include "IncrementalPoint.h"
#include "Coordinate.h"
#include "Experiment.h"
#include "Metric.h"
#include "Callpath.h"
#include "DataPoint.h"

CubeInterface::CubeInterface()
{
    this->m_experiment = new EXTRAP::Experiment();
    this->m_cube = new cube::Cube();
}

CubeInterface::~CubeInterface()
{
}

void CubeInterface::prepareCubeFileReader(bool debug, int scaling_type, std::string dir_name, std::string prefix,
    std::string postfix, std::string cube_file_name, int repetitions)
{
    m_scaling_type = scaling_type;
    std::string tmp1(dir_name);
    std::string tmp2(prefix);
    std::string tmp3(postfix);
    std::string tmp4(cube_file_name);
    m_dir_name = tmp1;
    m_prefix = tmp2;
    m_postfix = tmp3;
    m_cube_file_name = tmp4;
    m_repetitions = repetitions;

    if(debug==true)
    {
        std::cout << "Scaling type: " << m_scaling_type << std::endl;
        std::cout << "Directory name: " << m_dir_name << std::endl;
        std::cout << "Prefix: " << m_prefix << std::endl;
        std::cout << "Postfix: " << m_postfix << std::endl;
        std::cout << "Cube file name: " << m_cube_file_name << std::endl;
        std::cout << "Repetitions: " << m_repetitions << std::endl;
    }
}

void CubeInterface::addParameters(bool debug, int num_params, std::string displayed_names, 
    std::string names, std::string parameter_values)
{
    int m_num_params = num_params;
    std::string tmp1(displayed_names);
    std::string tmp2(names);
    std::string tmp3(parameter_values);
    m_displayed_names = tmp1;
    m_names = tmp2;
    m_parameter_value_string = tmp3;

    if(debug==true)
    {
        std::cout << "Number of parameters: " << m_num_params << std::endl;
        std::cout << "Displayed parameters names: " << m_displayed_names << std::endl;
        std::cout << "Parameters names: " << m_names << std::endl;
        std::cout << "Parameter value string: " << m_parameter_value_string << std::endl;
    }

    if(m_num_params==1)
    {
        EXTRAP::Parameter p = EXTRAP::Parameter(m_displayed_names);
        m_parameters.push_back(p);

        m_parameter_prefixes.push_back(m_names);

        std::stringstream ss(m_parameter_value_string);
        std::vector<double> values;

        while( ss.good() )
        {
            std::string substr;
            getline( ss, substr, ',' );
            double value = std::stod(substr);
            values.push_back( value );
        }

        m_parameter_values.push_back(values);
    
    }
    else
    {
        std::stringstream ss(m_displayed_names);
        std::vector<std::string> parameter_names;

        while( ss.good() )
        {
            std::string substr;
            getline( ss, substr, ',' );
            parameter_names.push_back( substr );
        }

        std::vector<std::string> parameter_prefixes;
        std::stringstream ss2(m_names);

        while( ss2.good() )
        {
            std::string substr;
            getline( ss2, substr, ',' );
            parameter_prefixes.push_back( substr );
        }

        std::vector<std::string> parameter_value_strings;
        std::stringstream ss3(m_parameter_value_string);

        while( ss3.good() )
        {
            std::string substr;
            getline( ss3, substr, ';' );
            parameter_value_strings.push_back( substr );
        }

        for (int i = 0; i < m_num_params; i++)
        {
            EXTRAP::Parameter p = EXTRAP::Parameter(parameter_names[i]);
            m_parameters.push_back(p);

            m_parameter_prefixes.push_back(parameter_prefixes[i]);
        
            std::stringstream ss4(parameter_value_strings[i]);
            std::vector<double> parameter_values;

            while( ss4.good() )
            {
                std::string substr;
                getline( ss4, substr, ',' );
                double value = std::stod(substr);
                parameter_values.push_back( value );
            }

            m_parameter_values.push_back(parameter_values);
        }
    }
}

struct comparer
{
    inline bool
    operator()( const FileName& one, const FileName& two )
    {
        return one.parameter1 < two.parameter1 || ( one.parameter1 == two.parameter1 && one.parameter2 < two.parameter2 ) || ( one.parameter1 == two.parameter1 && one.parameter2 == two.parameter2 && one.parameter3 < two.parameter3 ) || ( one.parameter1 == two.parameter1 && one.parameter2 == two.parameter2 && one.parameter3 == two.parameter3 && one.repetition < two.repetition );
    }
};

void CubeInterface::getFileNames(bool debug)
{
    std::vector<std::string> fileNames;
    std::string              postfix = "/profile.cubex";
    std::vector<std::string> folders;
    std::string              path  = this->m_dir_name;
    const char*              PATH  = path.c_str();
    DIR*                     dir   = opendir( PATH );
    struct dirent*           entry = readdir( dir );

    while ( entry != NULL )
    {
        if ( entry->d_type == DT_DIR )
        {
            folders.push_back( entry->d_name );
        }
        entry = readdir( dir );
    }

    closedir( dir );

    for ( int i = 0; i < folders.size(); i++ )
    {
        if ( folders.at( i ) == "." )
        {
            folders.erase( folders.begin() + i );
        }
        if ( folders.at( i ) == ".." )
        {
            folders.erase( folders.begin() + i );
        }
    }

    //construct full file path
    for ( int i = 0; i < folders.size(); i++ )
    {
        std::string file_path = path + "/" + folders.at( i ) + postfix;
        fileNames.push_back( file_path );
    }

    //populate the file name objects with the data
    std::vector<FileName> file_names( fileNames.size() );
    for ( int i = 0; i < fileNames.size(); i++ )
    {
        std::string postfix = "/profile.cubex";
        std::string dir     = fileNames.at( i );

        //remove prefix and postfix
        dir.erase( 0, this->m_dir_name.length() );
        int difference = dir.length() - postfix.length();
        dir.erase( difference, dir.length() );

        //remove prefix name
        int pos = dir.find( "." );
        dir.erase( 0, pos + 1 );

        //save repetition
        char dir_rep = dir.at( dir.length() - 1 );
        int  rep     =  dir_rep - 48;
        dir.erase( dir.length() - 3, dir.length() );

        std::vector<std::string> pairs;
        std::stringstream        ss( dir );
        while ( ss.good() )
        {
            std::string sub;
            getline( ss, sub, '.' );
            pairs.push_back( sub );
        }

        std::vector<double> parameter_values;
        for ( int j = 0; j < pairs.size(); j++ )
        {
            EXTRAP::Parameter   ptmp      = this->m_parameters.at( j );
            std::string ptmp_name = ptmp.getName();
            std::string value     = pairs.at( j );
            value.erase( 0, ptmp_name.length() );
            int pos = value.find(",");
            if(pos!=-1){
            	value = value.replace(pos, 1, ".");
            }
            char buffer[ value.length() + 1 ];
            std::strcpy( buffer, value.c_str() );
            double x = atof( buffer );
            parameter_values.push_back( x );
        }

        file_names[ i ].path       = fileNames.at( i );
        file_names[ i ].parameter1 = parameter_values[ 0 ];
        file_names[ i ].parameter2 = parameter_values[ 1 ];
        file_names[ i ].parameter3 = parameter_values[ 2 ];
        file_names[ i ].repetition = rep;
    }

    //sort the vector of file names and parameter values
    sort( file_names.begin(), file_names.end(), comparer() );

    if(debug==true)
    {
        for ( int i = 0; i < file_names.size(); i++ )
        {
            std::cout << file_names.at( i ).path << ", " << file_names.at( i ).parameter1 << ", " << file_names.at( i ).parameter2 << ", " << file_names.at( i ).parameter3 << ", " << file_names.at( i ).repetition << "\n";
        }
    }

    this->m_file_name_objects = file_names;
}

void CubeInterface::getCoordinates(bool debug)
{
    //clear coordinate list
    this->m_parameter_value_lists.clear();

    //create the coordinates
    for ( int i = 0; i < this->m_file_name_objects.size(); i += this->m_repetitions )
    {
        std::vector<double> values;
        values.push_back( this->m_file_name_objects[ i ].parameter1 );
        values.push_back( this->m_file_name_objects[ i ].parameter2 );
        values.push_back( this->m_file_name_objects[ i ].parameter3 );

        EXTRAP::Coordinate* cord = new EXTRAP::Coordinate();
        for ( int j = 0; j < this->m_parameters.size(); j++ )
        {
            cord->insert( std::pair<EXTRAP::Parameter, EXTRAP::Value>( this->m_parameters.at( j ), values.at( j ) ) );
        }
        this->m_parameter_value_lists.push_back( cord );
    }

    //debug print out for the coordinates
    if(debug==true)
    {
        std::cout << "Number coordinates: " << this->m_parameter_value_lists.size() << std::endl;
        for ( int i = 0; i < this->m_parameter_value_lists.size(); i++ )
        {
            std::cout << this->m_parameter_value_lists.at( i )->toString() << std::endl;
        }
    }
}

int CubeInterface::getNumParameters(void)
{
    return this->m_num_params;
}

std::vector<EXTRAP::Parameter> CubeInterface::getParameters(void)
{
    return this->m_parameters;
}

EXTRAP::CoordinateList CubeInterface::getParameterValueLists(void)
{
    return this->m_parameter_value_lists;
}

void CubeInterface::generateCubeFiles(void)
{
    for ( int i = 0; i < this->m_file_name_objects.size(); i++ )
    {
        this->m_cube_files.push_back( this->m_file_name_objects.at( i ).path );
    }
}

void CubeInterface::createExperimentParameters(void)
{
    for ( int i = 0; i < this->m_parameters.size(); i++ )
    {
        this->m_experiment->addParameter( this->m_parameters[ i ] );
    }
}

void CubeInterface::createExperimentCoordinates(void)
{
    for ( EXTRAP::CoordinateList::iterator it = this->m_parameter_value_lists.begin();
        it != this->m_parameter_value_lists.end();
        it++ )
    {
        this->m_experiment->addCoordinate( *it );
    }
}

void CubeInterface::initializeMeasurements(std::vector<std::vector<EXTRAP::IncrementalPoint>>& measurements, int numMetrics, int numCallpaths)
{
    for ( int metric = 0; metric < numMetrics; metric++ )
    {
        measurements.push_back( std::vector<EXTRAP::IncrementalPoint>() );
        for ( int cp = 0; cp < numCallpaths; cp++ )
        {
            measurements[ metric ].push_back( EXTRAP::IncrementalPoint() );
        }
    }
}

void CubeInterface::readValueFromCube(bool debug, EXTRAP::IncrementalPoint* experimentPoint, cube::Cube* cube, cube::Metric* metric, cube::Cnode* cnode, int scaling_type)
{
    if ( metric == NULL || cnode == NULL )
    {
        return;
    }

    const std::vector<cube::Thread*> threads = cube->get_thrdv();

    double accumulated_value = 0;
    for ( std::vector<cube::Thread*>::const_iterator thread = threads.begin();
          thread != threads.end();
          thread++ )
    {
        double thread_value = cube->get_sev( metric, cnode, *thread );
        if ( scaling_type == 1 )
        {
            if(debug==true)
            {
                std::cout << "Value: " << thread_value << std::endl;
            }

            experimentPoint->addValue( thread_value );
        }
        else
        {
            accumulated_value += thread_value;
        }
    }
    if ( scaling_type != 1 )
    {
        if(debug==true)
        {
            std::cout << "Value: " << accumulated_value << std::endl;
        }
        experimentPoint->addValue( accumulated_value );
    }
}

void CubeInterface::addCubeData(bool debug, std::vector<std::vector<EXTRAP::IncrementalPoint>> measurements)
{
    const EXTRAP::MetricList& m_metrics = this->m_experiment->getMetrics();
    const EXTRAP::CallpathList& m_callpaths = this->m_experiment->getAllCallpaths();

    for ( int i = 0; i < this->m_cube_files.size(); i += this->m_repetitions )
    {
        EXTRAP::Coordinate* coordinates = this->m_parameter_value_lists[ i / this->m_repetitions ];
        
        for ( int j = 0; j < this->m_repetitions; j++ )
        {
            // Read Cube file and create mapping
            this->m_cube->openCubeReport( this->m_cube_files[ i + j ] );
            EXTRAP::CubeMapping mapping( this->m_experiment, this->m_cube );

            if ( i == 0 && j == 0 )
            {
                mapping.createMetricDef();
                mapping.createRegionDef();
                mapping.createCallpathDef();
                this->initializeMeasurements( measurements,
                                        this->m_experiment->getMetrics().size(),
                                        this->m_experiment->getAllCallpaths().size() );
            }
            else
            {
                mapping.createMetricMapping();
                mapping.createRegionMapping();
                mapping.createCallpathMapping();
            }
    
            // Iterate over all metrics and callpaths and add data
            for ( EXTRAP::MetricList::const_iterator metric_it = m_metrics.begin();
                    metric_it != m_metrics.end();
                    metric_it++ )
            {
                cube::Metric* metric = mapping.getCubeMetric( *metric_it );
                
                for ( EXTRAP::CallpathList::const_iterator callpath_it = m_callpaths.begin();
                        callpath_it != m_callpaths.end();
                        callpath_it++ )
                {
                    EXTRAP::IncrementalPoint* point = &measurements[ ( *metric_it )->getId() ]
                                                [ ( *callpath_it )->getId() ];
                    cube::Cnode* cnode = mapping.getCubeCallpath( *callpath_it );
                    this->readValueFromCube(debug, point, this->m_cube, metric, cnode, this->m_scaling_type );
                }
            }

            if(debug==true)
            {
                std::cout << "making progess reading cube files..." << std::endl;
            }
        }

        // Iterate over all metrics and callpaths and create the experiment point
        for ( EXTRAP::MetricList::const_iterator metric_it = m_metrics.begin();
                metric_it != m_metrics.end();
                metric_it++ )
        {
            int metric_id = ( *metric_it )->getId();
            
            for ( EXTRAP::CallpathList::const_iterator callpath_it = m_callpaths.begin();
                    callpath_it != m_callpaths.end();
                    callpath_it++ )
            {
                int callpath_id = ( *callpath_it )->getId();
                EXTRAP::IncrementalPoint& ip = measurements[ metric_id ][ callpath_id ];
                EXTRAP::ExperimentPoint* point = ip.getExperimentPoint( coordinates,
                                                                        *metric_it,
                                                                        *callpath_it );

                this->m_experiment->addDataPoint( point,
                                            **metric_it,
                                            **callpath_it );
                measurements[ metric_id ][ callpath_id ].clear();
            }
        }
    }
}

EXTRAP::Experiment* CubeInterface::getExperiment(void)
{
    return this->m_experiment;
}

void CubeInterface::deleteCube(void)
{
    delete this->m_cube;
}

struct Data
{
    int num_metrics;
    int num_callpaths;
    int num_parameters;
    int num_coordinates;
    int mean;
    EXTRAP::ParameterList parameters;
    EXTRAP::CallpathList callpaths;
    EXTRAP::MetricList metrics;
    EXTRAP::CoordinateList coordinates;
    EXTRAP::Experiment* experiment;
};

extern "C"
{
    Data* exposed_function(int scaling_type, char* dir_name, char* prefix,
        char* postfix, char* cube_file_name, int repetitions, int num_params,
        char* displayed_names, char* names, char* parameter_values, int use_mean)
    {
        
        CubeInterface* test = new CubeInterface();

        test->prepareCubeFileReader(false, scaling_type, dir_name, prefix, postfix, cube_file_name, repetitions);

        test->addParameters(false, num_params, displayed_names, names, parameter_values);

        try
        {   
            std::vector<std::vector<EXTRAP::IncrementalPoint>> measurements;
            
            test->getFileNames(false);

            test->generateCubeFiles();

            test->getCoordinates(false);

            test->createExperimentParameters();

            test->createExperimentCoordinates();

            test->addCubeData(false, measurements);

            test->deleteCube();
        }
        catch( ... )
        {
            std::cout << "Error occured while reading cube files." << std::endl;
            return NULL;
        }
        
        // creating data object for transfer to python
        EXTRAP::Experiment* experiment = test->getExperiment();
        Data* data = new Data();

        bool debug = false;

        // parameters
        std::vector<EXTRAP::Parameter> parameters = experiment->getParameters();
        data->num_parameters = parameters.size();
        data->parameters = experiment->getParameters();
        if(debug==true)
        {
            std::cout << "Number of parameters: " << parameters.size() << std::endl;
            for (int i = 0; i < parameters.size(); i++)
            {
                EXTRAP::Parameter parameter = parameters[i];
                std::cout << "Parameter " << i+1 << ": " << parameter.getName() << std::endl;
            }
        }

        // coordinates
        EXTRAP::CoordinateList coordinates = experiment->getCoordinates();
        data->num_coordinates = coordinates.size();
        data->coordinates = experiment->getCoordinates();
        if(debug==true)
        {
            std::cout << "Number of coordinates: " << coordinates.size() << std::endl;
            for (int i = 0; i < coordinates.size(); i++)
            {
                EXTRAP::Coordinate* coordinate = coordinates.at(i);
                std::cout << "Coordinate " << i+1 << ": " << coordinate->toString() << std::endl;
            }
        }

        // callpaths
        data->callpaths = experiment->getAllCallpaths();
        data->num_callpaths = data->callpaths.size();

        EXTRAP::CallpathList callpaths = experiment->getAllCallpaths();

        if(debug==true)
        {
            std::cout << "Number of callpaths: " << callpaths.size() << std::endl;
            for (int i = 0; i < callpaths.size(); i++)
            {
                EXTRAP::Callpath* callpath = callpaths[i];
                std::cout << "Callpath " << i+1 << ": " << callpath->getFullName() << std::endl;
            }
        }
        
        // metrics
        data->metrics = experiment->getMetrics();
        data->num_metrics = data->metrics.size();

        EXTRAP::MetricList metrics = experiment->getMetrics();

        if(debug==true)
        {
            std::cout << "Number of metrics: " << metrics.size() << std::endl;
            for (int i = 0; i < metrics.size(); i++)
            {
                EXTRAP::Metric* metric = metrics[i];
                std::cout << "Metric " << i+1 << ": " << metric->getName() << std::endl;
            }
        }

        // measurements
        data->experiment = experiment;
        data->mean = use_mean; // 0 = median, 1 = mean

        //std::cout << "Cube files have been read successfully." << std::endl;
        return data;
    }

    int getNumCoordinates(Data* data)
    {
        return data->num_coordinates;
    }

    int getNumCharsCoordinates(Data* data, int element_id)
    {
        EXTRAP::Coordinate* coordinate = data->coordinates.at(element_id);
        std::string coordinate_string = coordinate->toString();
        int num_chars = coordinate_string.size();
        return num_chars;
    }

    char getCoordinateChar(Data* data, int element_id, int char_id)
    {
        EXTRAP::Coordinate* coordinate = data->coordinates.at(element_id);
        std::string coordinate_string = coordinate->toString();
        char coordinate_char = coordinate_string.at(char_id);
        return coordinate_char;
    }

    int getNumParameters(Data* data)
    {
        return data->num_parameters;
    }

    int getNumCharsParameters(Data* data, int element_id)
    {
        EXTRAP::Parameter parameter = data->parameters.at(element_id);
        std::string parameter_string = parameter.getName();
        int num_chars = parameter_string.size();
        return num_chars;
    }

    char getParameterChar(Data* data, int element_id, int char_id)
    {
        EXTRAP::Parameter parameter = data->parameters.at(element_id);
        std::string parameter_string = parameter.getName();
        char parameter_char = parameter_string.at(char_id);
        return parameter_char;
    }

    int getNumMetrics(Data* data)
    {
        return data->num_metrics;
    }

    int getNumCharsMetrics(Data* data, int element_id)
    {
        EXTRAP::Metric* metric = data->metrics.at(element_id);
        std::string metric_string = metric->getName();
        int num_chars = metric_string.size();
        return num_chars;
    }

    char getMetricChar(Data* data, int element_id, int char_id)
    {
        EXTRAP::Metric* metric = data->metrics.at(element_id);
        std::string metric_string = metric->getName();
        char metric_char = metric_string.at(char_id);
        return metric_char;
    }

    int getNumCallpaths(Data* data)
    {
        return data->num_callpaths;
    }

    int getNumCharsCallpath(Data* data, int element_id)
    {
        EXTRAP::Callpath* callpath = data->callpaths.at(element_id);
        std::string callpath_string = callpath->getFullName();
        int num_chars = callpath_string.size();
        return num_chars;
    }

    char getCallpathChar(Data* data, int element_id, int char_id)
    {
        EXTRAP::Callpath* callpath = data->callpaths.at(element_id);
        std::string callpath_string = callpath->getFullName();
        char callpath_char = callpath_string.at(char_id);
        return callpath_char;
    }

    double getDataPointValue(Data* data, int metric_id, int callpath_id, int data_point_id)
    {
        EXTRAP::Metric* metric = data->metrics.at(metric_id);
        EXTRAP::Callpath* callpath = data->callpaths.at(callpath_id);
        const EXTRAP::ExperimentPointList& data_points = data->experiment->getPoints( *metric, *callpath );
        std::vector<EXTRAP::DataPoint> dataPoints;
        dataPoints.reserve( data_points.size() );
        for (int i = 0; i < data_points.size(); i++)
        {
            EXTRAP::ExperimentPoint* point = data_points.at(i);
            if(data->mean==1)
            {
                dataPoints.push_back( EXTRAP::DataPoint( &( point->getCoordinate() ), point->getSampleCount(), point->getMean(), point->getMeanCI() ) );
            }
            else
            {
                dataPoints.push_back( EXTRAP::DataPoint( &( point->getCoordinate() ), point->getSampleCount(), point->getMedian(), point->getMedianCI() ) );
            }
        }
        double value = dataPoints.at(data_point_id).getValue();
        //debug code
        //EXTRAP::Parameter p1 = data->experiment->getParameters()[0];
        //double p1v = dataPoints.at(data_point_id).getParameterValue( p1 );
        // Notfalls kann man hier noch die Koordinate als String mit geben um zu erkennen wozu dieser Wert gehoert.
        //std::cout << "T(" << p1.getName() << "=" << p1v << ")=" << value << std::endl;
        return value;
    }
    
}
