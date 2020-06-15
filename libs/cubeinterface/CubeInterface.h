#ifndef CUBEINTERFACE_H
#define CUBEINTERFACE_H

#include <Cube.h>
#include <CubeMetric.h>
#include <CubeCartesian.h>
#include <CubeScaleFuncValue.h>
#include <CubeServices.h>

#include "Parameter.h"
#include "Coordinate.h"
#include "IncrementalPoint.h"

struct FileName
{
    std::string path;
    double      parameter1;
    double      parameter2;
    double      parameter3;
    int         repetition;
};

class CubeInterface
{

private:

    int m_scaling_type;
    std::string m_dir_name;
    std::string m_prefix;
    std::string m_postfix;
    std::string m_cube_file_name;
    int m_repetitions;
    int m_num_params;
    std::string m_displayed_names;
    std::string m_names;
    std::string m_parameter_value_string;
    std::vector<EXTRAP::Parameter> m_parameters;
    std::vector<std::string> m_parameter_prefixes;
    std::vector<std::vector<double> > m_parameter_values;
    EXTRAP::CoordinateList m_parameter_value_lists;
    std::vector<FileName> m_file_name_objects;
    std::vector<std::string> m_cube_files;
    EXTRAP::Experiment* m_experiment;
    cube::Cube* m_cube;

public:

    CubeInterface();

    ~CubeInterface();

    void prepareCubeFileReader(bool debug, int scaling_type, std::string dir_name, std::string prefix, std::string postfix, std::string cube_file_name, int repetitions);
    
    void addParameters(bool debug, int num_params, std::string displayed_names, std::string names, std::string parameter_values);

    void getFileNames(bool debug);

    void getCoordinates(bool debug);

    int getNumParameters(void);

    std::vector<EXTRAP::Parameter> getParameters(void);

    EXTRAP::CoordinateList getParameterValueLists(void);

    void generateCubeFiles(void);

    void createExperimentParameters(void);

    void createExperimentCoordinates(void);

    void addCubeData(bool debug, std::vector<std::vector<EXTRAP::IncrementalPoint>> measurements);

    void initializeMeasurements(std::vector<std::vector<EXTRAP::IncrementalPoint>>& measurements, int numMetrics, int numCallpaths);

    void readValueFromCube(bool debug, EXTRAP::IncrementalPoint* experimentPoint, cube::Cube* cube, cube::Metric* metric, cube::Cnode* cnode, int scaling_type);

    void deleteCube(void);

    EXTRAP::Experiment* getExperiment(void);

};

#endif