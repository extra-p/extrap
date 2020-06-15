#ifndef MULTI_PARAMETER_SPARSE_MODELER_H
#define MULTI_PARAMETER_SPARSE_MODELER_H

#include "ModelGenerator.h"
#include "MultiParameterFunctionModeler.h"
#include "CompoundTerm.h"
#include "MultiParameterFunction.h"
#include "Utilities.h"
#include "IoHelper.h"
#include "Experiment.h"
#include "SingleParameterSimpleModelGenerator.h"
#include "IncrementalPoint.h"
#include "MultiParameterHypothesis.h"
#include "Fraction.h"
#include <iostream>
#include <cassert>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <map>
#include <stdio.h>
#include <stdio.h>
#include <string.h>
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
class MultiParameterSparseFunctionModeler : public MultiParameterFunctionModeler
{
public:
    virtual MultiParameterHypothesis
    createModel( const Experiment*             experiment,
                 const ModelGeneratorOptions&  options,
                 const std::vector<DataPoint>& modeledDataPointList,
                 ModelCommentList&             comments,
                 const Function*               expectationFunction = NULL );

protected:

    //Start of external state
    // (nothing here)
    //End of external state

private:
    std::vector<CoordinateList>
    findFirstMeasurementPoints( const Experiment* parent_experiment,
                                int               min_measurement_points );

    std::vector<CoordinateList>
    findMaxMeasurementPoints( const Experiment* parent_experiment,
                              int               min_measurement_points );

    std::vector<CoordinateList>
    findExpensiveMeasurementPoints( const Experiment*             parent_experiment,
                                    int                           min_measurement_points,
                                    const std::vector<DataPoint>& modeledDataPointList );

    std::vector<CoordinateList>
    findCheapMeasurementPoints( const Experiment*             parent_experiment,
                                    int                           min_measurement_points,
                                    const std::vector<DataPoint>& modeledDataPointList );

    std::vector<double>
    getParameterValues( ParameterList parameters,
                        Coordinate*   reference_coordinate,
                        int           parameter_id );

    bool
    compareParameterValues( std::vector<double> parameter_value_list1,
                            std::vector<double> parameter_value_list2 );

    bool
    analyzeMeasurementPoint( ParameterList       parameters,
                             std::vector<double> parameter_value_list );

    std::vector<Experiment*>
    createAllExperiments( std::vector<CoordinateList>   coordinate_container_list,
                          const Experiment*             parent_experiment,
                          const std::vector<DataPoint>& modeledDataPointList );

    Experiment*
    createSingleParameterExperiment( const Experiment*             original_experiment,
                                     CoordinateList                coordinate_container,
                                     Parameter                     parameter,
                                     Metric*                       metric,
                                     Region*                       region,
                                     Callpath*                     callpath,
                                     const std::vector<DataPoint>& modeledDataPointList );

    std::vector<Experiment*>
    modelAllExperiments( std::vector<Experiment*> experiments,
                         ModelGeneratorOptions    options );

    Experiment*
    addSingleParameterModeler( Experiment*           exp,
                               ModelGeneratorOptions options );

    std::vector<SingleParameterFunction*>
    getAllExperimentFunctions( std::vector<Experiment*> experiments );

    MultiParameterHypothesis
    findBestMultiParameterHypothesis( const std::vector<DataPoint>&         modeledDataPointList,
                                      const Experiment*                     parent_experiment,
                                      std::vector<SingleParameterFunction*> functions );

    CoordinateList
    getBaseCoordinates( std::vector<CoordinateList> coordinate_container_list );

    std::vector<DataPoint>
    getBaseDataPoints( CoordinateList                base_cord_list,
                       const Experiment*             experiment,
                       const std::vector<DataPoint>& modeledDataPointList );

    std::vector<DataPoint>
    getAdditionalDataPoints( CoordinateList                base_cord_list,
                             const Experiment*             experiment,
                             const std::vector<DataPoint>& modeledDataPointList );

    std::vector<DataPoint>
    sortByCostDec(             const Experiment*             experiment,
                               const std::vector<DataPoint>& modeledDataPointList );

    std::vector<DataPoint>
    sortByCostInc(             const Experiment*             experiment,
                               const std::vector<DataPoint>& modeledDataPointList );

    std::vector<DataPoint>
    addAdditionalDataPoints( std::vector<DataPoint>&       data_points,
                             const std::vector<DataPoint>& modeledDataPointList,
                             int                           number_add_data_points );
};
};

#endif