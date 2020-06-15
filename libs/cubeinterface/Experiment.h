#ifndef EXPERIMENT_H
#define EXPERIMENT_H

/**
 * @file
 * This file contains the definition of the EXTRAP::Experiment class.
 */

#include "Coordinate.h"
#include "Metric.h"
#include "Callpath.h"
#include "ModelGenerator.h"
#include "ExperimentPoint.h"
#include <algorithm>
#include <set>
#include <fstream>
#include "IoHelper.h"
#include <stdint.h>

/**
 * Trick the SWIG wrapper generator.
 * If the return value is a const class reference, we can not access
 * methods of the class in python. Thus, we make SWIG think the return
 * values not const via defining an empty CONST. In all other cases
 * we want to have the returned reference of to be const to ensure
 * side-affect free programming.
 */
#ifndef CONST
#define CONST const
#endif

namespace EXTRAP
{
static const std::string EXTRAP_CLASSIFIER = "EXTRAP_EXPERIMENT";
static const std::string VERSION_QUALIFIER = "1.0";

/**
 * This class represents an Extra-P Experiment. It contains the data points, models
 * and all other data associated with an experiment.
 */
class Experiment
{
public:
    friend bool
    equal( const Experiment* lhs,
           const Experiment* rhs );

    /**
     * Creates an empty new Experiment object.
     */
    Experiment( void );

    /**
     * Deletes a whole experiment with all associated data.
     */
    virtual
    ~Experiment( void );

    /**
     * Returns the list of all metric definitions.
     */
    virtual CONST MetricList&
    getMetrics( void ) const;

    /**
     * Returns the list od all region definitions.
     */
    virtual CONST RegionList&
    getRegions( void ) const;

    /**
     * Returns a list of all root callpaths.
     */
    virtual CONST CallpathList&
    getRootCallpaths( void ) const;

    /**
     * Returns a list of all callpath definitions.
     */
    virtual CONST CallpathList&
    getAllCallpaths( void ) const;

    /**
     * Returns a list of all parameter definitions.
     */
    virtual CONST ParameterList&
    getParameters( void ) const;

    /**
     * Returns a list of all coordinate definitions.
     */
    virtual CONST CoordinateList&
    getCoordinates( void ) const;

    /**
     * Returns a list of all measuring points for the given parameter.
     */
    virtual CONST std::set<Value>
    getValuesForParameter( const Parameter& parameter ) const;

    /**
     * Return the models for a given @a metric and @a callpath.
     * @param metric     The metric definition for which you want to get
     *                   the models. The metric must be known to the
     *                   Experiment.
     * @param callpath   The callpath definition for which you want to
     *                   get the models. The callpath must be known to the
     *                   Experiment.
     */
    virtual CONST ModelList&
    getModels( const Metric&   metric,
               const Callpath& callpath ) const;

    /**
     * Return the points for a given @a metric and @a callpath.
     * @param metric     The metric definition for which you want to get
     *                   the points. The metric must be known to the
     *                   Experiment.
     * @param callpath   The callpath definition for which you want to
     *                   get the points. The callpath must be known to
     *                   the Experiment.
     */
    virtual CONST ExperimentPointList&
    getPoints( const Metric&   metric,
               const Callpath& callpath ) const;

    /**
     * Returns a list with all model generators.
     */
    virtual CONST ModelGeneratorList&
    getModelGenerators( void ) const;

    /**
     * Returns a list with all model comments.
     */
    virtual CONST ModelCommentList&
    getModelComments( void ) const;

    virtual Parameter
    getParameter( int id ) const;

    virtual Metric*
    getMetric( int id ) const;

    virtual Region*
    getRegion( int id ) const;

    virtual Callpath*
    getCallpath( int id ) const;

    virtual const Coordinate*
    getCoordinate( int id ) const;

    const Coordinate*
    /**
     * Gets a coordinate object by the indices for the different parameters.
     * Returns null if no such Coordinate could be found or the number of indices does not match the number of parameters.
     */
    getCoordinateByIndices( std::vector<Value> indices ) const;

    virtual ModelGenerator*
    getModelGenerator( int id ) const;

    virtual ModelComment*
    getModelComment( int id ) const;

    /**
     * Adds a parameter definition to the Experiment.
     * @param newParameter The new Parameter definition.
     */
    virtual void
    addParameter( Parameter& newParameter );

    /**
     * Adds a coordinate for which measurements exist to the Experiment.
     * @param newCoordinate The coordinate that is added to the
     *                      Experiment. The Experiment will take
     *                      ownership of the coordinate and delete
     *                      it when the Experiment is deleted.
     */
    virtual void
    addCoordinate( Coordinate* newCoordinate );

    /**
     * Adds a data point to the Experiment. The experiment will
     * take ownership of the data point.
     * @param newDataPoint The data point that is added to the
     *                     Experiment. The Experiment will take
     *                     ownership of the data point and delete
     *                     it when the Experiment is deleted.
     * @param metric       The metric for which the data point is
     *                     added.
     * @param callpath     The callpath for which the data point is
     *                     added.
     */
    virtual void
    addDataPoint( ExperimentPoint* newDataPoint,
                  const Metric&    metric,
                  const Callpath&  Callpath );

    /**
     * Adds a model to the Experiment. The experiment will
     * take ownership of the model.
     * @param  newModel  The model that is added to the
     *                   Experiment. The Experiment will take
     *                   ownership of the model and delete
     *                   it when the Experiment is deleted.
     * @param metric     The metric for which the model is
     *                   added.
     * @param callpath   The callpath for which the model is
     *                   added.
     */
    virtual void
    addModel( Model*          newModel,
              const Metric&   metric,
              const Callpath& callpath );

    /**
     * Adds a metric definitions to the Experiment. The experiment will
     * take ownership of the metric definition.
     * @param  newModel  The metric definition that is added to the
     *                   Experiment. The Experiment will take
     *                   ownership of the metric definition and delete
     *                   it when the Experiment is deleted.
     */
    virtual void
    addMetric( Metric* newMetric );

    /**
     * Adds a region definitions to the Experiment. The experiment will
     * take ownership of the region definition.
     * @param  newMetric  The region definition that is added to the
     *                    Experiment. The Experiment will take
     *                    ownership of the region definition and delete
     *                    it when the Experiment is deleted.
     */
    virtual void
    addRegion( Region* newRegion );

    /**
     * Adds a callpath definitions to the Experiment. The experiment will
     * take ownership of the callpath definition.
     * @param  newCallpath  The callpath definition that is added to the
     *                      Experiment. The Experiment will take
     *                      ownership of the callpath definition and delete
     *                      it when the Experiment is deleted.
     */
    virtual void
    addCallpath( Callpath* newCallpath );

    /**
     * Adds a model generator configuration to the Experiment. The experiment will
     * take ownership of the model generator.
     * @param  newGenerator  The model generator that is added to the
     *                       Experiment. The Experiment will take
     *                       ownership of the model generator and delete
     *                       it when the Experiment is deleted.
     */
    virtual void
    addModelGenerator( ModelGenerator* newGenerator );

    /**
     * Adds a model comment definiton the Experiment. The experiment will
     * take ownership of the model comment. It does not check for duplicates.
     * @param  newComment  The model comment definition that is added to the
     *                     Experiment. The Experiment will take
     *                     ownership of the model comment and delete
     *                     it when the Experiment is deleted.
     */
    virtual void
    addModelComment( ModelComment* newComment );

    /**
     * Adds a model comment to the experiment. It checks for duplicates.
     * If a comment with a matching message exists, it returns the pointer
     * to the existing model comment. Otherwise a new model comment object
     * is created and added to the experiment. The experiment holds the ownership
     * of the new model.
     * @param message The message that the new model displays.
     */
    virtual ModelComment*
    addModelComment( const std::string& message );

    /**
     * Uses the given model generator to generate new models for all
     * metrics and callpaths.
     * @param generator  The model generator used to generate the models
     */
    void
    modelAll( ModelGenerator&       generator,
              Experiment*           experiment,
              ModelGeneratorOptions options );

    /**
     * Deletes the model with the index modelIndex for the given
     * callpath/metric pair.
     * @param modelIndex The index of the model to delete
     * @param metric     The metric definition for which you want to delete
     *                   the models. The metric must be known to the
     *                   Experiment.
     * @param callpath   The callpath definition for which you want to
     *                   delete the models. The callpath must be known to the
     *                   Experiment.
     */
    void
    deleteModel( int             modelIndex,
                 const Metric&   metric,
                 const Callpath& callpath );

    /**
     * Deletes the model with the index modelIndex for all metrics and callpaths
     * @param modelIndex The index of the model to delete
     */
    void
    deleteModel( int modelIndex );

    /**
     * Opens an Extra-P experiment file and returns the Experiment in this file.
     * @param filename  The file name of the Extra-P experiment file.
     */
    static Experiment*
    openExtrapFile( const std::string& filename );

    /**
     * Opens an text input file in JSON format and returns an Experiment with this data.
     * @param filename  The file name of the text file.
     */
    static Experiment*
    openJsonInput( const std::string& filename );

    /**
     * Opens an text input file and returns an Experiment with this data.
     * @param filename  The file name of the text file.
     */
    static Experiment*
    openTextInput( const std::string& filename );

    /**
     * Creates a single or multi parameter modeler for an Experiment.
     * @param experiment  The experiment for which a modeler needs to be created.
     */
    static Experiment*
    createModelGenerator( Experiment* experiment );

    /**
     * Opens a hdf5 input file and returns an Experiment with its data.
     * @param filename  The file name of the text file.
     */
    static Experiment*
    openHDF5File( const std::string& filename );

    /**
     * Writes the data and the models to in the Extra-P file format
     * @param filename The name of the file.
     */
    bool
    writeExtrapFile( const std::string& filename ) const;

    /**
     * Writes the experiment data to a stream.
     */
    bool
    serialize( std::ofstream& stream ) const;

    /**
     * Creates an Experiment object from a stream.
     */
    static Experiment*
    deserialize( std::ifstream& stream );

protected:
    /**
     * The list of metric definitions for this experiment.
     */
    MetricList m_metrics;

    /**
     * The list of region definitions for this experiment.
     */
    RegionList m_regions;

    /**
     * The list of all callpaths in this experiment.
     */
    CallpathList m_root_callpaths;

    /**
     * The list of root callpaths in this experiment.
     */
    CallpathList m_all_callpaths;

    /**
     * The list of parameter definitions for this experiment.
     */
    ParameterList m_parameters;

    /**
     * The list of coordinates for which data points exist.
     */
    CoordinateList m_coordinates;

    /* Outer vector is the metric dimension.
     * Inner vector is the callpath dimension */
    typedef std::vector<std::vector<ModelList> >           ModelMatrix;
    typedef std::vector<std::vector<ExperimentPointList> > ExperimentPointMatrix;

    /**
     * The models for this experiment.
     */
    ModelMatrix m_models;

    /**
     * The measured points for this experiment.
     */
    ExperimentPointMatrix m_points;

    /**
     * Empty list to be returned if no models are available.
     */
    ModelList m_empty_model_list;

    /**
     * The list of model generators.
     */
    ModelGeneratorList m_generators;

    /**
     * The list of model comments.
     */
    ModelCommentList m_comments;

private:
    /**
     * Make copy constructor unusable to prevent someone
     * makeing a shallow copy.
     */
    Experiment( const Experiment& );

    /**
     * This validates the experiment regarding completeness of DataPoints and number of parameters.
     **/
    bool
    validate_experiment();

    /**
     * Make assignment operator unusable to prevent someoneFmodel
     * makeing a shallow copy.
     */
    Experiment&
    operator=( const Experiment& );

    std::map<ModelComment*, int> model_comment_reference_counter;
};

bool
equal( const Experiment* lhs,
       const Experiment* rhs );
};

#endif