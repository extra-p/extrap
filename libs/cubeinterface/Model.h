#ifndef MODEL_H
#define MODEL_H

#include "Function.h"
#include "ModelComment.h"
#include "Callpath.h"
#include "Metric.h"
#include "DataPoint.h"

namespace EXTRAP
{
class ModelGenerator;
typedef std::map<const Parameter, IntervalList> ValidIntervalsList;

class Model
{
public:

    static const std::string MODEL_PREFIX;
    /**
     * Creates a new model.
     * @param modelFunction      The model function. The object takes ownership of the
     *                           function.
     * @param generator          A reference to the generator used to generate this model.
     *                           The object does NOT take ownership of the generator.
     * @param confidenceInterval A function pair modeling the confidence interval.
     *                           The object does take ownership of the confidenceInterval
     *                           and its two functions.
     * @param errorCone          A function pair modeling the error cone.
     *                           The object does take ownership of the errorCone
     *                           and its two functions.
     * @param noiseError         A function pair modeling the noise error.
     *                           The object does take ownership of the noiseError
     *                           and its two functions.
     * @param comments           A list of model comments. The object does take ownership of
     *                           all the comments stored inside this list.
     */
    Model( Function*               modelFunction,
           ModelGenerator*         generator,
           FunctionPair*           confidenceInterval,
           FunctionPair*           errorCone,
           FunctionPair*           noiseError,
           const ModelCommentList& comments,
           double                  RSS,
           double                  AR2,
           double                  SMAPE,
           double                  RE,
           std::vector<double> predicted_points,
           std::vector<double> actual_points,
           std::vector<double> ps,
           std::vector<double> sizes );

    /**
     * Deletes the model.
     */
    virtual
    ~Model( void );

    /**
     * Returns the model function.
     */
    virtual Function*
    getModelFunction( void ) const;

    /**
     * Sets a new function for this model. The previously used
     * function will be deleted.
     * @param newFunction The new function, which must not be NULL.
     *                    The model takes ownership of this function.
     */
    virtual void
    setModelFunction( Function* newFunction );

    /**
     * Returns the confidence interval
     */
    virtual FunctionPair*
    getConfidenceInterval( void ) const;

    /**
     * Returns the error cone
     */
    virtual FunctionPair*
    getErrorCone( void ) const;

    /**
     * Returns the noise error
     */
    virtual FunctionPair*
    getNoiseError( void ) const;

    /**
     * Returns a pointer to the generator
     */
    virtual ModelGenerator*
    getGenerator( void ) const;

    double
    getRSS( void ) const;

    double
    getAR2( void ) const;

    double
    getSMAPE( void ) const;

    double
    getRE( void ) const;

    const ModelCommentList&
    getComments( void ) const;

    std::vector<double>
    getPredictedPoints( void );

    std::vector<double>
    getActualPoints( void );

    std::vector<double>
    getPs( void );

    std::vector<double>
    getSizes( void );

    void
    clearComments( void );

    void
    addComment( ModelComment* comment );

    virtual bool
    serialize(
        Metric*   metric,
        Callpath* cp,
        IoHelper* ioHelper ) const;
    static Model*
    deserialize(
        const Experiment* experiment,
        int64_t*          metricId,
        int64_t*          callpathId,
        IoHelper*         ioHelper );

    /**
     * Returns the intervals where the model is defined. Only positive values are possible
     * A negative value represent infinity.
     * @param parameter  The Parameter for which the intervals are valid.
     */
    virtual const IntervalList*
    getValidIntervals( const Parameter& parameter ) const;

    /**
     * Adds a validity interval to the model
     * @param parameter The parameter to which the validity interval belongs.
     * @param interval  The interval.
     */
    virtual void
    addValidInterval( const Parameter& parameter,
                      const Interval&  interval );

protected:
    /**
     * An interval list that holds only one interval which
     * contains the whole possible definition space.
     * Used to return valid intervals in case no specific intervals
     * are given.
     */
    static IntervalList m_whole_interval;

    /**
     * List of valid intervals
     */
    ValidIntervalsList m_valid_intervals;

    /**
     * Stores the model function
     */
    Function* m_model_function;

    /**
     * Stores the confidence interval
     */
    FunctionPair* m_confidence_interval;

    /**
     * Stores the error cone
     */
    FunctionPair* m_error_cone;

    /**
     * Stores the noise errer
     */
    FunctionPair* m_noise_error;

    /**
     * Stores a pointer to the generator.
     */
    ModelGenerator* m_generator;

    double m_RSS;

    double m_AR2;

    double m_SMAPE;

    double m_RE;

    std::vector<double> m_predicted_points;

    std::vector<double> m_actual_points;

    std::vector<double> m_ps;

    std::vector<double> m_sizes;

    ModelCommentList m_comments;


private:
    /**
     * Make copy constructor unusable to prevent someone
     * makeing a shallow copy.
     */
    Model( const Model& );

    /**
     * Make assignment operator unusable to prevent someone
     * makeing a shallow copy.
     */
    Model&
    operator=( const Model& );
};
bool
equal( const Model* lhs,
       const Model* rhs );

/**
 * Type of the list of models
 */
typedef std::vector<Model*> ModelList;
};

#endif