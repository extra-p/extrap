#ifndef SINGLE_PARAMETER_FUNCTION_MODELER_H
#define SINGLE_PARAMETER_FUNCTION_MODELER_H

#include "ModelGenerator.h"
#include "CompoundTerm.h"
#include "SingleParameterHypothesis.h"
#include "DataPoint.h"
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
class SingleParameterFunction;

class SingleParameterFunctionModeler
{
public:
    SingleParameterFunctionModeler();

    virtual SingleParameterHypothesis
    createModel( const Experiment*             experiment,
                 const ModelGeneratorOptions&  options,
                 const std::vector<DataPoint>& modeledDataPointList,
                 ModelCommentList&             comments,
                 const Function*               expectationFunction = NULL ) = 0;

    virtual SingleParameterFunction*
    createConstantModel( const std::vector<DataPoint>& modeledDataPointList );

    virtual void
    setCrossvalidationMethod( Crossvalidation cvMethod );

    virtual Crossvalidation
    getCrossvalidationMethod( void ) const;

    static void
    getSingleParameterInterval( const std::vector<DataPoint>& modeledDataPointList,
                                Value*                        low,
                                Value*                        high );

    void
        setEpsilon( Value );

    Value
    getEpsilon( void ) const;


protected:
    /**
     * Returns the best valid hypothesis from the currently configured search space, starting with a given initialHypothesis.
     * The algorithm uses the abstract functions initSearchSpace, nextHypothesis, getCurrentHypothesis and compareHypotheses
     * to traverse the search space.
     *
     * The function assumes ownership of the passed initialHypothesis and its model function and will delete it
     * unless it is being returned as the best found hypothesis.
     */
    SingleParameterHypothesis
    findBestHypothesis( const std::vector<DataPoint>& modeledDataPointList,
                        SingleParameterHypothesis&    initialHypothesis );

    /**
     * Initializes or resets the search space, such that generation of hypotheses starts from the beginning and
     * getCurrentHypothesis() will return the first one. Returns false if the search space is empty.
     * This will be called before the first call to getCurrentHypothesis().
     */
    virtual bool
    initSearchSpace( void ) = 0;

    /**
     * Changes the internal state such that getCurrentHypothesis() will return the next hypothesis from the currently
     * used search space. Returns true when a next hypothesis exists and false after the complete search
     * space has been exhausted.
     */
    virtual bool
    nextHypothesis( void ) = 0;

    /**
     * Returns the current hypothesis from the search space. It is undefined what this method returns before
     * initSearchSpace() has been successfully called, or after nextHypothesis() has returned false.
     * The caller is responsible for deleting the returned object.
     */
    virtual SingleParameterFunction*
    buildCurrentHypothesis( void ) = 0;

    /**
     * Compares two hypotheses with respect to an arbitrary criterion and returns false if the candidate model is worse or
     * equally good as the old one, and true if the candidate model is better than the old one.
     *
     * By default this compares the RSS.
     */
    virtual bool
    compareHypotheses( const SingleParameterHypothesis& old,
                       const SingleParameterHypothesis& candidate,
                       const std::vector<DataPoint>&    modeledDataPointList );

    //Start of external state
    Crossvalidation m_crossvalidation;
    int             m_crossvalidation_parameter;
    Value           m_eps;
    //End of external state
};
};

#endif