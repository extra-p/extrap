#ifndef MODEL_GENERATOR_H
#define MODEL_GENERATOR_H

#include "Model.h"
#include "ExperimentPoint.h"
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
/**
 * This is the base class and interface for model generators.
 */
class ModelGenerator
{
public:
    /**
     * Make destructor virtual
     */
    virtual
    ~ModelGenerator();

    /**
     * Creates a new model.
     * @param dataPoints    A list of data points from which we want to derive
     *                      a model.
     * @expectationFunction A function with an expectation of the model.
     */
    virtual Model*
    createModel( const Experiment*            experiment,
                 const ModelGeneratorOptions& options,
                 const ExperimentPointList&   dataPoints,
                 const Function*              expectationFunction = NULL );

    virtual bool
    serialize(
        IoHelper* ioHelper ) const;

    /**
     * Returns the user specified name for the generator configuration.
     */
    virtual std::string
    getUserName( void ) const;

    /**
     * Sets the user specified name for the generator configuration.
     * It allows a user to name his models.
     * @param newName  The name given to the model.
     */
    virtual void
    setUserName( const std::string& newName );

    virtual void
    setId( int64_t id );

    virtual int64_t
    getId( void ) const;

protected:
    std::string m_user_name;
    int64_t     m_id;
};

typedef std::vector<ModelGenerator*> ModelGeneratorList;

bool
equal( const ModelGenerator* lhs,
       const ModelGenerator* rhs );
};

#endif