#include "Model.h"
#include "Function.h"
#include "Utilities.h"
#include <limits>
#include "Experiment.h"

namespace EXTRAP
{
IntervalList Model::m_whole_interval;

const std::string Model::MODEL_PREFIX = "Model";
Model::~Model( void )
{
    // delete all functions and function pairs
    delete m_model_function;
    delete m_confidence_interval->lower;
    delete m_confidence_interval->upper;
    delete m_confidence_interval;
    delete m_error_cone->lower;
    delete m_error_cone->upper;
    delete m_error_cone;
    delete m_noise_error->lower;
    delete m_noise_error->upper;
    delete m_noise_error;

    for ( ModelCommentList::iterator it = m_comments.begin(); it != m_comments.end(); ++it )
    {
        delete *it;
    }
}

Model::Model( Function*               modelFunction,
              ModelGenerator*         generator,
              FunctionPair*           confidenceInterval,
              FunctionPair*           errorCone,
              FunctionPair*           noiseError,
              const ModelCommentList& comments,
              double                  RSS,
              double                  AR2,
              double                  SMAPE,
              double                  RE,
              std::vector<double>     predicted_points,
              std::vector<double>     actual_points,
              std::vector<double>     ps,
              std::vector<double>     sizes )
    : m_model_function( modelFunction ),
    m_generator( generator ),
    m_confidence_interval( confidenceInterval ),
    m_error_cone( errorCone ),
    m_noise_error( noiseError ),
    m_comments( comments ),
    m_RSS( RSS ),
    m_AR2( AR2 ),
    m_SMAPE( SMAPE ),
    m_RE( RE ),
    m_predicted_points(predicted_points),
    m_actual_points(actual_points),
    m_ps(ps),
    m_sizes(sizes)
{
    if ( m_whole_interval.empty() )
    {
        Interval i = { 1, std::numeric_limits<Value>::max() }; // Interval from 1 to infinity
        m_whole_interval.push_back( i );
    }
}

Function*
Model::getModelFunction( void ) const
{
    return m_model_function;
}

double
Model::getRSS( void ) const
{
    return m_RSS;
}

double
Model::getAR2( void ) const
{
    return m_AR2;
}

double
Model::getSMAPE( void ) const
{
    return m_SMAPE;
}

double
Model::getRE( void ) const
{
    return m_RE;
}

std::vector<double>
Model::getPredictedPoints( void )
{
    return m_predicted_points;
}

std::vector<double>
Model::getActualPoints( void )
{
    return m_actual_points;
}

std::vector<double>
Model::getPs( void )
{
    return m_ps;
}

std::vector<double>
Model::getSizes( void )
{
    return m_sizes;
}

void
Model::setModelFunction( Function* f )
{
    delete m_model_function;
    m_model_function = f;
}


FunctionPair*
Model::getConfidenceInterval( void ) const
{
    return m_confidence_interval;
}

FunctionPair*
Model::getErrorCone( void ) const
{
    return m_error_cone;
}

FunctionPair*
Model::getNoiseError( void ) const
{
    return m_noise_error;
}

ModelGenerator*
Model::getGenerator( void ) const
{
    return m_generator;
}

const ModelCommentList&
Model::getComments( void ) const
{
    return m_comments;
}

void
Model::addComment( ModelComment* comment )
{
    m_comments.push_back( comment );
}
void
Model::clearComments( void )
{
    m_comments.clear();
}


const IntervalList*
Model::getValidIntervals( const Parameter& parameter ) const
{
    ValidIntervalsList::const_iterator it = m_valid_intervals.find( parameter );
    if ( it == m_valid_intervals.end() )
    {
        return &m_whole_interval;
    }
    return &it->second;
}

void
Model::addValidInterval( const Parameter& parameter,
                         const Interval&  interval )
{
    m_valid_intervals[ parameter ].push_back( interval );
}

bool
Model::serialize(
    Metric*   metric,
    Callpath* cp,
    IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  MODEL_PREFIX ) );
    SAFE_RETURN( ioHelper->writeId(  metric->getId() ) );
    SAFE_RETURN( ioHelper->writeId(  cp->getId() ) );
    SAFE_RETURN( ioHelper->writeId(  m_generator->getId() ) );
    SAFE_RETURN(  this->m_model_function->serialize(  ioHelper ) );
    SAFE_RETURN(  this->m_confidence_interval->upper->serialize(  ioHelper ) );
    SAFE_RETURN(  this->m_confidence_interval->lower->serialize(  ioHelper ) );
    SAFE_RETURN(  this->m_error_cone->upper->serialize(  ioHelper ) );
    SAFE_RETURN(  this->m_error_cone->lower->serialize(  ioHelper ) );
    SAFE_RETURN(  this->m_noise_error->upper->serialize(  ioHelper ) );
    SAFE_RETURN(  this->m_noise_error->lower->serialize(  ioHelper ) );
    SAFE_RETURN( ioHelper->writeValue(  this->m_RSS ) );
    SAFE_RETURN( ioHelper->writeValue(  this->m_AR2 ) );
    SAFE_RETURN( ioHelper->writeValue(  this->m_SMAPE ) );
    SAFE_RETURN( ioHelper->writeValue(  this->m_RE ) );
    int length = this->m_comments.size();
    SAFE_RETURN( ioHelper->writeInt(  length ) );
    for ( ModelCommentList::const_iterator it = this->m_comments.begin();
          it != this->m_comments.end();
          it++ )
    {
        ModelComment* c = *it;
        SAFE_RETURN( ioHelper->writeId(  c->getId() ) );
    }

    // Write valid intervals
    SAFE_RETURN( ioHelper->writeInt(  m_valid_intervals.size() ) );
    for ( ValidIntervalsList::const_iterator it = m_valid_intervals.begin();
          it != m_valid_intervals.end();
          it++ )
    {
        SAFE_RETURN( ioHelper->writeString(  it->first.getName() ) );
        const IntervalList* current_list = &it->second;
        SAFE_RETURN( ioHelper->writeInt(  current_list->size() ) );
        for ( IntervalList::const_iterator interval = current_list->begin();
              interval != current_list->end();
              interval++ )
        {
            SAFE_RETURN( ioHelper->writeValue(  interval->start ) );
            SAFE_RETURN( ioHelper->writeValue(  interval->end ) );
        }
    }
    return true;
}
Model*
Model::deserialize(
    const Experiment* experiment,
    int64_t*          metricId,
    int64_t*          callpathId,
    IoHelper*         ioHelper )
{
    std::string prefix;
    *metricId   = ioHelper->readId();
    *callpathId = ioHelper->readId();
    int64_t         generator_id    = ioHelper->readId();
    ModelGenerator* model_generator = experiment->getModelGenerator( generator_id );
    prefix = ioHelper->readString();
    Function*     model_function      = Function::deserialize(  ioHelper, prefix );
    FunctionPair* confidence_interval = new FunctionPair();
    prefix                     = ioHelper->readString();
    confidence_interval->upper = Function::deserialize(  ioHelper, prefix );
    SAFE_RETURN_NULL(  confidence_interval->upper );
    prefix                     = ioHelper->readString();
    confidence_interval->lower =  Function::deserialize(  ioHelper, prefix );
    SAFE_RETURN_NULL(  confidence_interval->lower );
    FunctionPair* error_cone_interval = new FunctionPair();
    prefix                     = ioHelper->readString();
    error_cone_interval->upper = Function::deserialize(  ioHelper, prefix );
    SAFE_RETURN_NULL(  error_cone_interval->upper );
    prefix                     = ioHelper->readString();
    error_cone_interval->lower = Function::deserialize(  ioHelper, prefix );
    SAFE_RETURN_NULL(  error_cone_interval->lower );
    FunctionPair* noise_error_interval = new FunctionPair();
    prefix                      = ioHelper->readString();
    noise_error_interval->upper = Function::deserialize(  ioHelper, prefix );
    SAFE_RETURN_NULL(  noise_error_interval->upper );
    prefix                      = ioHelper->readString();
    noise_error_interval->lower = Function::deserialize(  ioHelper, prefix );
    SAFE_RETURN_NULL(  noise_error_interval->lower );

    Value RSS   = ioHelper->readValue();
    Value AR2   = ioHelper->readValue();
    Value SMAPE = ioHelper->readValue();
    Value RE = ioHelper->readValue();

    ModelCommentList comments;
    int              length = ioHelper->readInt();
    for ( int i = 0; i < length; i++ )
    {
        int64_t       comment_id   = ioHelper->readId();
        ModelComment* modelComment = experiment->getModelComment( comment_id );
        comments.push_back( modelComment );
    }

    std::vector<double> predicted_points;
    std::vector<double> actual_points;
    std::vector<double> ps;
    std::vector<double> sizes;

    Model* model = new Model( model_function,
                              model_generator,
                              confidence_interval,
                              error_cone_interval,
                              noise_error_interval,
                              comments,
                              RSS,
                              AR2,
                              SMAPE,
                              RE,
                              predicted_points,
                              actual_points,
                              ps,
                              sizes);

    length = ioHelper->readInt();
    for ( int i = 0; i < length; i++ )
    {
        Parameter param( ioHelper->readString() );
        int       num_intervals = ioHelper->readInt();
        for ( int j = 0; j < num_intervals; j++ )
        {
            Interval interval;
            interval.start = ioHelper->readValue();
            interval.end   = ioHelper->readValue();
            model->addValidInterval( param, interval );
        }
    }

    return model;
}

bool
equal( const Model* lhs, const Model* rhs )
{
    if ( lhs == rhs )
    {
        return true;
    }
    if ( lhs == NULL || rhs == NULL )
    {
        return false;
    }
    bool result = true;
    result &= equal( lhs->getModelFunction(), rhs->getModelFunction() );
    //check is not activated because it fails for multi parameter generators, as a proper check for theses is not implemented!
    //result &= equal( lhs->getGenerator(), rhs->getGenerator() );
    result &= equal( lhs->getConfidenceInterval()->lower, rhs->getConfidenceInterval()->lower );
    result &= equal( lhs->getConfidenceInterval()->upper, rhs->getConfidenceInterval()->upper );
    result &= equal( lhs->getErrorCone()->lower, rhs->getErrorCone()->lower );
    result &= equal( lhs->getErrorCone()->upper, rhs->getErrorCone()->upper );
    result &= equal( lhs->getNoiseError()->lower, rhs->getNoiseError()->lower );
    result &= equal( lhs->getNoiseError()->upper, rhs->getNoiseError()->upper );

    ModelCommentList commentsLhs;
    ModelCommentList commentsRhs;
    result &= commentsLhs.size() == commentsRhs.size();
    if ( !result )
    {
        return false;
    }
    for ( int i = 0; i < commentsLhs.size(); i++ )
    {
        result &= equal( commentsLhs[ i ], commentsRhs[ i ] );
    }
    result &= lhs->getRSS() == rhs->getRSS();
    result &= lhs->getAR2() == rhs->getAR2();
    result &= lhs->getSMAPE() == rhs->getSMAPE();
    result &= lhs->getRE() == rhs->getRE();
    return result;
}
};