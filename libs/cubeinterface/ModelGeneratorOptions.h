#ifndef MODEL_GENERATOR_OPTIONS_H
#define MODEL_GENERATOR_OPTIONS_H

#include "Types.h"

namespace EXTRAP
{
/**
 * This is the interface for model generator options.
 */
class ModelGeneratorOptions
{
public:

    ModelGeneratorOptions();

    ~ModelGeneratorOptions();

    void
    setGenerateModelOptions( GenerateModelOptions value );

    GenerateModelOptions
    getGenerateModelOptions( void );

    void
    setMinNumberPoints( int value );

    int
    getMinNumberPoints( void );

    void
    setSinglePointsStrategy( SparseModelerSingleParameterStrategy value );

    SparseModelerSingleParameterStrategy
    getSinglePointsStrategy( void );

    void
    setUseAddPoints( bool value );

    bool
    getUseAddPoints( void );

    void
    setNumberAddPoints( int value );

    int
    getNumberAddPoints( void );

    void
    setMultiPointsStrategy( SparseModelerMultiParameterStrategy value );

    SparseModelerMultiParameterStrategy
    getMultiPointsStrategy( void );

    void
    setUseLogTerms( bool value );

    bool
    getUseLogTerms( void );

    void
    setUseAutoSelect( bool value );

    bool
    getUseAutoSelect( void );

private:

    //options used for all modelers
    GenerateModelOptions generate_model_options;

    //single parameter experiment options (only sparse modeler)
    int                                  min_number_points;
    SparseModelerSingleParameterStrategy single_points_strategy;

    //multi parameter experiment options (only sparse modeler)
    bool                                use_add_points;
    int                                 number_add_points;
    SparseModelerMultiParameterStrategy multi_points_strategy;

    //to determine if log terms should be used for modeling
    bool allow_log_terms;

    //use auto select points for sparse modeling
    bool auto_select;
};

bool
equal( const ModelGeneratorOptions lhs,
       const ModelGeneratorOptions rhs );
}

#endif