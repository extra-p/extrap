#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
ModelGeneratorOptions::ModelGeneratorOptions()
{
}

ModelGeneratorOptions::~ModelGeneratorOptions()
{
}

void
ModelGeneratorOptions::setGenerateModelOptions( GenerateModelOptions value )
{
    generate_model_options = value;
}

GenerateModelOptions
ModelGeneratorOptions::getGenerateModelOptions( void )
{
    return generate_model_options;
}

void
ModelGeneratorOptions::setMinNumberPoints( int value )
{
    min_number_points = value;
}

int
ModelGeneratorOptions::getMinNumberPoints( void )
{
    return min_number_points;
}

void
ModelGeneratorOptions::setSinglePointsStrategy( SparseModelerSingleParameterStrategy value )
{
    single_points_strategy = value;
}

SparseModelerSingleParameterStrategy
ModelGeneratorOptions::getSinglePointsStrategy( void )
{
    return single_points_strategy;
}

void
ModelGeneratorOptions::setUseAddPoints( bool value )
{
    use_add_points = value;
}

bool
ModelGeneratorOptions::getUseAddPoints( void )
{
    return use_add_points;
}

void
ModelGeneratorOptions::setNumberAddPoints( int value )
{
    number_add_points = value;
}

int
ModelGeneratorOptions::getNumberAddPoints( void )
{
    return number_add_points;
}

void
ModelGeneratorOptions::setMultiPointsStrategy( SparseModelerMultiParameterStrategy value )
{
    multi_points_strategy = value;
}

SparseModelerMultiParameterStrategy
ModelGeneratorOptions::getMultiPointsStrategy( void )
{
    return multi_points_strategy;
}

void
ModelGeneratorOptions::setUseLogTerms( bool value ){
    allow_log_terms = value;
}

bool
ModelGeneratorOptions::getUseLogTerms( void ){
    return allow_log_terms;
}

void
ModelGeneratorOptions::setUseAutoSelect( bool value ){
    auto_select = value;
}

bool
ModelGeneratorOptions::getUseAutoSelect( void ){
    return auto_select;
}

bool
equal( const ModelGeneratorOptions lhs,
       const ModelGeneratorOptions rhs )
{
    ModelGeneratorOptions o1 = lhs;
    ModelGeneratorOptions o2 = rhs;
    if ( o1.getGenerateModelOptions() == o2.getGenerateModelOptions() )
    {
        if ( o1.getMinNumberPoints() == o2.getMinNumberPoints() )
        {
            if ( o2.getSinglePointsStrategy() == o2.getSinglePointsStrategy() )
            {
                if ( o1.getUseAddPoints() == o2.getUseAddPoints() )
                {
                    if ( o1.getNumberAddPoints() == o2.getNumberAddPoints() )
                    {
                        if ( o1.getMultiPointsStrategy() == o2.getMultiPointsStrategy() )
                        {
                            if (o1.getUseLogTerms() == o2.getUseLogTerms() )
                            {
                                if (o1.getUseAutoSelect() == o2.getUseAutoSelect() )
                                {
                                    return true;
                                }
                                else
                                {
                                    return false;
                                }
                            }
                            else
                            {
                                return false;
                            }
                        }
                        else
                        {
                            return false;
                        }
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }
}
};