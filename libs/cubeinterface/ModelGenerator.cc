#include "ModelGenerator.h"
#include "SingleParameterSimpleModelGenerator.h"
#include "SingleParameterExhaustiveModelGenerator.h"
#include "SingleParameterModelGenerator.h"
#include "SingleParameterRefiningModelGenerator.h"

namespace EXTRAP
{
ModelGenerator::~ModelGenerator()
{
}

Model*
ModelGenerator::createModel( const Experiment*            experiment,
                             const ModelGeneratorOptions& options,
                             const ExperimentPointList&   dataPoints,
                             const Function*              expectationFunction )
{
    return NULL;
}

bool
ModelGenerator::serialize( IoHelper* ioHelper ) const
{
    ErrorStream << "NOT IMPLEMENTED ModelGenerator should not be initialized directly!" << std::endl;
    return false;
}

std::string
ModelGenerator::getUserName( void ) const
{
    return m_user_name;
}

void
ModelGenerator::setUserName( const std::string& newName )
{
    m_user_name = newName;
}

void
ModelGenerator::setId( int64_t id )
{
    m_id = id;
}

int64_t
ModelGenerator::getId( void ) const
{
    return m_id;
}

bool
equal( const ModelGenerator* lhs, const ModelGenerator* rhs )
{
    const SingleParameterModelGenerator* lhsBase = dynamic_cast<const SingleParameterModelGenerator*>( lhs );
    const SingleParameterModelGenerator* rhsBase = dynamic_cast<const SingleParameterModelGenerator*>( rhs );
    if ( lhsBase != NULL  && rhsBase != NULL )
    {
        const SingleParameterExhaustiveModelGenerator* lhs1 = dynamic_cast<const SingleParameterExhaustiveModelGenerator*>( lhs );
        const SingleParameterExhaustiveModelGenerator* rhs1 = dynamic_cast<const SingleParameterExhaustiveModelGenerator*>( rhs );
        if ( lhs1 != NULL && rhs1 != NULL )
        {
            return equal( lhs1, rhs1 );
        }
        const SingleParameterRefiningModelGenerator* lhs2 = dynamic_cast<const SingleParameterRefiningModelGenerator*>( lhs );
        const SingleParameterRefiningModelGenerator* rhs2 = dynamic_cast<const SingleParameterRefiningModelGenerator*>( rhs );
        if ( lhs2 != NULL && rhs2 != NULL )
        {
            return equal( lhs2, rhs2 );
        }
        const SingleParameterSimpleModelGenerator* lhs3 = dynamic_cast<const SingleParameterSimpleModelGenerator*>( lhs );
        const SingleParameterSimpleModelGenerator* rhs3 = dynamic_cast<const SingleParameterSimpleModelGenerator*>( rhs );
        if ( lhs3 != NULL && rhs3 != NULL )
        {
            return equal( lhs3, rhs3 );
        }
        ErrorStream << "Failed to compare model generators. Both are subclass of SingleParameterModelGenerator though." << std::endl;
        return false;
    }
    else
    {
        ErrorStream << "Failed to compare model generators. Either both have different Types or are of unknown Type." << std::endl;
        return false;
    }
}
}; // Close namespace