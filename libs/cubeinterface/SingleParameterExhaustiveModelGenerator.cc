#include "SingleParameterExhaustiveModelGenerator.h"
#include "SingleParameterExhaustiveFunctionModeler.h"

namespace EXTRAP
{
const std::string SingleParameterExhaustiveModelGenerator::SINGLEPARAMETEREXHAUSTIVEMODELGENERATOR_PREFIX = "SingleParameterExhaustiveModelGenerator";

SingleParameterExhaustiveModelGenerator::SingleParameterExhaustiveModelGenerator()
{
    m_modeler = new SingleParameterExhaustiveFunctionModeler();
}

void
SingleParameterExhaustiveModelGenerator::openOutputFile( const std::string& file )
{
    this->m_modeler->m_outputStream.open( file.c_str() );
    assert( this->m_modeler->m_outputStream.is_open() );
    this->m_modeler->writeHeader();
}

void
SingleParameterExhaustiveModelGenerator::closeOutputFile( void )
{
    this->m_modeler->m_outputStream.close();
}

void
SingleParameterExhaustiveModelGenerator::setFunctionName( const std::string& name )
{
    this->m_modeler->m_functionName = name;
}

SingleParameterFunctionModeler&
SingleParameterExhaustiveModelGenerator::getFunctionModeler() const
{
    return *m_modeler;
}

bool
SingleParameterExhaustiveModelGenerator::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString( SINGLEPARAMETEREXHAUSTIVEMODELGENERATOR_PREFIX ) );
    SAFE_RETURN( this->SingleParameterModelGenerator::serialize(  ioHelper ) );
    return true;
}

SingleParameterExhaustiveModelGenerator*
SingleParameterExhaustiveModelGenerator::deserialize( IoHelper* ioHelper )
{
    SingleParameterExhaustiveModelGenerator* generator = new SingleParameterExhaustiveModelGenerator();
    generator->SingleParameterModelGenerator::deserialize( ioHelper );
    return generator;
}

bool
equal( const SingleParameterExhaustiveModelGenerator* lhs,
       const SingleParameterExhaustiveModelGenerator* rhs )
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
    result &= lhs->getCrossvalidationMethod() == rhs->getCrossvalidationMethod();
    result &= lhs->getEpsilon() == rhs->getEpsilon();
    result &= equal( lhs->getModelGeneratorOptions(), rhs->getModelGeneratorOptions() );
    return result;
}
}; // Close namespace