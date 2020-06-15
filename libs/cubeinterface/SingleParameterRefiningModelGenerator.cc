#include "SingleParameterRefiningModelGenerator.h"
#include "SingleParameterRefiningFunctionModeler.h"

namespace EXTRAP
{
const std::string SingleParameterRefiningModelGenerator::SINGLEPARAMETERREFININGMODELGENERATOR_PREFIX = "SingleParameterRefiningModelGenerator";

SingleParameterRefiningModelGenerator::SingleParameterRefiningModelGenerator()
{
    m_modeler = new SingleParameterRefiningFunctionModeler();
}

SingleParameterFunctionModeler&
SingleParameterRefiningModelGenerator::getFunctionModeler() const
{
    return *m_modeler;
}

bool
SingleParameterRefiningModelGenerator::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  SINGLEPARAMETERREFININGMODELGENERATOR_PREFIX ) );
    //write id
    SAFE_RETURN( this->SingleParameterModelGenerator::serialize( ioHelper ) );
    return true;
}

SingleParameterRefiningModelGenerator*
SingleParameterRefiningModelGenerator::deserialize( IoHelper* ioHelper )
{
    SingleParameterRefiningModelGenerator* generator = new SingleParameterRefiningModelGenerator();
    generator->SingleParameterModelGenerator::deserialize( ioHelper );
    return generator;
}

bool
equal( const SingleParameterRefiningModelGenerator* lhs,
       const SingleParameterRefiningModelGenerator* rhs )
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