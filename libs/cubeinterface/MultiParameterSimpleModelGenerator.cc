#include "MultiParameterSimpleModelGenerator.h"
#include "MultiParameterSimpleFunctionModeler.h"
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
const std::string MultiParameterSimpleModelGenerator::MULTIPARAMETERSIMPLEMODELGENERATOR_PREFIX = "MultiParameterSimpleModelGenerator";

MultiParameterSimpleModelGenerator::MultiParameterSimpleModelGenerator()
{
    m_modeler = new MultiParameterSimpleFunctionModeler();
}

MultiParameterFunctionModeler&
MultiParameterSimpleModelGenerator::getFunctionModeler() const
{
    return *m_modeler;
}

bool
MultiParameterSimpleModelGenerator::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  MULTIPARAMETERSIMPLEMODELGENERATOR_PREFIX ) );
    MultiParameterModelGenerator::serialize( ioHelper );
    return true;
}

MultiParameterSimpleModelGenerator*
MultiParameterSimpleModelGenerator::deserialize( IoHelper* ioHelper )
{
    MultiParameterSimpleModelGenerator* generator = new MultiParameterSimpleModelGenerator();
    generator->MultiParameterModelGenerator::deserialize( ioHelper );
    return generator;
}

bool
equal( const MultiParameterSimpleModelGenerator* lhs,
       const MultiParameterSimpleModelGenerator* rhs )
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
    result &= equal( lhs->getModelGeneratorOptions(), rhs->getModelGeneratorOptions() );
    return result;
}
}; // Close namespace