#include "MultiParameterSparseModelGenerator.h"
#include "MultiParameterSparseFunctionModeler.h"

namespace EXTRAP
{
const std::string MultiParameterSparseModelGenerator::MULTIPARAMETERSPARSEMODELGENERATOR_PREFIX = "MultiParameterSparseModelGenerator";

MultiParameterSparseModelGenerator::MultiParameterSparseModelGenerator()
{
    m_modeler = new MultiParameterSparseFunctionModeler();
}

MultiParameterFunctionModeler&
MultiParameterSparseModelGenerator::getFunctionModeler() const
{
    return *m_modeler;
}

bool
MultiParameterSparseModelGenerator::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  MULTIPARAMETERSPARSEMODELGENERATOR_PREFIX ) );
    MultiParameterModelGenerator::serialize( ioHelper );
    return true;
}

MultiParameterSparseModelGenerator*
MultiParameterSparseModelGenerator::deserialize( IoHelper* ioHelper )
{
    MultiParameterSparseModelGenerator* generator = new MultiParameterSparseModelGenerator();
    generator->MultiParameterModelGenerator::deserialize( ioHelper );
    return generator;
}

bool
equal( const MultiParameterSparseModelGenerator* lhs,
       const MultiParameterSparseModelGenerator* rhs )
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