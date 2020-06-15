#include "SingleParameterSimpleModelGenerator.h"
#include "SingleParameterSimpleFunctionModeler.h"
#include "Utilities.h"
#include <sstream>

namespace EXTRAP
{
const std::string SingleParameterSimpleModelGenerator::SINGLEPARAMETERSIMPLEMODELGENERATOR_PREFIX = "SingleParameterSimpleModelGenerator";

SingleParameterSimpleModelGenerator::SingleParameterSimpleModelGenerator()
{
    m_modeler = new SingleParameterSimpleFunctionModeler();
    //percentage term contribution
    m_modeler->setEpsilon( 0.0005 );
    m_modeler->setMaxTermCount( 1 );
}

void
SingleParameterSimpleModelGenerator::generateHypothesisBuildingBlockSet
(
    const std::vector<double>& poly_exponents,
    const std::vector<double>& log_exponents
)
{
    this->m_modeler->generateHypothesisBuildingBlockSet( poly_exponents, log_exponents );
}

/*
void
SingleParameterSimpleModelGenerator::generateDefaultHypothesisBuildingBlockSet()
{
    this->m_modeler->generateDefaultHypothesisBuildingBlocks();
}

//TODO: fix for the sparse modeler evaluation
void
SingleParameterSimpleModelGenerator::generateDefaultHypothesisBuildingBlocks()
{
    this->m_modeler->generateDefaultHypothesisBuildingBlocks();
}
*/

void
SingleParameterSimpleModelGenerator::addHypothesisBuildingBlock( CompoundTerm term )
{
    this->m_modeler->addHypothesisBuildingBlock( term );
}

void
SingleParameterSimpleModelGenerator::printHypothesisBuildingBlocks( void )
{
    this->m_modeler->printHypothesisBuildingBlocks();
}

void
SingleParameterSimpleModelGenerator::setMaxTermCount( int term_count )
{
    this->m_modeler->setMaxTermCount( term_count );
}

int
SingleParameterSimpleModelGenerator::getMaxTermCount( void ) const
{
    return this->m_modeler->getMaxTermCount();
}

SingleParameterFunctionModeler&
SingleParameterSimpleModelGenerator::getFunctionModeler() const
{
    return *m_modeler;
}

bool
SingleParameterSimpleModelGenerator::serialize( IoHelper* ioHelper ) const
{
    std::vector<CompoundTerm> buildingBlocks = this->getBuildingBlocks();
    SAFE_RETURN( ioHelper->writeString(  SINGLEPARAMETERSIMPLEMODELGENERATOR_PREFIX ) );
    SAFE_RETURN(  this->SingleParameterModelGenerator::serialize(  ioHelper ) );
    //Write CompoundTerms
    SAFE_RETURN( ioHelper->writeInt(  buildingBlocks.size() ) );
    for ( std::vector<CompoundTerm>::const_iterator it = buildingBlocks.begin(); it != buildingBlocks.end(); it++ )
    {
        CompoundTerm c = *it;
        SAFE_RETURN(  c.serialize(  ioHelper ) );
    }
    SAFE_RETURN( ioHelper->writeInt(  this->getMaxTermCount() ) );
    return true;
}

SingleParameterSimpleModelGenerator*
SingleParameterSimpleModelGenerator::deserialize( IoHelper* ioHelper )
{
    SingleParameterSimpleModelGenerator* generator = new SingleParameterSimpleModelGenerator();
    generator->SingleParameterModelGenerator::deserialize(  ioHelper );
    //Read CompoundTerms
    int                       length = ioHelper->readInt();
    std::vector<CompoundTerm> buildingBlocks;
    for ( int i = 0; i < length; i++ )
    {
        std::string prefix = ioHelper->readString();
        assert( prefix.compare( CompoundTerm::COMPOUNDTERM_PREFIX ) == 0 );
        CompoundTerm compoundTerm = CompoundTerm::deserialize(  ioHelper );
        generator->addHypothesisBuildingBlock( compoundTerm );
    }
    //Read MaxTermCount
    int maxTermCount = ioHelper->readInt();
    generator->setMaxTermCount( maxTermCount );
    return generator;
}

const std::vector<CompoundTerm>&
SingleParameterSimpleModelGenerator::getBuildingBlocks() const
{
    return this->m_modeler->getBuildingBlocks();
}

bool
equal( const SingleParameterSimpleModelGenerator* lhs,
       const SingleParameterSimpleModelGenerator* rhs )
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
    const std::vector<CompoundTerm>& terms1 = lhs->getBuildingBlocks();
    const std::vector<CompoundTerm>& terms2 = rhs->getBuildingBlocks();
    result &= terms1.size() == terms2.size();
    if ( !result )
    {
        return false;
    }
    for ( unsigned int i = 0; i < terms1.size(); i++ )
    {
        result &= equal( &terms1[ i ], &terms2[ i ] );
    }
    result &= lhs->getMaxTermCount() == rhs->getMaxTermCount();
    return result;
}
}; // Close namespace