#include "MultiParameterFunction.h"
#include <sstream>
#include <string>
#include "Utilities.h"
#include "IoHelper.h"

namespace EXTRAP
{
const std::string MultiParameterFunction::MULTIPARAMETERFUNCTION_PREFIX = "MultiParameterFunction";
MultiParameterFunction::MultiParameterFunction( void ) : m_constant_coefficient( 0 )
{
}

MultiParameterFunction::MultiParameterFunction( const MultiParameterFunction& obj ) :
    m_constant_coefficient( obj.m_constant_coefficient ),
    m_multi_parameter_terms( std::vector<MultiParameterTerm>( obj.m_multi_parameter_terms.begin(), obj.m_multi_parameter_terms.end() ) )
{
}

void
MultiParameterFunction::addMultiParameterTerm( const MultiParameterTerm& mt )
{
    this->m_multi_parameter_terms.push_back( mt );
}

std::vector<MultiParameterTerm>&
MultiParameterFunction::getMultiParameterTerms( void )
{
    return this->m_multi_parameter_terms;
}

const std::vector<MultiParameterTerm>&
MultiParameterFunction::getMultiParameterTerms( void ) const
{
    return this->m_multi_parameter_terms;
}

void
MultiParameterFunction::setConstantCoefficient( Value v )
{
    this->m_constant_coefficient = v;
}

Value
MultiParameterFunction::getConstantCoefficient( void ) const
{
    return this->m_constant_coefficient;
}



Value
MultiParameterFunction::evaluate( const ParameterValueList& parameterValues ) const
{
    Value functionValue = this->m_constant_coefficient;
    for ( int i = 0; i < this->m_multi_parameter_terms.size(); i++ )
    {
        functionValue += m_multi_parameter_terms[ i ].evaluate( parameterValues );
    }
    return functionValue;
}


std::string
MultiParameterFunction::getAsString( const ParameterList& parameterNames ) const
{
    std::stringstream returnValue;
    returnValue << " + " << this->m_constant_coefficient;
    for ( int i = 0; i < this->m_multi_parameter_terms.size(); i++ )
    {
        returnValue << m_multi_parameter_terms[ i ].getAsString( parameterNames );
    }
    std::string returnString;
    returnString = returnValue.str();
    return returnString;
}

Function*
MultiParameterFunction::clone( void ) const
{
    return new MultiParameterFunction( *this );
}

FunctionClass
MultiParameterFunction::getFunctionClass() const
{
    return FUNCTION_CLASS_MULTIPARAMETERFUNCTION;
}

bool
MultiParameterFunction::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString( MULTIPARAMETERFUNCTION_PREFIX ) );
    SAFE_RETURN( ioHelper->writeValue( m_constant_coefficient ) );

    uint64_t size = m_multi_parameter_terms.size();
    SAFE_RETURN( ioHelper->writeInt( size ) );
    for ( uint64_t i = 0; i < size; i++ )
    {
        SAFE_RETURN( m_multi_parameter_terms[ i ].serialize( ioHelper ) );
    }

    return true;
}

MultiParameterFunction*
MultiParameterFunction::deserialize( IoHelper* ioHelper )
{
    MultiParameterFunction* function = new MultiParameterFunction();

    function->setConstantCoefficient( ioHelper->readValue() );
    uint64_t size = ioHelper->readInt();
    for ( uint64_t i = 0; i < size; i++ )
    {
        MultiParameterTerm term = MultiParameterTerm::deserialize( ioHelper );
        function->addMultiParameterTerm( term );
    }

    return function;
}

bool
equal( const MultiParameterFunction* lhs, const MultiParameterFunction* rhs )
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
    result &= lhs->getConstantCoefficient() == rhs->getConstantCoefficient();
    const std::vector<MultiParameterTerm> termsLhs = lhs->getMultiParameterTerms();
    const std::vector<MultiParameterTerm> termsRhs = rhs->getMultiParameterTerms();
    if ( !( termsLhs.size() == termsRhs.size() ) )
    {
        return false;
    }
    for ( int i = 0; i < termsLhs.size(); i++ )
    {
        result &= equal( &termsLhs[ i ], &termsRhs[ i ] );
    }
    return result;
}
};