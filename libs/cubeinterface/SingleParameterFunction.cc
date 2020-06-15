#include "SingleParameterFunction.h"
#include <sstream>
#include <string>
#include "IoHelper.h"
#include "Utilities.h"

namespace EXTRAP
{
const std::string SingleParameterFunction::SINGLEPARAMETERFUNCTION_PREFIX = "SingleParameterFunction";
SingleParameterFunction::SingleParameterFunction( void ) : m_constant_coefficient( 0 )
{
}

SingleParameterFunction::SingleParameterFunction( const SingleParameterFunction& obj ) :
    m_constant_coefficient( obj.m_constant_coefficient ),
    m_compound_terms( std::vector<CompoundTerm>( obj.m_compound_terms.begin(), obj.m_compound_terms.end() ) )
{
}

void
SingleParameterFunction::addCompoundTerm( const CompoundTerm& ct )
{
    this->m_compound_terms.push_back( ct );
}

std::vector<CompoundTerm>&
SingleParameterFunction::getCompoundTerms( void )
{
    return this->m_compound_terms;
}

const std::vector<CompoundTerm>&
SingleParameterFunction::getCompoundTerms( void ) const
{
    return this->m_compound_terms;
}

void
SingleParameterFunction::setConstantCoefficient( Value v )
{
    this->m_constant_coefficient = v;
}

Value
SingleParameterFunction::getConstantCoefficient( void ) const
{
    return this->m_constant_coefficient;
}



Value
SingleParameterFunction::evaluate( const ParameterValueList& parameterValues ) const
{
    Value functionValue = this->m_constant_coefficient;
    for ( int i = 0; i < this->m_compound_terms.size(); i++ )
    {
        functionValue += m_compound_terms[ i ].evaluate( parameterValues );
    }
    return functionValue;
}

std::string
SingleParameterFunction::getAsString( const std::string& parameterName ) const
{
    std::stringstream returnValue;
    returnValue << this->m_constant_coefficient;
    for ( int i = 0; i < this->m_compound_terms.size(); i++ )
    {
        returnValue << m_compound_terms[ i ].getAsString( parameterName );
    }
    std::string returnString;
    returnValue >> returnString;
    return returnString;
}

std::string
SingleParameterFunction::getAsString( const ParameterList& parameterNames ) const
{
    return getAsString( parameterNames[ 0 ].getName() );
}

Function*
SingleParameterFunction::clone( void ) const
{
    return new SingleParameterFunction( *this );
}

bool
SingleParameterFunction::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  SINGLEPARAMETERFUNCTION_PREFIX ) );
    //write constant coefficient
    SAFE_RETURN( ioHelper->writeValue(  this->m_constant_coefficient ) );
    SAFE_RETURN( ioHelper->writeInt(  this->m_compound_terms.size() ) );
    for ( std::vector<CompoundTerm>::const_iterator it = this->m_compound_terms.begin(); it != this->m_compound_terms.end(); it++ )
    {
        CompoundTerm t = *it;
        SAFE_RETURN(  t.serialize(  ioHelper ) );
    }
    return true;
}
SingleParameterFunction
SingleParameterFunction::deserialize( IoHelper* ioHelper )
{
    SingleParameterFunction function    = SingleParameterFunction();
    Value                   coefficient = ioHelper->readValue();
    function.setConstantCoefficient( coefficient );
    int length = ioHelper->readInt();
    for ( int i = 0; i < length; i++ )
    {
        std::string prefix = ioHelper->readString();
        assert( prefix.compare( CompoundTerm::COMPOUNDTERM_PREFIX ) == 0 );
        CompoundTerm term = CompoundTerm::deserialize(  ioHelper );
        function.addCompoundTerm( term );
    }
    return function;
}

FunctionClass
SingleParameterFunction::getFunctionClass() const
{
    return FUNCTION_CLASS_SINGLEPARAMETERFUNCTION;
}

bool
equal( const SingleParameterFunction* lhs, const SingleParameterFunction* rhs )
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
    const std::vector<CompoundTerm> termsLhs = lhs->getCompoundTerms();
    const std::vector<CompoundTerm> termsRhs = rhs->getCompoundTerms();
    if ( termsLhs.size() != termsRhs.size() )
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