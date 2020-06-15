#include "CompoundTerm.h"
#include <iostream>
#include <sstream>
#include <cassert>
#include "Utilities.h"

namespace EXTRAP
{
const std::string CompoundTerm::COMPOUNDTERM_PREFIX = "CompoundTerm";
CompoundTerm::CompoundTerm( void )
{
}

CompoundTerm::CompoundTerm( const CompoundTerm& obj ) :
    m_coefficient( obj.m_coefficient ),
    m_simple_terms( std::vector<SimpleTerm>( obj.m_simple_terms.begin(), obj.m_simple_terms.end() ) )
{
}

/*static*/ CompoundTerm
CompoundTerm::fromLegacy( int a, int b, int c )
{
    CompoundTerm result;
    result.setCoefficient( 1 );
    if ( a != 0 )
    {
        SimpleTerm t;
        t.setFunctionType( polynomial );
        t.setExponent( ( double )a / b );
        result.addSimpleTerm( t );
    }
    if ( c != 0 )
    {
        SimpleTerm t;
        t.setFunctionType( logarithm );
        t.setExponent( ( double )c );
        result.addSimpleTerm( t );
    }

    return result;
}

void
CompoundTerm::setCoefficient( Value v )
{
    this->m_coefficient = v;
}

Value
CompoundTerm::getCoefficient( void ) const
{
    return this->m_coefficient;
}

void
CompoundTerm::addSimpleTerm( const SimpleTerm& st )
{
    std::vector<SimpleTerm>::iterator it;
    for ( it = this->m_simple_terms.begin(); it != this->m_simple_terms.end(); it++ )
    {
        assert( st.getFunctionType() != it->getFunctionType() && "No function type may appear more than once" );
    }
    this->m_simple_terms.push_back( st );
}

const std::vector<SimpleTerm>&
CompoundTerm::getSimpleTerms( void ) const
{
    return this->m_simple_terms;
}

Value
CompoundTerm::evaluate( const ParameterValueList& parameterValues ) const
{
    Value functionValue = this->m_coefficient;
    for ( int i = 0; i < this->m_simple_terms.size(); i++ )
    {
        functionValue *= this->m_simple_terms[ i ].evaluate( parameterValues );
    }
    return functionValue;
}

std::string
CompoundTerm::getAsString( const std::string& parameterName ) const
{
    std::stringstream returnValue;
    returnValue << "+" << this->m_coefficient;
    for ( int i = 0; i < this->m_simple_terms.size(); i++ )
    {
        returnValue << m_simple_terms[ i ].getAsString( parameterName );
    }
    std::string returnString;
    returnValue >> returnString;
    return returnString;
}


std::string
CompoundTerm::getAsStringMultiParameter( const std::string& parameterName ) const
{
    std::stringstream returnValue;
    for ( int i = 0; i < this->m_simple_terms.size(); i++ )
    {
        returnValue << "*" << m_simple_terms[ i ].getAsStringMultiParameter( parameterName );
    }
    std::string returnString;
    returnValue >> returnString;
    return returnString;
}

std::string
CompoundTerm::getAsString( const ParameterList& parameterNames ) const
{
    return getAsString( parameterNames[ 0 ].getName() );
}

Function*
CompoundTerm::clone( void ) const
{
    return new CompoundTerm( *this );
}

bool
CompoundTerm::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  COMPOUNDTERM_PREFIX ) );
    SAFE_RETURN( ioHelper->writeValue(  this->m_coefficient ) );
    SAFE_RETURN( ioHelper->writeInt(  this->m_simple_terms.size() ) );
    for ( std::vector<SimpleTerm>::const_iterator it = this->m_simple_terms.begin(); it != this->m_simple_terms.end(); it++ )
    {
        SimpleTerm t = *it;
        SAFE_RETURN( t.serialize(  ioHelper ) );
    }
    return true;
}

CompoundTerm
CompoundTerm::deserialize( IoHelper* ioHelper )
{
    CompoundTerm compoundTerm;
    Value        coefficient = ioHelper->readValue();
    compoundTerm.setCoefficient( coefficient );
    int length = ioHelper->readInt();
    for ( int i = 0; i < length; i++ )
    {
        std::string prefix = ioHelper->readString();
        assert( prefix.compare( SimpleTerm::SIMPLETERM_PREFIX ) == 0 );
        SimpleTerm term = SimpleTerm::deserialize(  ioHelper );
        compoundTerm.addSimpleTerm( term );
    }
    return compoundTerm;
}

FunctionClass
CompoundTerm::getFunctionClass() const
{
    return FUNCTION_CLASS_COMPOUNDTERM;
}

bool
equal( const CompoundTerm* lhs, const CompoundTerm* rhs )
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
    result &= lhs->getCoefficient() == rhs->getCoefficient();
    const std::vector<SimpleTerm> termsLhs = lhs->getSimpleTerms();
    const std::vector<SimpleTerm> termsRhs = rhs->getSimpleTerms();
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