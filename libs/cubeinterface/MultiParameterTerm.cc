#include "MultiParameterTerm.h"
#include "CompoundTerm.h"
#include <iostream>
#include <sstream>
#include <cassert>
#include "Utilities.h"

namespace EXTRAP
{
const std::string MultiParameterTerm::MULTIPARAMETERTERM_PREFIX = "MultiParameterTerm";
MultiParameterTerm::MultiParameterTerm( void )
{
}

MultiParameterTerm::MultiParameterTerm( const MultiParameterTerm& obj ) :
    m_coefficient( obj.m_coefficient ),
    m_compound_term_parameter_pairs( std::vector<std::pair<CompoundTerm, Parameter> >( obj.m_compound_term_parameter_pairs.begin(), obj.m_compound_term_parameter_pairs.end() ) )
{
}

void
MultiParameterTerm::setCoefficient( Value v )
{
    this->m_coefficient = v;
}

Value
MultiParameterTerm::getCoefficient( void ) const
{
    return this->m_coefficient;
}

void
MultiParameterTerm::addCompoundTermParameterPair( const CompoundTerm& ct, const Parameter& param )
{
    std::vector<std::pair<CompoundTerm, Parameter> >::iterator it;
    for ( it = this->m_compound_term_parameter_pairs.begin(); it != this->m_compound_term_parameter_pairs.end(); it++ )
    {
        assert( param.getName() != it->second.getName() && "No function type may appear more than once" );
    }
    this->m_compound_term_parameter_pairs.push_back( std::make_pair( ct, param ) );
}

std::vector<std::pair<CompoundTerm, Parameter> >&
MultiParameterTerm::getCompoundTermParameterPairs( void )
{
    return this->m_compound_term_parameter_pairs;
}

const std::vector<std::pair<CompoundTerm, Parameter> >&
MultiParameterTerm::getCompoundTermParameterPairs( void ) const
{
    return this->m_compound_term_parameter_pairs;
}

Value
MultiParameterTerm::evaluate( const ParameterValueList& parameterValues ) const
{
    Value functionValue = this->m_coefficient;
    for ( int i = 0; i < this->m_compound_term_parameter_pairs.size(); i++ )
    {
        ParameterValueList tmp;
        if ( parameterValues.count( this->m_compound_term_parameter_pairs[ i ].second ) == 1 )
        {
            Value v = parameterValues.find( this->m_compound_term_parameter_pairs[ i ].second )->second;

            tmp.insert( std::make_pair( this->m_compound_term_parameter_pairs[ i ].second, v ) );
        }
        //assert( tmp.size() != 0 && "Parameters do not match expectation" );
        functionValue *= this->m_compound_term_parameter_pairs[ i ].first.evaluate( tmp );
    }
    return functionValue;
}


std::string
MultiParameterTerm::getAsString( const ParameterList& parameterNames ) const
{
    std::stringstream returnValue;
    returnValue << " + " << this->m_coefficient;
    for ( int i = 0; i < this->m_compound_term_parameter_pairs.size(); i++ )
    {
        returnValue << m_compound_term_parameter_pairs[ i ].first.getAsStringMultiParameter( m_compound_term_parameter_pairs[ i ].second.getName() );
    }
    std::string returnString;
    returnString = returnValue.str();
    return returnString;
}

Function*
MultiParameterTerm::clone( void ) const
{
    return new MultiParameterTerm( *this );
}

bool
MultiParameterTerm::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeValue(  this->m_coefficient ) );
    uint64_t size = this->m_compound_term_parameter_pairs.size();
    SAFE_RETURN( ioHelper->writeInt(  size ) );
    for ( uint64_t i = 0; i < size; i++ )
    {
        SAFE_RETURN( ioHelper->writeString( m_compound_term_parameter_pairs[ i ].second.getName() ) );
        SAFE_RETURN( m_compound_term_parameter_pairs[ i ].first.serialize( ioHelper ) );
    }
    return true;
}

MultiParameterTerm
MultiParameterTerm::deserialize( IoHelper* ioHelper )
{
    MultiParameterTerm new_term;
    new_term.setCoefficient( ioHelper->readValue() );
    uint64_t size = ioHelper->readInt();
    for ( uint64_t i = 0; i < size; i++ )
    {
        std::string name   = ioHelper->readString();
        std::string prefix = ioHelper->readString();
        assert( prefix.compare( CompoundTerm::COMPOUNDTERM_PREFIX ) == 0 );
        CompoundTerm term = CompoundTerm::deserialize( ioHelper );
        new_term.addCompoundTermParameterPair( term, Parameter( name ) );
    }
    return new_term;
}

bool
equal( const MultiParameterTerm* lhs, const MultiParameterTerm* rhs )
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
    const std::vector<std::pair<CompoundTerm, Parameter> > termsLhs = lhs->getCompoundTermParameterPairs();
    const std::vector<std::pair<CompoundTerm, Parameter> > termsRhs = rhs->getCompoundTermParameterPairs();
    if ( !termsLhs.size() == termsRhs.size() )
    {
        return false;
    }
    for ( int i = 0; i < termsLhs.size(); i++ )
    {
        result &= equal( &termsLhs[ i ].first, &termsRhs[ i ].first ) && equal( &termsLhs[ i ].second, &termsRhs[ i ].second );
    }
    return result;
}
};