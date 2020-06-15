#include <cmath>
#include <iostream>
#include "SimpleTerm.h"
#include "Utilities.h"
#include <sstream>
#include <string>

namespace EXTRAP
{
const std::string SimpleTerm::SIMPLETERM_PREFIX = "SimpleTerm";
void
SimpleTerm::setFunctionType( FunctionType ft )
{
    this->m_function_type = ft;
}

void
SimpleTerm::setExponent( double exponent )
{
    this->m_exponent = exponent;
}

FunctionType
SimpleTerm::getFunctionType( void ) const
{
    return this->m_function_type;
}

double
SimpleTerm::getExponent( void ) const
{
    return this->m_exponent;
}

std::string
SimpleTerm::toString( void )
{
    std::string result = "NA";
    return result;
}


std::string
SimpleTerm::getAsString( const std::string& parameterName ) const
{
    std::stringstream returnValue;
    returnValue << "*";
    switch ( this->m_function_type )
    {
        case polynomial:
            returnValue << "(" << parameterName << "^" << this->m_exponent << ")";
            break;
        case logarithm:
            returnValue << "log2^" << this->m_exponent << "(" << parameterName << ")";
            break;
        default:
            ErrorStream << "Failed to convert to string. Unrecognized function type" << std::endl;
            break;
    }
    std::string returnString;
    returnValue >> returnString;
    return returnString;
}

std::string
SimpleTerm::getAsStringMultiParameter( const std::string& parameterName ) const
{
    std::stringstream returnValue;
    switch ( this->m_function_type )
    {
        case polynomial:
            returnValue << "(" << parameterName << "^" << this->m_exponent << ")";
            break;
        case logarithm:
            returnValue << "log2^" << this->m_exponent << "(" << parameterName << ")";
            break;
        default:
            ErrorStream << "Unrecognized function type" << std::endl;
            break;
    }
    std::string returnString;
    returnValue >> returnString;
    return returnString;
}


std::string
SimpleTerm::getAsString( const ParameterList& parameterNames ) const
{
    return getAsString( parameterNames[ 0 ].getName() );
}

Value
SimpleTerm::evaluate( const ParameterValueList& parameterValues ) const
{
    Value functionValue = 0;

    if ( parameterValues.size() != 1 )
    {
        //ErrorStream << "Failed to evaluate function. Multiple or no parameters in simple term" << std::endl;
    }
    double parameter = parameterValues.begin()->second;
    switch ( this->m_function_type )
    {
        case polynomial:
            functionValue = std::pow( parameter, this->m_exponent );
            break;
        case logarithm:
            functionValue = log2( parameter );
            functionValue = std::pow( functionValue, this->m_exponent );
            break;
        default:
            ErrorStream << "Failed to evaluate function. Unrecognized function type" << std::endl;
            break;
    }

    return functionValue;
}

Function*
SimpleTerm::clone( void ) const
{
    return new SimpleTerm( *this );
}

bool
SimpleTerm::serialize(  IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  SIMPLETERM_PREFIX ) );
    //write function Type
    std::string type;
    switch ( this->m_function_type )
    {
        case logarithm:
            type = std::string( FunctionTypeNames[ 0 ] );
            break;
        case polynomial:
            type = std::string( FunctionTypeNames[ 1 ] );
            break;
        default:
            ErrorStream << "Failed to serialize SimpleTerm. Wrong FunctionType: " << this->m_function_type << std::endl;
            return false;
            break;
    }
    SAFE_RETURN( ioHelper->writeString(  type ) );
    SAFE_RETURN( ioHelper->writeValue(  this->m_exponent ) );
    return true;
}

SimpleTerm
SimpleTerm::deserialize( IoHelper* ioHelper )
{
    //Read FunctionType
    std::string  functionType = ioHelper->readString();
    FunctionType type;
    if ( 0 == functionType.compare( FunctionTypeNames[ 0 ] ) )
    {
        type = logarithm;
    }
    else if ( 0 == functionType.compare( FunctionTypeNames[ 1 ] ) )
    {
        type = polynomial;
    }
    else
    {
        ErrorStream << "Failed to deserialize SimpleTerm. Read wrong FunctionType: " << functionType << "Defaulting to Polynomial" << std::endl;
        type = polynomial;
    }
    SimpleTerm term;
    Value      exponent = ioHelper->readValue();
    term.setExponent( exponent );
    term.setFunctionType( type );
    return term;
}

FunctionClass
SimpleTerm::getFunctionClass() const
{
    return FUNCTION_CLASS_SIMPLETERM;
}

bool
equal( const SimpleTerm* lhs, const SimpleTerm* rhs )
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
    result &= lhs->getExponent() == rhs->getExponent();
    result &= lhs->getFunctionType() == rhs->getFunctionType();
    return result;
}
};