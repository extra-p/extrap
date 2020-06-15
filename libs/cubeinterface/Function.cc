#include "Function.h"
#include "CompoundTerm.h"
#include "SingleParameterFunction.h"
#include "MultiParameterFunction.h"
#include "Utilities.h"

namespace EXTRAP
{
const std::string Function::FUNCTION_PREFIX = "Function";
Function::~Function()
{
}

Value
Function::evaluate( const ParameterValueList& parameterValues ) const
{
    return 0;
}

std::string
Function::getAsString( const ParameterList& parameterNames ) const
{
    return "0";
}

Function*
Function::clone( void ) const
{
    return new Function( *this );
}

bool
equal( const Function* lhs, const Function* rhs )
{
    if ( lhs->getFunctionClass() != rhs->getFunctionClass() )
    {
        ErrorStream << "Different Function Classes: " << lhs->getFunctionClass() << " " << rhs->getFunctionClass() << std::endl;
        return false;
    }
    if ( lhs->getFunctionClass() == FUNCTION_CLASS_SIMPLETERM )
    {
        const SimpleTerm* lhs1 = dynamic_cast<const SimpleTerm*>( lhs );
        const SimpleTerm* rhs1 = dynamic_cast<const SimpleTerm*>( rhs );
        return equal( lhs1, rhs1 );
    }
    else if ( lhs->getFunctionClass() == FUNCTION_CLASS_COMPOUNDTERM )
    {
        const CompoundTerm* lhs1 = dynamic_cast<const CompoundTerm*>( lhs );
        const CompoundTerm* rhs1 = dynamic_cast<const CompoundTerm*>( rhs );
        return equal( lhs1, rhs1 );
    }
    else if ( lhs->getFunctionClass() == FUNCTION_CLASS_SINGLEPARAMETERFUNCTION )
    {
        const SingleParameterFunction* lhs1 = dynamic_cast<const SingleParameterFunction*>( lhs );
        const SingleParameterFunction* rhs1 = dynamic_cast<const SingleParameterFunction*>( rhs );
        return equal( lhs1, rhs1 );
    }
    else if ( lhs->getFunctionClass() == FUNCTION_CLASS_FUNCTION )
    {
        return true;
    }
    else
    {
        ErrorStream << "Unknown Subtype of Class Function" << std::endl;
        return false;
    }
}

bool
Function::serialize( IoHelper* ioHelper ) const
{
    SAFE_RETURN( ioHelper->writeString(  FUNCTION_PREFIX ) );
    return true;
}

Function*
Function::deserialize( IoHelper* ioHelper, std::string prefix )
{
    if ( prefix.compare( SingleParameterFunction::SINGLEPARAMETERFUNCTION_PREFIX ) == 0 )
    {
        //Read a SingleParameterFunction
        SingleParameterFunction f = SingleParameterFunction::deserialize(  ioHelper );
        return f.clone();
    }
    else if ( prefix.compare( MultiParameterFunction::MULTIPARAMETERFUNCTION_PREFIX ) == 0 )
    {
        //Read a MultiParameterFunction
        MultiParameterFunction* f = MultiParameterFunction::deserialize(  ioHelper );
        return f;
    }
    else if ( prefix.compare( SimpleTerm::SIMPLETERM_PREFIX ) == 0 )
    {
        //Read a SimpleTerm
        SimpleTerm f = SimpleTerm::deserialize(  ioHelper );
        return f.clone();
    }
    else if ( prefix.compare( CompoundTerm::COMPOUNDTERM_PREFIX ) == 0 )
    {
        //Read a CompoundTerm
        CompoundTerm f = CompoundTerm::deserialize(  ioHelper );
        return f.clone();
    }
    else if ( prefix.compare( Function::FUNCTION_PREFIX ) == 0 )
    {
        Function f = Function();
        return f.clone();
    }
    else
    {
        std::cout << prefix << std::endl;
        ErrorStream << "Could not identify Function type: " << prefix << std::endl;
        return NULL;
    }
}

FunctionClass
Function::getFunctionClass() const
{
    return FUNCTION_CLASS_FUNCTION;
}
};