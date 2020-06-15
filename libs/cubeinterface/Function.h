#ifndef FUNCTION_H
#define FUNCTION_H

#include "Parameter.h"
#include <stdint.h>

namespace EXTRAP
{
class Function
{
public:
    static const std::string FUNCTION_PREFIX;
    virtual
    ~Function();

    virtual Value
    evaluate( const ParameterValueList& parameterValues ) const;

    virtual std::string
    getAsString( const ParameterList& parameterNames ) const;

    virtual Function*
    clone( void ) const;

    virtual bool
    serialize(
        IoHelper* ioHelper ) const;
    static Function*
    deserialize(
        IoHelper*   ioHelper,
        std::string prefix );

    virtual FunctionClass
    getFunctionClass() const;
};
bool
equal( const Function* lhs,
       const Function* rhs );

typedef struct
{
    Function* upper;
    Function* lower;
} FunctionPair;
};

#endif