#ifndef MODELCOMMENT_H
#define MODELCOMMENT_H

#include "Utilities.h"
#include "Types.h"
#include <set>
#include <fstream>
#include <map>
#include <cassert>
#include "IoHelper.h"

namespace EXTRAP
{
class ModelComment
{
protected:
    std::string m_message;
    int64_t     m_id;

public:
    static const std::string MODELCOMMENT_PREFIX;
    ModelComment( const std::string& message );

    virtual
    ~ModelComment();

    const std::string&
    getMessage();

    virtual bool
    serialize(                IoHelper* ioHelper ) const;

    static ModelComment*
    deserialize(                  IoHelper* ioHelper );

    virtual void
    setId( int64_t id );

    virtual int64_t
    getId( void ) const;
};
bool
equal( ModelComment* lhs,
       ModelComment* rhs );

/**
 * Type of the list of model comments
 */
typedef std::vector<ModelComment*> ModelCommentList;
};
#endif