#ifndef METRIC_H
#define METRIC_H

#include <string>
#include <vector>
#include <set>
#include <map>
#include <fstream>
#include <cassert>
#include "IoHelper.h"

namespace EXTRAP
{
class Metric
{
public:

    static const std::string METRIC_PREFIX;
    Metric( const std::string& name,
            const std::string& unit );
    virtual
    ~Metric();
    virtual std::string
    getName( void ) const;
    virtual std::string
    getUnit( void ) const;
    int64_t
    getId( void ) const;
    void
    setId( int64_t id );
    virtual bool
    serialize(
        IoHelper* ioHelper ) const;
    static Metric*
    deserialize(
        IoHelper* ioHelper );

protected:
    std::string m_name;
    std::string m_unit;

private:
    int64_t m_id;
};
bool
equal( const Metric* lhs,
       const Metric* rhs );

bool
less( const Metric* lhs,
      const Metric* rhs );

typedef std::vector<Metric*> MetricList;
};

#endif