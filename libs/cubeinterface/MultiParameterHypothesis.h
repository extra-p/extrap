#ifndef MULTIPARAMETERHYPOTHESIS_H
#define MULTIPARAMETERHYPOTHESIS_H

#include "MultiParameterFunction.h"
#include "DataPoint.h"

namespace EXTRAP
{
class MultiParameterHypothesis
{
public:

    MultiParameterHypothesis( void );

    MultiParameterHypothesis( MultiParameterFunction* f );

    void
    computeCost( const std::vector<DataPoint>& points );

    void
    computeAdjustedRSquared( double                        TSS,
                             const std::vector<DataPoint>& points );

    void
    computeCostLeaveOneOutCrossValidation( const std::vector<DataPoint>& points,
                                           DataPoint                     missing );

    void
    estimateParameters( const std::vector<DataPoint>& points );

    MultiParameterFunction*
    getFunction( void ) const;

    double
    getRSS( void ) const;

    void
    setRSS( double rss );

    double
    getrRSS( void ) const;

    void
    setrRSS( double rrss );

    double
    getAR2( void ) const;

    void
    setAR2( double rss );

    double
    getSMAPE( void ) const;

    void
    setSMAPE( double smape );

    double
    getRE( void ) const;

    void
    setRE( double re );

    std::vector<double>
    getPredictedPoints( void );

    std::vector<double>
    getActualPoints( void );

    std::vector<double>
    getPs( void );

    std::vector<double>
    getSizes( void );

    bool
    isValid( void );

    double
    calculateMaximalTermContribution( int                           termIndex,
                                      const std::vector<DataPoint>& points ) const;

    void
    cleanConstantCoefficient( double                        epsilon,
                              const std::vector<DataPoint>& points );

    void
    keepFunctionAlive( void );

    void
    freeFunction( void );

private:
    struct HypothesisFunctionRefcount
    {
        unsigned int count;
        // The `func` member is there to find bugs in the usage of the reference counting scheme early and
        // more easily, because it allows us to use assertions that can detect when a hypothesis references
        // a reference counter that is actually referring to a different function than the hypothesis.
        // Theoretically `func` (and the associated assertions) could be removed in release builds.
        MultiParameterFunction* func;
    };

    MultiParameterFunction*     m_function;
    HypothesisFunctionRefcount* m_function_refcount;
    double                      m_RSS;
    double                      m_rRSS;
    double                      m_AR2;
    double                      m_SMAPE;
    double                      m_RE;
    bool                        m_keepFunctionAlive;
    std::vector<double>         m_predicted_points;
    std::vector<double>         m_actual_points;
    std::vector<double>         m_ps;
    std::vector<double>         m_sizes;
};
};

#endif