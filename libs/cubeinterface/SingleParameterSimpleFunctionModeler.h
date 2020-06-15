#ifndef SINGLE_PARAMETER_SIMPLE_MODELER_H
#define SINGLE_PARAMETER_SIMPLE_MODELER_H

#include "SingleParameterFunctionModeler.h"
#include "ModelGenerator.h"
#include "CompoundTerm.h"
#include "SingleParameterFunction.h"
#include "SingleParameterHypothesis.h"
#include "ModelGeneratorOptions.h"

namespace EXTRAP
{
class SingleParameterSimpleFunctionModeler : public SingleParameterFunctionModeler
{
public:
    SingleParameterSimpleFunctionModeler();

    void
    addHypothesisBuildingBlock( CompoundTerm Term );

    void
    generateHypothesisBuildingBlockSet( const std::vector<double>& poly_exponents,
                                        const std::vector<double>& log_exponents );

    void
    generateDefaultHypothesisBuildingBlocks(bool allow_log);

    const std::vector<CompoundTerm>&
    getBuildingBlocks( void ) const;

    void
    printHypothesisBuildingBlocks( void );

    void
    setMaxTermCount( int );

    virtual SingleParameterHypothesis
    createModel( const Experiment*             experiment,
                 const ModelGeneratorOptions&  options,
                 const std::vector<DataPoint>& modeledDataPointList,
                 ModelCommentList&             comments,
                 const Function*               expectationFunction = NULL );

    virtual bool
    initSearchSpace( void );

    virtual bool
    nextHypothesis( void );

    virtual SingleParameterFunction*
    buildCurrentHypothesis( void );

    int
    getMaxTermCount( void ) const;

protected:
    //Start of external state
    std::vector<CompoundTerm> m_hypotheses_building_blocks;
    int                       m_max_term_count;
    //End of external state

    int              m_current_hypothesis;
    int              m_current_term_count;
    std::vector<int> m_current_hypothesis_building_block_vector;
};
};

#endif