#ifndef SINGLE_PARAMETER_SIMPLE_MODEL_GENERATOR_H
#define SINGLE_PARAMETER_SIMPLE_MODEL_GENERATOR_H

#include "ModelGenerator.h"
#include "CompoundTerm.h"
#include "SingleParameterHypothesis.h"
#include "SingleParameterModelGenerator.h"

namespace EXTRAP
{
class SingleParameterFunction;

class SingleParameterSimpleModelGenerator : public SingleParameterModelGenerator
{
public:
    static const std::string SINGLEPARAMETERSIMPLEMODELGENERATOR_PREFIX;
    SingleParameterSimpleModelGenerator();

    void
    generateHypothesisBuildingBlockSet
    (
        const std::vector<double>& poly_exponents,
        const std::vector<double>& log_exponents
    );

    /*
    void
    generateDefaultHypothesisBuildingBlockSet();

    //TODO: fix for the sparse modeler evaluation
    void
    generateDefaultHypothesisBuildingBlocks();
    */

    void
    addHypothesisBuildingBlock( CompoundTerm term );

    void
    printHypothesisBuildingBlocks( void );

    void
    setMaxTermCount( int );

    int
    getMaxTermCount( void ) const;

    const std::vector<CompoundTerm>&
    getBuildingBlocks( void ) const;

    virtual bool
    serialize(
        IoHelper* ioHelper ) const;

    static SingleParameterSimpleModelGenerator*
    deserialize(
        IoHelper* ioHelper );

protected:
    virtual SingleParameterFunctionModeler&
    getFunctionModeler() const;

    SingleParameterSimpleFunctionModeler* m_modeler; // FIXME: this should be a SingleParameterModelGenerator (base class)
};

bool
equal( const SingleParameterSimpleModelGenerator* lhs,
       const SingleParameterSimpleModelGenerator* rhs );
};

#endif