#include "MultiParameterModelGenerator.h"
#include "MultiParameterHypothesis.h"
#include "MultiParameterFunction.h"
#include "SingleParameterRefiningFunctionModeler.h"
#include "SingleParameterSimpleFunctionModeler.h"
#include "Utilities.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cassert>

namespace EXTRAP
{
MultiParameterFunctionModeler::MultiParameterFunctionModeler()
{
    //TODO: simple modeler here for the evaluation
    m_single_parameter_function_modeler = new SingleParameterSimpleFunctionModeler();
    //m_single_parameter_function_modeler = new SingleParameterRefiningFunctionModeler();
}

MultiParameterFunction*
MultiParameterFunctionModeler::createConstantModel( const std::vector<DataPoint>& modeledDataPointList )

{
    MultiParameterFunction* constantFunction = new MultiParameterFunction();

    DebugStream << "Creating constant model." << std::endl;

    double meanModel = 0;

    for ( int i = 0; i < modeledDataPointList.size(); i++ )
    {
        meanModel += modeledDataPointList[ i ].getValue() / ( double )modeledDataPointList.size();
    }

    constantFunction->setConstantCoefficient( meanModel );

    return constantFunction;
}
}; // Close namespace