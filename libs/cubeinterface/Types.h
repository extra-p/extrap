#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <cstdlib>

namespace EXTRAP
{
/**
 * This type defines the base type for values that we can calculate and
 * display, or that might be returned by functions.
 * Currently, we use a double. However, if we discover that a double
 * is insufficient, we might change this to some other type, e.g. a class
 */
typedef double             Value;

typedef std::vector<Value> ValueList;

/**
 * This provides a constant to mark an invalid value.
 */
const Value INVALID_VALUE = 0;

/**
 * This type defines atype for intervals that have start and an end.
 */
typedef struct
{
    Value start;
    Value end;
} Interval;

/**
 * A vector of intervals.
 */
typedef std::vector<Interval> IntervalList;

//Serialization is done in SingleParameterModelGenerator, so when this is extended please add them respectively there
enum Crossvalidation
{
    CROSSVALIDATION_NONE,
    CROSSVALIDATION_LEAVE_ONE_OUT,
    CROSSVALIDATION_LEAVE_P_OUT,
    CROSSVALIDATION_K_FOLD,
    CROSSVALIDATION_TWO_FOLD
};

//Serialization is done in SingleParameterModelGenerator, so when this is extended please add them respectively there
enum GenerateModelOptions
{
    GENERATE_MODEL_MEAN,
    GENERATE_MODEL_MEDIAN
};

enum SparseModelerSingleParameterStrategy
{
    FIRST_POINTS_FOUND,
    MAX_NUMBER_POINTS,
    MOST_EXPENSIVE_POINTS,
    CHEAPEST_POINTS
};

enum SparseModelerMultiParameterStrategy
{
    INCREASING_COST,
    DECREASING_COST
};

//TODO: add class for multi parameter functions...
enum FunctionClass
{
    FUNCTION_CLASS_FUNCTION,
    FUNCTION_CLASS_MULTIPARAMETERFUNCTION,
    FUNCTION_CLASS_SINGLEPARAMETERFUNCTION,
    FUNCTION_CLASS_SIMPLETERM,
    FUNCTION_CLASS_COMPOUNDTERM
};

enum EndiannessConversion
{
    CONVERSION_REVERSE,
    CONVERSION_NONE
};
};

#endif