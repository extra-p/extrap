#ifndef CUBE_MAPPING_H
#define CUBE_MAPPING_H

/**
 * @file
 * This file contains the definition of the EXTRAP::CubeMappings class.
 */

#include "Experiment.h"
#include <Cube.h>

namespace EXTRAP
{
/**
 * This class provides a mapping between metric, region and callpath/cnode
 * definitions in an Experiment and a Cube object. The mappings in both
 * directions are created.
 * There are two possibilities to create a mapping:
 * The first approach assumes that the experiment does not yet contain
 * any definitions. It copies all definitions from the Cube into the
 * Experiment and creates a mapping.
 * The second approach assumes that the Extra-P experiment already contains
 * all definitions and just creates the mappings.
 */
class CubeMapping
{
public:

    /**
     * Constructs a mapping object. It does not perform the mapping.
     * @param experiment The Extra-P experiment which is mapped to the Cube.
     * @param cube       The Cube that is mapped to the experiment.
     */
    CubeMapping( EXTRAP::Experiment* experiment,
                 cube::Cube*         cube );

    /**
     * Destructor
     */
    virtual
    ~CubeMapping();

    /**
     * Creates a mapping of the metric definitions. It assumes that
     * the Extra-P experiment already contains metric definitions that
     * can be mapped.
     */
    void
    createMetricMapping( void );

    /**
     * Creates a mapping of the region definitions. It assumes that
     * the Extra-P experiment already contains region definitions that
     * can be mapped.
     */
    void
    createRegionMapping( void );

    /**
     * Creates a mapping of the callpath definitions. It assumes that
     * the Extra-P experiment already contains callpath definitions that
     * can be mapped.
     */
    void
    createCallpathMapping( void );

    /**
     * Copies the metric definitions from the Cube to the Extra-P experiment
     * and creates a mapping between them. It does not check for duplicates
     * but assumes that the Extra-P experiment does not contain the metric
     * definitions, yet.
     */
    void
    createMetricDef( void );

    /**
     * Copies the region definitions from the Cube to the Extra-P experiment
     * and creates a mapping between them. It does not check for duplicates
     * but assumes that the Extra-P experiment does not contain the region
     * definitions, yet.
     */
    void
    createRegionDef( void );

    /**
     * Copies the callpath definitions from the Cube to the Extra-P experiment
     * and creates a mapping between them. It does not check for duplicates
     * but assumes that the Extra-P experiment does not contain the callpath
     * definitions, yet.
     */
    void
    createCallpathDef( void );

    /**
     * Returns the matching Cube metric definition for a given Extra-P
     * definition. If no matching definition exist, it returns NULL.
     * @param metric  A Extra-P metric definition for which the matching
     *                Cube definition is requested.
     */
    cube::Metric*
    getCubeMetric( const Metric* metric ) const;

    /**
     * Returns the matching Cube region definition for a given Extra-P
     * definition. If no matching definition exist, it returns NULL.
     * @param region  A Extra-P region definition for which the matching
     *                Cube definition is requested.
     */
    cube::Region*
    getCubeRegion( const Region* region ) const;

    /**
     * Returns the matching Cube Cnode definition for a given Extra-P
     * definition. If no matching definition exist, it returns NULL.
     * @param callpath  A Extra-P callpath definition for which the matching
     *                  Cube definition is requested.
     */
    cube::Cnode*
    getCubeCallpath( const Callpath* callpath ) const;

    /**
     * Returns the matching Extra-P metric definition for a given Cube
     * definition. If no matching definition exist, it returns NULL.
     * @param metric  A Cube metric definition for which the matching
     *                Extra-P definition is requested.
     */
    Metric*
    getExtrapMetric( const cube::Metric* ) const;

    /**
     * Returns the matching Extra-P region definition for a given Cube
     * definition. If no matching definition exist, it returns NULL.
     * @param metric  A Cube region definition for which the matching
     *                Extra-P definition is requested.
     */
    Region*
    getExtrapRegion( const cube::Region* ) const;

    /**
     * Returns the matching Extra-P callpath definition for a given Cube
     * definition. If no matching definition exist, it returns NULL.
     * @param metric  A Cube Cnode definition for which the matching
     *                Extra-P definition is requested.
     */
    Callpath*
    getExtrapCallpath( const cube::Cnode* ) const;

protected:
    /**
     * Compares a Cube definition with an Extra-P definition.
     */
    bool
    compareMetric( const EXTRAP::Metric& extrapMetric,
                   const cube::Metric&   cubeMetric );

    /**
     * Compares a Cube definition with an Extra-P definition.
     */
    bool
    compareRegion( const EXTRAP::Region& extrapRegion,
                   const cube::Region&   cubeRegion );

    /**
     * Compares a Cube definition with an Extra-P definition.
     */
    bool
    compareCallpath( const Callpath&    extrapCallpath,
                     const cube::Cnode& cubeCallpath );

    /**
     * Recursive helper function to create callpath mappings.
     */
    void
    createCallpathMappingRec( const CallpathList&              extrapCallpaths,
                              const std::vector<cube::Cnode*>& cubeCallpaths );

    /**
     * Recursive helper function to create callpath definitions and mappings.
     */
    void
    createCallpathDefRec( cube::Cnode* cubeCnode );

private:
    /**
     * The Extra-P experiment.
     */
    Experiment* m_experiment;

    /**
     * The Cube
     */
    cube::Cube* m_cube;

    /**
     * Maps Extra-P metrics to Cube metrics.
     */
    std::map<const Metric*, cube::Metric*> m_metric_extrap_to_cube;

    /**
     * Maps Extra-P regions to Cube regions.
     */
    std::map<const Region*, cube::Region*> m_region_extrap_to_cube;

    /**
     * Maps Extra-P callpaths to Cube cnodes.
     */
    std::map<const Callpath*, cube::Cnode*> m_callpath_extrap_to_cube;

    /**
     * Maps Cube metrics to Extra-P metrics.
     */
    std::map<const cube::Metric*, Metric*> m_metric_cube_to_extrap;

    /**
     * Maps Cube regions to Extra-P regions.
     */
    std::map<const cube::Region*, Region*> m_region_cube_to_extrap;

    /**
     * Maps Cube cnodes to Extra-P callpaths.
     */
    std::map<const cube::Cnode*, Callpath*> m_callpath_cube_to_extrap;
};
}; // namespace EXTRAP

#endif