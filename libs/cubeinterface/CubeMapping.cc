#include "CubeMapping.h"

namespace EXTRAP
{
/* **********************************************************************
* class CubeMapping
* **********************************************************************/

CubeMapping::CubeMapping( Experiment* experiment, cube::Cube* cube )
{
    m_experiment = experiment;
    m_cube       = cube;
}

CubeMapping::~CubeMapping()
{
}

cube::Metric*
CubeMapping::getCubeMetric( const Metric* extrapMetric ) const
{
    return m_metric_extrap_to_cube.at( extrapMetric );
}

cube::Region*
CubeMapping::getCubeRegion( const Region* extrapRegion ) const
{
    return m_region_extrap_to_cube.at( extrapRegion );
}

cube::Cnode*
CubeMapping::getCubeCallpath( const Callpath* extrapCallpath ) const
{
    return m_callpath_extrap_to_cube.at( extrapCallpath );
}

Region*
CubeMapping::getExtrapRegion( const cube::Region* cubeRegion ) const
{
    return m_region_cube_to_extrap.at( cubeRegion );
}

Metric*
CubeMapping::getExtrapMetric( const cube::Metric* cubeMetric ) const
{
    return m_metric_cube_to_extrap.at( cubeMetric );
}

Callpath*
CubeMapping::getExtrapCallpath( const cube::Cnode* cubeCallpath ) const
{
    return m_callpath_cube_to_extrap.at( cubeCallpath );
}

void
CubeMapping::createMetricDef()
{
    const std::vector<cube::Metric*> cube_metrics = m_cube->get_metv();

    int i;
    for ( i = 0; i < cube_metrics.size(); i++ )
    {
        Metric* metric = new Metric( cube_metrics[ i ]->get_uniq_name(),
                                     cube_metrics[ i ]->get_uom() );
        m_experiment->addMetric( metric );
        m_metric_extrap_to_cube[ metric ]            = cube_metrics[ i ];
        m_metric_cube_to_extrap[ cube_metrics[ i ] ] = metric;
    }
}

void
CubeMapping::createRegionDef()
{
    const std::vector<cube::Region*> cube_regions = m_cube->get_regv();

    int i;
    for ( i = 0; i < cube_regions.size(); i++ )
    {
        Region* region = new Region( cube_regions[ i ]->get_name(),
                                     cube_regions[ i ]->get_mod(),
                                     cube_regions[ i ]->get_begn_ln() );
        m_experiment->addRegion( region );
        m_region_extrap_to_cube[ region ]            = cube_regions[ i ];
        m_region_cube_to_extrap[ cube_regions[ i ] ] = region;
    }
}

void
CubeMapping::createCallpathDef()
{
    const std::vector<cube::Cnode*> cube_roots = m_cube->get_root_cnodev();
    int                             i;
    for ( i = 0; i < cube_roots.size(); i++ )
    {
        createCallpathDefRec( cube_roots[ i ] );
    }
}

void
CubeMapping::createCallpathDefRec( cube::Cnode* cubeCnode )
{
    Region*   region = getExtrapRegion( cubeCnode->get_callee() );
    Callpath* parent = NULL;
    if ( cubeCnode->get_parent() != NULL )
    {
        parent = getExtrapCallpath( cubeCnode->get_parent() );
    }
    Callpath* callpath = new Callpath( region, parent );
    m_experiment->addCallpath( callpath );
    m_callpath_extrap_to_cube[ callpath ]  = cubeCnode;
    m_callpath_cube_to_extrap[ cubeCnode ] = callpath;

    int i;
    for ( i = 0; i < cubeCnode->num_children(); i++ )
    {
        createCallpathDefRec( cubeCnode->get_child( i ) );
    }
}

void
CubeMapping::createMetricMapping()
{
    const std::vector<cube::Metric*> cube_metrics   = m_cube->get_metv();
    const MetricList&                extrap_metrics = m_experiment->getMetrics();

    // Initialize CUBE -> Extra-P
    int i;
    for ( i = 0; i < cube_metrics.size(); i++ )
    {
        m_metric_cube_to_extrap[ cube_metrics[ i ] ] = NULL;
    }

    // Build the mapping
    for ( i = 0; i < extrap_metrics.size(); i++ )
    {
        // Try same index guess first. Expect to match in most cases
        if ( i < cube_metrics.size() &&
             compareMetric( *extrap_metrics[ i ], *cube_metrics[ i ] ) )
        {
            m_metric_extrap_to_cube[ extrap_metrics[ i ] ] = cube_metrics[ i ];
            m_metric_cube_to_extrap[ cube_metrics[ i ] ]   = extrap_metrics[ i ];
            continue;
        }

        // Initialize the map with NULL in case no match is found. Is
        // overwritten if a match is found.
        m_metric_extrap_to_cube[ extrap_metrics[ i ] ] = NULL;

        // Else need to iterate over all entries to find the matching part
        int j;
        for ( j = 0; j < cube_metrics.size(); j++ )
        {
            if ( compareMetric( *extrap_metrics[ i ], *cube_metrics[ j ] ) )
            {
                m_metric_extrap_to_cube[ extrap_metrics[ i ] ] = cube_metrics[ j ];
                m_metric_cube_to_extrap[ cube_metrics[ j ] ]   = extrap_metrics[ i ];
                break;
            }
        }

        if ( m_metric_extrap_to_cube[ extrap_metrics[ i ] ] == NULL )
        {
            WarningStream << "No match found for metric '"
                          << extrap_metrics[ i ]->getName()
                          << "'" << std::endl;
        }
    }
}

void
CubeMapping::createRegionMapping()
{
    const std::vector<cube::Region*> cube_regions   = m_cube->get_regv();
    const RegionList&                extrap_regions = m_experiment->getRegions();

    // Initialize CUBE -> Extra-P
    int i;
    for ( i = 0; i < cube_regions.size(); i++ )
    {
        m_region_cube_to_extrap[ cube_regions[ i ] ] = NULL;
    }

    // Build the mapping
    for ( i = 0; i < extrap_regions.size(); i++ )
    {
        // Try same index guess first. Expect to match in most cases
        if ( i < cube_regions.size() &&
             compareRegion( *extrap_regions[ i ], *cube_regions[ i ] ) )
        {
            m_region_extrap_to_cube[ extrap_regions[ i ] ] = cube_regions[ i ];
            m_region_cube_to_extrap[ cube_regions[ i ] ]   = extrap_regions[ i ];
            continue;
        }

        // Initialize the map with NULL in case no match is found. Is
        // overwritten if a match is found.
        m_region_extrap_to_cube[ extrap_regions[ i ] ] = NULL;

        // Else need to iterate over all entries to find the matching part
        int j;
        for ( j = 0; j < cube_regions.size(); j++ )
        {
            if ( compareRegion( *extrap_regions[ i ], *cube_regions[ j ] ) )
            {
                m_region_extrap_to_cube[ extrap_regions[ i ] ] = cube_regions[ j ];
                m_region_cube_to_extrap[ cube_regions[ j ] ]   = extrap_regions[ i ];
                break;
            }
        }

        if ( m_region_extrap_to_cube[ extrap_regions[ i ] ] == NULL )
        {
            WarningStream << "No match found for region '"
                          << extrap_regions[ i ]->getName()
                          << "'" << std::endl;
        }
    }
}

void
CubeMapping::createCallpathMapping()
{
    const CallpathList& extrap_roots = m_experiment->getRootCallpaths();
    const CallpathList& extrap_all   = m_experiment->getAllCallpaths();

    const std::vector<cube::Cnode*>& cube_roots = m_cube->get_root_cnodev();
    const std::vector<cube::Cnode*>& cube_all   = m_cube->get_fullcnodev();

    // Initialize CUBE -> Extra-P
    for ( int i = 0; i < cube_all.size(); i++ )
    {
        m_callpath_cube_to_extrap[ cube_all[ i ] ] = NULL;
    }

    // Initialize Extra-P -> CUBE
    for ( int i = 0; i < extrap_all.size(); i++ )
    {
        m_callpath_extrap_to_cube[ extrap_all[ i ] ] = NULL;
    }

    // Build mapping
    createCallpathMappingRec( extrap_roots, cube_roots );
}

void
CubeMapping::createCallpathMappingRec( const CallpathList&              extrapCallpaths,
                                       const std::vector<cube::Cnode*>& cubeCallpaths )
{
    cube::Cnode* match;

    int i;
    for ( i = 0; i < extrapCallpaths.size(); i++ )
    {
        match = NULL;

        // Try same index guess first. Expect to match in most cases
        if ( i < cubeCallpaths.size() &&
             compareCallpath( *extrapCallpaths[ i ], *cubeCallpaths[ i ] ) )
        {
            match = cubeCallpaths[ i ];
        }
        else
        {
            // Else need to iterate over all entries to find the matching part
            int j;
            for ( j = 0; j < cubeCallpaths.size(); j++ )
            {
                if ( compareCallpath( *extrapCallpaths[ i ], *cubeCallpaths[ j ] ) )
                {
                    match = cubeCallpaths[ j ];
                    break;
                }
            }
        }

        if ( match != NULL )
        {
            m_callpath_extrap_to_cube[ extrapCallpaths[ i ] ] = match;

            m_callpath_cube_to_extrap[ match ] = extrapCallpaths[ i ];

            // Process children
            const CallpathList&       extrap_children = extrapCallpaths[ i ]->getChildren();
            std::vector<cube::Cnode*> cube_children;
            int                       j;
            for ( j = 0; j < match->num_children(); j++ )
            {
                cube_children.push_back( match->get_child( j ) );
            }
            createCallpathMappingRec( extrap_children, cube_children );
        }
        else
        {
            WarningStream << "No match found for callpath '"
                          << extrapCallpaths[ i ]->getFullName()
                          << "'" << std::endl;
        }
    }
}

bool
CubeMapping::compareCallpath( const Callpath&    extrapCallpath,
                              const cube::Cnode& cubeCallpath )
{
    if ( m_region_extrap_to_cube[ extrapCallpath.getRegion() ] != cubeCallpath.get_callee() )
    {
        return false;
    }
    cube::Cnode* parent = NULL;
    if ( extrapCallpath.getParent() != NULL )
    {
        parent = m_callpath_extrap_to_cube[ extrapCallpath.getParent() ];
    }
    return parent == cubeCallpath.get_parent();
}

bool
CubeMapping::compareRegion( const EXTRAP::Region& extrapRegion,
                            const cube::Region&   cubeRegion )
{
    return extrapRegion.getName() == cubeRegion.get_name();
}

bool
CubeMapping::compareMetric( const EXTRAP::Metric& extrapMetric,
                            const cube::Metric&   cubeMetric )
{
    return extrapMetric.getName() == cubeMetric.get_uniq_name();
}
}; // namespace EXTRAP