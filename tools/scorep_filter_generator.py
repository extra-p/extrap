# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import argparse
import sys
from collections import defaultdict

import numpy as np
from pycubexr import CubexParser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', action='store', type=str)
    parser.add_argument('--metric', type=str, default='time')
    parser.add_argument('--metric-visits', type=str, default='visits')
    parser.add_argument('--percentile', type=float, default=0.25)
    parser.add_argument('--percentile-visits-filter', type=float, default=0.5)
    parser.add_argument('--exclude-parents', action='store_true', default=False)

    args = parser.parse_args()

    with CubexParser(args.path) as cubexFile:
        metric = cubexFile.get_metric_by_name(args.metric)
        metric_values = cubexFile.get_metric_values(metric, False)
        visits_metrics = cubexFile.get_metric_by_name(args.metric_visits)
        visits_values = cubexFile.get_metric_values(visits_metrics, False)

        filter_more = False
        percentile = args.percentile
        visits_filter_percentile = args.percentile_visits_filter

        total = 0
        visits = 0
        location_length = 0
        for cnode in cubexFile.get_root_cnodes():
            visits_raw = np.array(visits_values.cnode_values(cnode, False, True))
            times_raw = np.array(metric_values.cnode_values(cnode, False, True))
            mask = (times_raw != 0) & (visits_raw != 0)
            total += np.sum(times_raw[mask])
            visits = max(visits, np.sum(visits_raw[mask]))
            location_length = max(location_length, len(visits_raw[mask]))

        if total == 0:
            return

        print("Total runtime:", total, "Total visits:", visits)

        cnode_time = defaultdict(float)
        cnode_visits = defaultdict(float)

        for cnode in cubexFile.all_cnodes():
            if cnode.region.paradigm != "compiler":
                continue
            visits_raw = np.array(visits_values.cnode_values(cnode, True))
            times_raw = np.array(metric_values.cnode_values(cnode, True))
            mask = (times_raw != 0) & (visits_raw != 0)
            if np.any(mask):
                time = np.sum(times_raw[mask])
                visits = np.sum(visits_raw[mask])
            else:
                time = 0
                visits = 1

            cnode_time[cnode.id] += time
            cnode_visits[cnode.id] += visits

        region_time_per_visit = {r: cnode_time[r] / cnode_visits[r] for r in cnode_time}

        length = len(cnode_visits)

        sorted_cnode_time = sorted(cnode_time.items(), key=lambda x: x[1], reverse=True)
        sorted_cnode_time_per_visit = sorted(region_time_per_visit.items(), key=lambda x: x[1], reverse=True)
        sorted_cnode_visits = sorted(cnode_visits.items(), key=lambda x: x[1])

        included_cnodes = set()
        excluded_cnodes = set()

        included_cnodes.update([i for i, t in sorted_cnode_time_per_visit[:int(length * percentile)]])
        excluded_cnodes.update([i for i, t in sorted_cnode_time_per_visit[int(length * percentile):]])

        print(f"Filtering all regions with the {percentile} for {metric.name} "
              f"resulting in {len(included_cnodes)} included and "
              f"{len(excluded_cnodes)} excluded call-tree nodes.")

        significant_visits = [(i, k) for i, k in sorted_cnode_visits if k >= location_length]
        visits_threshold = significant_visits[int(len(significant_visits) * visits_filter_percentile)][1]

        print(
            f"Filtering all functions with more than {visits_threshold} visits, using {visits_filter_percentile} percentile.")

        important_time_cnodes = []
        for i, k in sorted_cnode_time[:int(length * percentile)]:
            cid = i
            while cid > 0:
                if cid in included_cnodes:
                    break
                visits = cnode_visits[cid]
                if visits <= visits_threshold:
                    important_time_cnodes.append((cid, k, visits))
                    break
                else:
                    parent_id = next((pid for pid in range(cid, 0, -1)
                                      if any(c.id == cid for c in cubexFile.get_cnode(pid).get_children())), None)
                    if parent_id is None:
                        break
                    else:
                        cid = parent_id

        print(f"Found {len(important_time_cnodes)} call-tree nodes with long runtime not included.")

        included_cnodes.update([i for i, t, v in important_time_cnodes])
        excluded_cnodes.difference_update([i for i, t, v in important_time_cnodes])

        if not args.exclude_parents:
            parent_nodes = set()
            for i in included_cnodes:
                cid = i
                while cid > 0:
                    if cid in parent_nodes:
                        break
                    parent_nodes.add(cid)
                    parent_id = next((pid for pid in range(cid, 0, -1)
                                      if any(c.id == cid for c in cubexFile.get_cnode(pid).get_children())), None)
                    if parent_id is None:
                        break
                    else:
                        cid = parent_id

            included_cnodes.update(parent_nodes)
            excluded_cnodes.difference_update(parent_nodes)

        if filter_more:
            included_cnodes, excluded_cnodes = excluded_cnodes, included_cnodes

        excluded_names = {cubexFile.get_cnode(i).region.mangled_name for i in excluded_cnodes}
        included_names = {cubexFile.get_cnode(i).region.mangled_name for i in included_cnodes}

        print(f"In total there are {len(included_names)} regions included and {len(excluded_names)} excluded.")

    filter_file = f"#Generated by Extra-P {' '.join(sys.argv[1:])}\n"
    filter_file += "SCOREP_REGION_NAMES_BEGIN\n"
    if len(excluded_names) <= len(included_names):
        filter_file += '  EXCLUDE MANGLED ' + '\n    '.join(name.replace(' ', '\\ ') for name in excluded_names)
    else:
        filter_file += '  EXCLUDE *\n'
        filter_file += '  INCLUDE MANGLED ' + '\n    '.join(name.replace(' ', '\\ ') for name in included_names)
    filter_file += '\nSCOREP_REGION_NAMES_END\n'
    print(filter_file)


if __name__ == '__main__':
    main()
