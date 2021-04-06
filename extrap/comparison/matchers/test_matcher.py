import re
from typing import Sequence, Tuple, Mapping, List, Optional, Dict

from extrap.comparison.experiment_comparison import COMPARISON_NODE_NAME, ComparisonExperiment
from extrap.comparison.matchers import AbstractMatcher
from extrap.comparison.matches import AbstractMatches, IdentityMatches, MutableAbstractMatches
from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree, Node
from extrap.entities.coordinate import Coordinate
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.modelers.model_generator import ModelGenerator


class TestMatcher(AbstractMatcher):
    NAME = 'Test'

    def match_metrics(self, *metric: Sequence[Metric]) -> Tuple[Sequence[Metric], AbstractMatches[Metric]]:
        metrics = [m for m in metric[0] if m in metric[1]]
        return metrics, IdentityMatches(2, metrics)

    def match_call_tree(self, *call_tree: CallTree):
        root = CallTree()
        matches = {}

        main_node0 = self._find_main(call_tree[0])
        main_node1 = self._find_main(call_tree[1])
        n = Node(self._normalizeName(main_node0.name), main_node0.path.copy())
        matches[n] = [main_node0, main_node1]
        root.add_child_node(n)
        self._merge_call_trees(n, main_node0, main_node1, 'main', matches)
        return root, matches

    def make_measurements_and_update_call_tree(self, experiment: ComparisonExperiment,
                                               call_tree_match: MutableAbstractMatches[Node],
                                               *source_measurements: Dict[Tuple[Callpath, Metric], List[Measurement]]):
        measurements = {}
        new_matches = {}
        for metric, source_metrics in experiment.metrics_match.items():
            for node, source_nodes in call_tree_match.items():
                node: Node
                source_nodes: Sequence[Node]
                origin_node = Node(COMPARISON_NODE_NAME, node.path.concat(COMPARISON_NODE_NAME))
                for i, (s_node, s_metric, s_measurements, s_name) in enumerate(
                        zip(source_nodes, source_metrics, source_measurements, experiment.experiment_names)):
                    source_key = (s_node.path, s_metric)
                    name = f"[{s_name}] {node.name}"
                    cp = origin_node.path.concat(name)
                    ct_node = Node(name, cp)
                    origin_node.add_child_node(ct_node)

                    t_measurements: Dict[Coordinate, Measurement] = {
                        coordinate: Measurement(coordinate, cp, metric, 0) for coordinate in
                        experiment.coordinates}
                    if source_key in s_measurements:
                        for m in s_measurements[source_key]:
                            new_m = t_measurements.get(m.coordinate)
                            if new_m:
                                new_m.merge(m)

                    # find children that are not part of the comparison
                    children = s_node.childs.copy()
                    for child in node:
                        if child in call_tree_match:
                            s_child = call_tree_match[child][i]
                            children.remove(s_child)

                    for s_child in children:
                        self._add_subtree_and_merge_measurements(ct_node, s_child, s_measurements, i, metric,
                                                                 t_measurements, new_matches, measurements)

                    new_match: List[Optional[Node]] = [None] * len(source_measurements)
                    new_match[i] = s_node
                    new_matches[ct_node] = new_match
                    measurements[cp, metric] = t_measurements

                if origin_node.childs:
                    node.childs.insert(0, origin_node)

        call_tree_match.update(new_matches)
        return measurements

    def match_modelers(self, *mg: Sequence[ModelGenerator]) -> Mapping[str, Sequence[ModelGenerator]]:
        mg_map = {m.name: m for m in mg[1]}
        return {m.name: [m, mg_map[m.name]] for m in mg[0] if m.name in mg_map}

    def _normalizeName(self, name):
        name = re.sub(r'\(.*?\)', '', name)
        name = re.sub(r'<.*?>', '', name)
        return name

    def _merge_call_trees(self, parent: Node, parent1: Node, parent2: Node, path, matches):
        for n1 in parent1.childs:
            normalized_name = self._normalizeName(n1.name)
            for n2 in parent2.childs:
                if self._normalizeName(n2.name) == normalized_name:
                    new_path = path + '->' + normalized_name
                    n = Node(normalized_name, Callpath(new_path))
                    matches[n] = [n1, n2]
                    parent.add_child_node(n)
                    self._merge_call_trees(n, n1, n2, new_path, matches)

    def _find_main(self, node: Node):
        if 'main' == node.name.strip().casefold():
            return node
        else:
            for child in node:
                n = self._find_main(child)
                if n:
                    return n
            return None

    def _add_subtree_and_merge_measurements(self, ct_node: Node, s_child: Node, s_measurements, i, metric,
                                            measurements_out: Dict[Coordinate, Measurement], new_matches, measurements):
        cp = ct_node.path.concat(s_child.name)
        cp.tags = s_child.path.tags.copy()
        child = Node(s_child.name, cp)
        ct_node.add_child_node(child)

        new_match: List[Optional[Node]] = [None] * len(s_measurements)
        new_match[i] = s_child
        new_matches[ct_node] = new_match

        c_measurements = s_measurements.get((ct_node.path, metric), None)
        if c_measurements is not None:
            for m in c_measurements:
                measurements_out[m.coordinate].merge(m)
            measurements[cp, metric] = measurements_out

        for child2 in s_child:
            self._add_subtree_and_merge_measurements(child, child2, s_measurements, i, metric, measurements_out,
                                                     new_matches, measurements)
