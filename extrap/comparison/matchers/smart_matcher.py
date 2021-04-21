from __future__ import annotations

import re
from typing import Sequence, Tuple, Mapping, List, Optional, Dict, TYPE_CHECKING

from extrap.comparison.matchers import AbstractMatcher
from extrap.comparison.matches import AbstractMatches, IdentityMatches, MutableAbstractMatches
from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree, Node
from extrap.entities.coordinate import Coordinate
from extrap.entities.functions import ConstantFunction
from extrap.entities.hypotheses import ConstantHypothesis
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.modelers.aggregation.abstract_binary_aggregation import BinaryAggregation
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.progress_bar import DUMMY_PROGRESS

if TYPE_CHECKING:
    from extrap.comparison.experiment_comparison import ComparisonExperiment


class SmartMatcher(AbstractMatcher):
    NAME = 'Smart Matcher'
    DESCRIPTION = 'Tries to find the common call tree and integrates the remaining call paths into this call tree.'

    def match_metrics(self, *metric: Sequence[Metric], progress_bar=DUMMY_PROGRESS) -> Tuple[
        Sequence[Metric], AbstractMatches[Metric]]:
        # preliminary, TODO we need to add new matches for hw counters, etc.
        metrics = [m for m in progress_bar(metric[0]) if m in metric[1]]
        return metrics, IdentityMatches(2, metrics)

    def match_call_tree(self, *call_tree: CallTree, progress_bar=DUMMY_PROGRESS):
        root = CallTree()
        matches = {}
        self._update_tree_paths(call_tree[0])
        self._update_tree_paths(call_tree[1])
        main_node0 = self._find_main(call_tree[0])
        main_node1 = self._find_main(call_tree[1])
        n = Node(self._normalizeName(main_node0.name), Callpath('main'))
        matches[n] = [main_node0, main_node1]
        root.add_child_node(n)
        self._merge_call_trees(n, main_node0, main_node1, 'main', matches, progress_bar)
        return root, matches

    def make_measurements_and_update_call_tree(self, experiment: ComparisonExperiment,
                                               call_tree_match: MutableAbstractMatches[Node],
                                               *source_measurements: Dict[Tuple[Callpath, Metric], List[Measurement]]):
        from extrap.comparison.experiment_comparison import COMPARISON_NODE_NAME
        measurements = {}
        new_matches = {}
        for metric, source_metrics in experiment.metrics_match.items():
            for node, source_nodes in call_tree_match.items():
                node: Node
                source_nodes: Sequence[Node]
                origin_node = Node(COMPARISON_NODE_NAME, node.path.concat(COMPARISON_NODE_NAME))
                for i, (s_node, s_metric, s_measurements, s_name) in enumerate(
                        zip(source_nodes, source_metrics, source_measurements, experiment.experiment_names)):

                    if s_node.path == Callpath.EMPTY:
                        print(s_node.name)

                    source_key = (s_node.path, s_metric)
                    name = f"[{s_name}] {node.name}"
                    agg_cp = origin_node.path.concat(name)
                    agg_cp.tags = {'comparison_part_agg': True}
                    part_agg_node = Node(name, agg_cp)
                    origin_node.add_child_node(part_agg_node)

                    part_cp = agg_cp.concat(node.name)
                    part_node = Node(node.name, part_cp)
                    part_agg_node.add_child_node(part_node)

                    t_measurements: Dict[Coordinate, Measurement] = {
                        coordinate: Measurement(coordinate, agg_cp, metric, 0) for coordinate in
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
                        self._add_subtree_and_merge_measurements(part_node, s_child, s_measurements, i,
                                                                 len(source_measurements), metric,
                                                                 t_measurements, new_matches, measurements)

                    new_match: List[Optional[Node]] = [None] * len(source_measurements)
                    new_match[i] = s_node
                    new_matches[part_node] = new_match
                    new_matches[part_agg_node] = new_match
                    measurements[part_cp, metric] = s_measurements
                    measurements[agg_cp, metric] = list(t_measurements.values())

                if origin_node.childs:
                    node.childs.insert(0, origin_node)
                    node.path.tags['comparison'] = True

        call_tree_match.update(new_matches)
        return measurements

    def match_modelers(self, *mg: Sequence[ModelGenerator], progress_bar=DUMMY_PROGRESS) -> Mapping[
        str, Sequence[ModelGenerator]]:
        mg_map = {m.name: m for m in mg[1]}
        return {m.name: [m, mg_map[m.name]] for m in mg[0] if m.name in mg_map}

    def _normalizeName(self, name):
        name = re.sub(r'\(.*?\)', '', name)
        name = re.sub(r'<.*?>', '', name)
        return name

    def _merge_call_trees(self, parent: Node, parent1: Node, parent2: Node, path, matches, progress_bar):
        for n1 in parent1.childs:
            normalized_name = self._normalizeName(n1.name)
            for n2 in parent2.childs:
                if self._normalizeName(n2.name) == normalized_name:
                    new_path = path + '->' + normalized_name
                    n = Node(normalized_name, Callpath(new_path))
                    matches[n] = [n1, n2]
                    parent.add_child_node(n)
                    self._merge_call_trees(n, n1, n2, new_path, matches, progress_bar)
        progress_bar.update()

    def _find_main(self, node: Node):
        if 'main' == node.name.strip().casefold():
            return node
        else:
            for child in node:
                n = self._find_main(child)
                if n:
                    return n
            return None

    def _update_tree_paths(self, node, parent_path=''):
        if node.name:
            if parent_path == "":
                path = node.name
            else:
                path = parent_path + '->' + node.name
            if node.path == Callpath.EMPTY:
                node.path = Callpath(path)
        else:
            path = parent_path
        for child in node:
            self._update_tree_paths(child, path)

    def _add_subtree_and_merge_measurements(self, ct_parent: Node, s_node: Node, s_measurements, i, total, metric,
                                            measurements_out: Dict[Coordinate, Measurement], new_matches, measurements):
        cp = ct_parent.path.concat(s_node.name)
        cp.tags = s_node.path.tags.copy()
        node = Node(s_node.name, cp)
        ct_parent.add_child_node(node)

        new_match: List[Optional[Node]] = [None] * total
        new_match[i] = s_node
        new_matches[node] = new_match

        c_measurements = s_measurements.get((s_node.path, metric), None)
        if c_measurements is not None:
            if not any(s_node.path.tags.get(tag, False) for tag in
                       ['agg_sum__not_calculable', 'gpu_overlap', 'agg_gpu_overlap']):
                for m in c_measurements:
                    measurements_out[m.coordinate].merge(m)
            measurements[cp, metric] = c_measurements

        for child in s_node:
            self._add_subtree_and_merge_measurements(node, child, s_measurements, i, total, metric, measurements_out,
                                                     new_matches, measurements)

    def make_model_generator(self, experiment: ComparisonExperiment, name: str, modelers: Sequence[ModelGenerator],
                             progress_bar):
        from extrap.comparison.experiment_comparison import PlaceholderModeler, ComparisonModel
        mg = ModelGenerator(experiment, PlaceholderModeler(False), name, modelers[0].modeler.use_median)
        mg.models = {}
        for metric, source_metrics in experiment.metrics_match.items():
            for node, source_nodes in experiment.call_tree_match.items():
                models = []
                progress_bar.update()
                if node.path.tags.get('comparison_part_agg', False):
                    for i, (s_node, s_metric, s_modeler, s_name) in enumerate(
                            zip(source_nodes, source_metrics, modelers, experiment.experiment_names)):
                        if s_node is not None:
                            node_subtree_root = node.childs[0]
                            allowed_s_childs = [experiment.call_tree_match[child][i]
                                                for child in node_subtree_root if
                                                child in experiment.call_tree_match]
                            t_node = Node(s_node.name, s_node.path, [c for c in s_node.childs if c in allowed_s_childs])
                            model = self.walk_nodes(t_node, s_modeler.models, metric,
                                                    path=s_node.path.name.rpartition('->')[0],
                                                    progress_bar=progress_bar)
                            if model:
                                models.append(model)

                elif node.path.tags.get('comparison', False):
                    for i, (s_node, s_metric, s_modeler, s_name) in enumerate(
                            zip(source_nodes, source_metrics, modelers, experiment.experiment_names)):
                        if s_node is not None:
                            t_node = Node(s_node.name, s_node.path, [c for c in s_node.childs])
                            for child in node:
                                if child in experiment.call_tree_match:
                                    s_child = experiment.call_tree_match[child][i]
                                    t_node.childs.remove(s_child)

                            model = self.walk_nodes(t_node, s_modeler.models, metric,
                                                    path=s_node.path.name.rpartition('->')[0],
                                                    progress_bar=progress_bar)
                            if model:
                                models.append(model)
                            else:
                                hypothesis = ConstantHypothesis(ConstantFunction(0), False)
                                hypothesis.compute_cost([])
                                models.append(Model(hypothesis))
                else:
                    for i, (s_node, s_metric, s_modeler, s_name) in enumerate(
                            zip(source_nodes, source_metrics, modelers, experiment.experiment_names)):
                        if s_node is not None and (s_node.path, s_metric) in s_modeler.models:
                            models.append(s_modeler.models[(s_node.path, s_metric)])

                if len(models) == 1:
                    mg.models[node.path, metric] = models[0]
                elif len(models) > 1:
                    mg.models[node.path, metric] = ComparisonModel(node.path, metric, models)
        return mg

    def walk_nodes(self, node: Node,
                   models: Dict[Tuple[Callpath, Metric], Model], metric: Metric, path='', progress_bar=DUMMY_PROGRESS,
                   agg_models=None):
        if agg_models is None:
            agg_models: List[Model] = []
        if node.name:
            if path == "":
                path = node.name
            else:
                path = path + '->' + node.name
        callpath = Callpath(path)
        key = (callpath, metric)
        if key in models:
            agg_models.append(models[key])
        else:
            progress_bar.total += 1
        for c in node:
            if not (c.path and 'agg_sum__not_calculable' in c.path.tags):
                self.walk_nodes(c, models, metric, path, progress_bar, agg_models)

        if not agg_models or (node.path and 'agg_sum__not_calculable' in node.path.tags):
            model = None
        elif len(agg_models) == 1:
            model = agg_models[0]
        else:
            measurements = BinaryAggregation.aggregate_measurements(agg_models, lambda a, b: a + b)
            model = BinaryAggregation.aggregate_model(agg_models, callpath, measurements, metric, lambda a, b: a + b,
                                                      'sum')
            model.callpaths = [m.callpath for m in agg_models]
            model.measurements = measurements

        progress_bar.update(1)
        return model
