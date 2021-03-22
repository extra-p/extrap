from typing import Sequence, Tuple, Mapping

from extrap.comparison.matchers import AbstractMatcher
from extrap.comparison.matches import AbstractMatches, IdentityMatches
from extrap.entities.calltree import CallTree, Node
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
        self._merge_call_trees(root, call_tree[0], call_tree[1], matches)
        return root, matches

    def match_modelers(self, *mg: Sequence[ModelGenerator]) -> Mapping[str, Sequence[ModelGenerator]]:
        mg_map = {m.name: m for m in mg[1]}
        return {m.name: [m, mg_map[m.name]] for m in mg[0] if m.name in mg_map}

    def _merge_call_trees(self, parent: Node, parent1: Node, parent2: Node, matches):
        for n1 in parent1.childs:
            n2 = parent2.find_child(n1.name)
            if n2:
                n = Node(n1.name, n1.path.copy())
                matches[n] = [n1, n2]
                parent.add_child_node(n)
                self._merge_call_trees(n, n1, n2, matches)