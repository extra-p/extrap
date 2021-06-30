# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Sequence, Mapping, Tuple

from extrap.comparison.matchers import AbstractMatcher
from extrap.comparison.matches import AbstractMatches
from extrap.comparison.matches import IdentityMatches
from extrap.entities.calltree import CallTree, Node
from extrap.entities.metric import Metric
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.progress_bar import DUMMY_PROGRESS


class MinimumMatcher(AbstractMatcher):
    NAME = 'Minimum'
    DESCRIPTION = 'Selects the minimal common values in all comparisons.'

    def match_metrics(self, *metric: Sequence[Metric], progress_bar=DUMMY_PROGRESS) -> Tuple[
        Sequence[Metric], AbstractMatches[Metric]]:

        metrics = [m for m in progress_bar(metric[0], 0) if m in metric[1]]
        return metrics, IdentityMatches(2, metrics)

    def match_call_tree(self, *call_tree: CallTree, progress_bar=DUMMY_PROGRESS):
        root = CallTree()
        matches = {}
        self._merge_call_trees(root, call_tree[0], call_tree[1], matches, progress_bar)
        return root, matches

    def match_modelers(self, *mg: Sequence[ModelGenerator], progress_bar=DUMMY_PROGRESS) -> Mapping[
        str, Sequence[ModelGenerator]]:

        mg_map = {m.name: m for m in mg[1]}
        return {m.name: [m, mg_map[m.name]] for m in progress_bar(mg[0], 0) if m.name in mg_map}

    def _merge_call_trees(self, parent: Node, parent1: Node, parent2: Node, matches, progress_bar):
        for n1 in parent1.childs:
            n2 = parent2.find_child(n1.name)
            if n2:
                n = Node(n1.name, n1.path.copy())
                matches[n] = [n1, n2]
                parent.add_child_node(n)
                self._merge_call_trees(n, n1, n2, matches, progress_bar)
            progress_bar.update()
