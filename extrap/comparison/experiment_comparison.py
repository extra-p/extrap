from functools import reduce
from itertools import product
from typing import List, Sequence, Union, Mapping

import numpy as np

from extrap.comparison.matcher import AbstractMatcher
from extrap.comparison.matches import IdentityMatches, MutableAbstractMatches
from extrap.entities.calltree import Node
from extrap.entities.experiment import Experiment
from extrap.entities.functions import Function
from extrap.entities.hypotheses import Hypothesis
from extrap.entities.measurement import Measurement
from extrap.entities.model import Model
from extrap.entities.parameter import Parameter
from extrap.modelers.abstract_modeler import AbstractModeler
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.exceptions import RecoverableError
from extrap.util.progress_bar import DUMMY_PROGRESS

COMPARISON_NODE_NAME = '[Comparison]'


class ComparisonError(RecoverableError):
    NAME = "Experiment Comparison Error"


class PlaceholderModeler(AbstractModeler):
    NAME = "<Placeholder>"

    def __init__(self, use_median: bool):
        super().__init__(use_median)

    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:
        raise NotImplementedError()


class ComparisonFunction(Function):
    def __init__(self, functions):
        super().__init__()
        self.compound_terms = functions
        self.functions = functions

    def evaluate(self, parameter_value):
        """
        Evaluate the function according to the given value and return the result.
        """
        if isinstance(parameter_value, Mapping):
            pos_hash = self._pos_hash(sum(parameter_value.values()))
        else:
            pos_hash = self._pos_hash(parameter_value)
        function_value = [f.evaluate(parameter_value) for f in self.functions]
        if isinstance(parameter_value, np.ndarray):
            function_value = np.array(function_value)
            if len(parameter_value.shape) == 2:
                pos_hash = self._pos_hash(np.sum(parameter_value, axis=0))

        selected_function = pos_hash % len(self.functions)
        if isinstance(function_value, np.ndarray):
            if len(function_value.shape) == 2:
                return function_value[selected_function, :]
            else:
                return function_value[selected_function]
        return function_value[int(selected_function)]

    def _pos_hash(self, pos):
        pos = np.cast['int32'](pos)
        hash = 0
        for i in range(4):
            hash += pos & (0xFF << i * 8)
            hash += hash << 10
            hash ^= hash >> 6

        hash += hash << 3
        hash ^= hash >> 11
        hash += hash << 15
        return hash

    def to_string(self, *parameters: Union[str, Parameter]):
        """
        Return a string representation of the function.
        """
        return '(' + ', '.join(t.to_string(*parameters) for t in self.compound_terms) + ')'


class ComparisonModel(Model):
    def __init__(self, callpath, metric, models: Sequence[Model]):
        super().__init__(self._make_comparison_hypothesis(models), callpath, metric)
        self.models = models

    @staticmethod
    def _make_comparison_hypothesis(models):
        function = ComparisonFunction([m.hypothesis.function for m in models])
        hypothesis_type = type(models[0].hypothesis)
        hypothesis = hypothesis_type(function, models[0].hypothesis._use_median)
        hypothesis._RSS = sum(m.hypothesis.RSS for m in models)
        hypothesis._RE = reduce(lambda a, b: a * b, (m.hypothesis.RE for m in models)) ** (1 / len(models))
        hypothesis._rRSS = reduce(lambda a, b: a * b, (m.hypothesis.rRSS for m in models)) ** (1 / len(models))
        hypothesis._AR2 = reduce(lambda a, b: a * b, (m.hypothesis.AR2 for m in models)) ** (1 / len(models))
        hypothesis._SMAPE = reduce(lambda a, b: a * b, (m.hypothesis.SMAPE for m in models)) ** (1 / len(models))
        hypothesis._costs_are_calculated = True
        return hypothesis


class ComparisonExperiment(Experiment):
    def __init__(self, exp1: Experiment, exp2: Experiment, matcher: AbstractMatcher):
        super(ComparisonExperiment, self).__init__()
        if exp1 is None and exp2 is None:
            self.compared_experiments: List[Experiment] = []
            return
        self.compared_experiments: List[Experiment] = [exp1, exp2]
        self.experiment_names: List[str] = ['exp1', 'exp2']
        self.matcher = matcher
        self._do_comparison(exp1, exp2)

    def _do_comparison(self, exp1, exp2):
        if exp1.parameters != exp2.parameters:
            raise ComparisonError("Parameters do not match.")
        if exp1.coordinates != exp2.coordinates:
            raise ComparisonError("Coordinates do not match.")
        if exp1.scaling != exp2.scaling:
            raise ComparisonError("Scaling does not match.")

        self.parameters = exp1.parameters
        self.coordinates = exp1.coordinates
        self.scaling = exp1.scaling

        if exp1.metrics == exp2.metrics:
            self.metrics = exp1.metrics
            self.metrics_match = IdentityMatches(2, self.metrics)
        else:
            self.metrics, self.metrics_match = self.matcher.match_metrics(exp1.metrics, exp2.metrics)

        self.call_tree, self.call_tree_match = self.matcher.match_call_tree(exp1.call_tree, exp2.call_tree)

        self.measurements = self._make_measurements_and_update_call_tree(self.call_tree_match, exp1.measurements,
                                                                         exp2.measurements)

        self.callpaths, self.callpaths_match = self._callpaths_from_tree(self.call_tree_match)

        self.modelers_match = self.matcher.match_modelers(exp1.modelers, exp2.modelers)
        self.modelers = [self._make_model_generator(k, match) for k, match in self.modelers_match.items()]

    def _make_model_generator(self, name: str, modelers: Sequence[ModelGenerator]):
        mg = ModelGenerator(self, PlaceholderModeler(False), name, modelers[0].modeler.use_median)
        mg.models = {}
        for metric, source_metrics in self.metrics_match.items():
            for node, source_nodes in self.call_tree_match.items():
                models = []
                for i, (s_node, s_metric, s_modeler, s_name) in enumerate(
                        zip(source_nodes, source_metrics, modelers, self.experiment_names)):
                    if s_node is not None:
                        source_key = (s_node.path, s_metric)
                        if source_key in s_modeler.models:
                            models.append(s_modeler.models[source_key])
                if len(models) == 1:
                    mg.models[node.path, metric] = models[0]
                elif len(models) > 1:
                    mg.models[node.path, metric] = ComparisonModel(node.path, metric, models)
        return mg

    def _callpaths_from_tree(self, match):
        cp_match = {n.path: [n1.path if n1 else None for n1 in n12] for n, n12 in match.items()}
        return list(cp_match.keys()), cp_match

    def _make_measurements_and_update_call_tree(self, call_tree_match: MutableAbstractMatches[Node],
                                                *source_measurements):
        measurements = {}
        new_matches = {}
        for metric, source_metrics in self.metrics_match.items():
            for node, source_nodes in call_tree_match.items():
                origin_node = Node(COMPARISON_NODE_NAME, node.path.concat(COMPARISON_NODE_NAME))
                for i, (s_node, s_metric, s_measurement, s_name) in enumerate(
                        zip(source_nodes, source_metrics, source_measurements, self.experiment_names)):
                    source_key = (s_node.path, s_metric)
                    if source_key in s_measurement:
                        name = f"[{s_name}] {node.name}"
                        cp = origin_node.path.concat(name)
                        ct_node = Node(name, cp)
                        origin_node.add_child_node(ct_node)
                        new_match = [None] * len(source_measurements)
                        new_match[i] = s_node
                        new_matches[ct_node] = new_match
                        measurements[cp, metric] = s_measurement[source_key]
                if origin_node.childs:
                    node.childs.insert(0, origin_node)

        call_tree_match.update(new_matches)
        return measurements