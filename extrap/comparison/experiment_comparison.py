from typing import List, Sequence

from extrap.comparison.matcher import AbstractMatcher
from extrap.comparison.matches import IdentityMatches
from extrap.entities.experiment import Experiment
from extrap.entities.measurement import Measurement
from extrap.entities.model import Model
from extrap.modelers.abstract_modeler import AbstractModeler
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.exceptions import RecoverableError
from extrap.util.progress_bar import DUMMY_PROGRESS


class ComparisonError(RecoverableError):
    NAME = "Experiment Comparison Error"


class PlaceholderModeler(AbstractModeler):
    NAME = "<Placeholder>"

    def __init__(self, use_median: bool):
        super().__init__(use_median)

    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:
        raise NotImplementedError()


class ComparisonExperiment(Experiment):
    def __init__(self, exp1: Experiment, exp2: Experiment, matcher: AbstractMatcher):
        super(ComparisonExperiment, self).__init__()
        if exp1 is None and exp2 is None:
            self.compared_experiments: List[Experiment] = []
            return
        self.compared_experiments: List[Experiment] = [exp1, exp2]
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

        self.callpaths, self.callpaths_match = self._callpaths_from_tree(self.call_tree, self.call_tree_match)

        self.measurements = self._make_measurements(exp1.measurements, exp2.measurements)

        self.modelers_match = self.matcher.match_modelers(exp1.modelers, exp2.modelers)

        self.modelers = [self._make_model_generator(*match) for match in self.modelers_match]

    def _make_model_generator(self, name: str, modelers: Sequence[ModelGenerator]):
        mg = ModelGenerator(self, PlaceholderModeler(False), name, modelers[0].modeler.use_median)
        mg.models = {}
        return mg

    def _callpaths_from_tree(self, call_tree, match):
        return []

    def _make_measurements(self, measurements, measurements1):
        return {}
