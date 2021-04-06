from __future__ import annotations
from asyncio import Event
from threading import Thread
from typing import Sequence, Mapping, Tuple, TYPE_CHECKING

from PySide2.QtCore import QEventLoop, QWaitCondition
from PySide2.QtWidgets import QWizardPage, QApplication

from extrap.comparison.matchers import AbstractMatcher
from extrap.comparison.matches import MutableAbstractMatches, AbstractMatches
from extrap.entities.calltree import CallTree, Node
from extrap.entities.metric import Metric
from extrap.gui.components.worker import Worker, MTWorker
from extrap.modelers.model_generator import ModelGenerator

if TYPE_CHECKING:
    from extrap.gui.comparison.comparison_wizard import ComparisonWizard


class MappingPage(QWizardPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.finished_event = Event()
        self.next_id = -1

    def get_matches(self, *data):
        return {}

    def nextId(self) -> int:
        return self.next_id


class MapMetricsPage(MappingPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle("Map metrics")

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = self.wizard()
        wizard.experiment.do_metrics_merge()
        return super().validatePage()


class MapCallTreePage(MappingPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle("Map call tree")

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = self.wizard()
        wizard.experiment.do_call_tree_merge()
        return super().validatePage()


class MapModelSetPage(MappingPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle("Map models")

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = self.wizard()
        wizard.experiment.do_model_set_merge()
        return super().validatePage()


class InteractiveMatcher(AbstractMatcher):
    def __init__(self, wizard: ComparisonWizard):
        self._wizard = wizard
        self._metrics_page = MapMetricsPage(self._wizard)
        self._metrics_page_id = wizard.addPage(self._metrics_page)

        self._call_tree_page = MapCallTreePage(self._wizard)
        self._call_tree_page_id = wizard.addPage(self._call_tree_page)
        self._metrics_page.next_id = self._call_tree_page_id

        self._model_set_page = MapModelSetPage(self._wizard)
        self._model_set_page_id = wizard.addPage(self._model_set_page)
        self._call_tree_page.next_id = self._model_set_page_id

    def determine_next_page_id(self):
        self._wizard.experiment.do_initial_checks()
        if not self._wizard.experiment.try_metric_merge():
            return self._metrics_page_id
        else:
            return self._call_tree_page_id

    def match_metrics(self, *metric: Sequence[Metric]) -> Tuple[Sequence[Metric], AbstractMatches[Metric]]:
        matches = self._metrics_page.get_matches(*metric)
        return [], matches

    def match_call_tree(self, *call_tree: Sequence[CallTree]) -> Tuple[CallTree, MutableAbstractMatches[Node]]:
        matches = self._call_tree_page.get_matches(*call_tree)
        return CallTree(), matches

    def match_modelers(self, *mg: Sequence[ModelGenerator]) -> Mapping[str, Sequence[ModelGenerator]]:
        return self._model_set_page.get_matches(*mg)

    NAME = 'Interactive'