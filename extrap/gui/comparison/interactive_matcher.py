# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from asyncio import Event
from typing import Sequence, Mapping, Tuple, TYPE_CHECKING

from PySide2.QtWidgets import QWizardPage

from extrap.comparison.matchers import AbstractMatcher
from extrap.comparison.matches import MutableAbstractMatches, AbstractMatches
from extrap.comparison.metric_conversion import AbstractMetricConverter
from extrap.entities.calltree import CallTree, Node
from extrap.entities.metric import Metric
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.progress_bar import DUMMY_PROGRESS

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

    def match_metrics(self, *metric: Sequence[Metric], progress_bar=DUMMY_PROGRESS) -> Tuple[
        Sequence[Metric], AbstractMatches[Metric], Sequence[AbstractMetricConverter]]:
        matches = self._metrics_page.get_matches(*metric)
        return [], matches, []

    def match_call_tree(self, *call_tree: Sequence[CallTree], progress_bar=DUMMY_PROGRESS) -> Tuple[
        CallTree, MutableAbstractMatches[Node]]:
        matches = self._call_tree_page.get_matches(*call_tree)
        return CallTree(), matches

    def match_modelers(self, *mg: Sequence[ModelGenerator], progress_bar=DUMMY_PROGRESS) -> Mapping[
        str, Sequence[ModelGenerator]]:
        return self._model_set_page.get_matches(*mg)

    NAME = 'Interactive'
