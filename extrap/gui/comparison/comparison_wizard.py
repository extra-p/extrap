# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import warnings
from asyncio import Event
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Optional, Type, cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCommandLinkButton, QFileDialog, QFormLayout,
                               QLabel, QLineEdit, QSizePolicy, QSpacerItem,
                               QWizard, QWizardPage, QComboBox, QGridLayout)

from extrap.comparison import matchers
from extrap.comparison.experiment_comparison import ComparisonExperiment
from extrap.entities.parameter import Parameter
from extrap.fileio.experiment_io import ExperimentReader
from extrap.fileio.file_reader import FileReader, all_readers
from extrap.gui.comparison.interactive_matcher import InteractiveMatcher
from extrap.gui.components import file_dialog
from extrap.gui.components.wizard_pages import ProgressPage, ScrollAreaPage
from extrap.modelers.model_generator import ModelGenerator


class ComparisonWizard(QWizard):
    file_reader: Type[FileReader]
    file_name: str

    def __init__(self, experiment1, experiment2=None, name1='exp1', name2='exp2'):
        super().__init__()

        self.setWindowTitle("Compare With Other Experiment")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        if not experiment2:
            self.addPage(FileSelectionPage(self))
            self.addPage(FileLoadingPage(self))
        self.addPage(NamingPage(self))
        self.addPage(ModelSelectionPage(self))
        self.addPage(ParameterMappingPage(self))
        self.addPage(MatcherSelectionPage(self))
        self.comparing_page_id = self.addPage(ComparingPage(self))
        self.matcher = None
        self.exp_names = [name1, name2]
        self.experiment1 = experiment1
        self.experiment2 = experiment2
        self.experiment: Optional[ComparisonExperiment] = None
        self.model_mapping = {}
        self.parameter_mapping = {}
        self.is_cancelled = Event()
        self.rejected.connect(self.on_reject)

    def on_reject(self):
        self.is_cancelled.set()

    def back(self) -> None:
        self.restart()


class FileSelectionPage(ScrollAreaPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Select input format')
        layout = self.scroll_layout
        for reader in chain([ExperimentReader], all_readers.values()):
            def _(reader):
                btn = QCommandLinkButton(reader.GUI_ACTION.replace('&', ''))
                # btn.setDescription(reader.DESCRIPTION)
                btn.clicked.connect(
                    lambda: file_dialog.show(self, partial(self.open_file, reader), reader.DESCRIPTION,
                                             filter=reader.FILTER,
                                             file_mode=QFileDialog.FileMode.Directory if reader.LOADS_FROM_DIRECTORY else None))
                layout.addWidget(btn)

            _(reader)
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        self.scroll_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))

    def open_file(self, reader, name):
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        wizard.file_name = name
        wizard.exp_names[1] = Path(name).stem
        wizard.file_reader = reader
        wizard.next()

    def isComplete(self) -> bool:
        return False


class MatcherSelectionPage(ScrollAreaPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Select mapping provider')
        layout = self.scroll_layout
        for matcher in chain(matchers.all_matchers.values(), [InteractiveMatcher]):
            def _(matcher):
                btn = QCommandLinkButton(matcher.NAME)
                btn.setDescription(matcher.DESCRIPTION)
                btn.clicked.connect(lambda: self.select_matcher(matcher))
                layout.addWidget(btn)

            _(matcher)
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))

    def select_matcher(self, matcher):
        self.wizard().matcher = matcher
        self.wizard().next()

    def isComplete(self) -> bool:
        return False


class FileLoadingPage(ProgressPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Loading experiment')

    def do_process(self, pbar):
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        wizard.experiment2 = wizard.file_reader().read_experiment(wizard.file_name, pbar)
        if wizard.file_reader.GENERATE_MODELS_AFTER_LOAD:
            from extrap.gui.MainWidget import DEFAULT_MODEL_NAME
            ModelGenerator(wizard.experiment2, name=DEFAULT_MODEL_NAME).model_all(pbar)


class ComparingPage(ProgressPage):
    def __init__(self, parent):
        super().__init__(parent)
        self._override_next_id = None
        self.setTitle('Comparing experiments')

    def cleanupPage(self) -> None:
        self._override_next_id = None
        super().cleanupPage()

    def do_process(self, pbar):
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        if wizard.matcher == InteractiveMatcher:
            matcher = wizard.matcher(wizard)
        else:
            matcher = wizard.matcher()

        wizard.experiment = ComparisonExperiment(wizard.experiment1, wizard.experiment2, matcher=matcher)
        wizard.experiment.experiment_names = wizard.exp_names
        wizard.experiment.modelers_match = wizard.model_mapping
        wizard.experiment.parameter_mapping = wizard.parameter_mapping
        if wizard.matcher == InteractiveMatcher:
            self._override_next_id = matcher.determine_next_page_id()
        else:
            wizard.experiment.do_comparison(pbar)


class NamingPage(QWizardPage):
    def __init__(self, parent: ComparisonWizard):
        super().__init__(parent)
        self.setTitle('Name compared experiments')
        layout = QFormLayout(self)
        self.setLayout(layout)
        self._tb_name1 = QLineEdit()
        self._tb_name2 = QLineEdit()
        layout.addRow(QLabel("These names are used in Extra-P to show the sources of the compared models."))
        layout.addRow("Name of experiment 1", self._tb_name1)
        layout.addRow("Name of experiment 2", self._tb_name2)

    def initializePage(self) -> None:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        self._tb_name1.setText(wizard.exp_names[0])
        self._tb_name2.setText(wizard.exp_names[1])

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        wizard.exp_names[0] = self._tb_name1.text()
        if not wizard.exp_names[0]:
            wizard.exp_names[0] = 'exp1'
        wizard.exp_names[1] = self._tb_name2.text()
        if not wizard.exp_names[1]:
            wizard.exp_names[1] = 'exp2'
        if wizard.exp_names[0] == wizard.exp_names[1]:
            wizard.exp_names[0] += '1'
            wizard.exp_names[1] += '2'
        return super().validatePage()


class ModelSelectionPage(QWizardPage):
    def __init__(self, parent: ComparisonWizard):
        super().__init__(parent)
        self.setTitle('Choose Models for Comparison')
        self.layout = QFormLayout(self)
        self.setLayout(self.layout)
        self.model_lists = []

    def initializePage(self) -> None:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        for modeler in wizard.experiment1.modelers:
            if len(self.model_lists) < len(wizard.experiment1.modelers):
                model_list = QComboBox()
                model_list.addItems([model.name for model in wizard.experiment2.modelers])
                default_value = model_list.findText(modeler.name)
                if default_value >= 0:
                    model_list.setCurrentIndex(default_value)
                self.model_lists.append(model_list)
                self.layout.addRow(f"Compare {modeler.name} with: ", model_list)

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        exp1_models = wizard.experiment1.modelers
        exp2_models = wizard.experiment2.modelers
        wizard.model_mapping = {
            m.name: [m, exp2_models[self.model_lists[i].currentIndex()]] for i, m in enumerate(exp1_models)
        }

        return super().validatePage()


class ParameterMappingPage(QWizardPage):
    def __init__(self, parent: ComparisonWizard):
        super().__init__(parent)
        self.setTitle('Apply parameter mapping')
        self._layout = QGridLayout(self)
        self._param_lists = []
        self._name_edits = []
        self.setLayout(self._layout)

    def initializePage(self) -> None:
        self._clear_layout()
        self._name_edits.clear()
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        r_ctr = 0
        self._layout.addWidget(QLabel('Parameter Experiment 1'), r_ctr, 0)
        self._layout.addWidget(QLabel('Parameter Experiment 2'), r_ctr, 1)
        self._layout.addWidget(QLabel('New Parameter Name'), r_ctr, 2)
        r_ctr = 3
        for param in wizard.experiment1.parameters:
            self._layout.addWidget(QLabel(param.name + ':'), r_ctr, 0)
            param_list = QComboBox()
            param_list.addItems([param2.name for param2 in wizard.experiment2.parameters])
            default_value = param_list.findText(param.name)
            if default_value >= 0:
                param_list.setCurrentIndex(default_value)
            self._param_lists.append(param_list)
            self._layout.addWidget(param_list, r_ctr, 1)

            name_edit = QLineEdit(param.name)
            self._name_edits.append(name_edit)
            self._layout.addWidget(name_edit, r_ctr, 2)
            r_ctr += 1

    def _clear_layout(self):
        for i in reversed(range(self._layout.count())):
            widget = self._layout.itemAt(i).widget()
            self._layout.removeWidget(widget)
            widget.setParent(None)

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        old_param_names = [p.name for p in wizard.experiment1.parameters]
        new_param_names = [name_edit.text().strip() for name_edit in self._name_edits]

        for i, param_name in enumerate(new_param_names):
            if not param_name:
                warnings.warn("Parameters cannot be empty.")
                return False
            if old_param_names[i] != param_name \
                    and Parameter(param_name) in wizard.experiment1.parameters:
                warnings.warn(f"Parameter {param_name} already exists, "
                              "you cannot have two parameters with the same name.")
                return False
            if new_param_names.count(param_name) > 1:
                warnings.warn(f"Parameter {param_name} already exists, "
                              "you cannot have two parameters with the same name.")
                return False

        wizard.parameter_mapping = {
            name: [str(param1), cb_param2.currentText()] for name, param1, cb_param2 in
            zip(new_param_names, wizard.experiment1.parameters, self._param_lists)
        }

        return super().validatePage()
