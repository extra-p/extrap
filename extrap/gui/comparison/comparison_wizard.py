# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from asyncio import Event
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Optional, Type

from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QCommandLinkButton, QFileDialog, QFormLayout,
                               QLabel, QLineEdit, QSizePolicy, QSpacerItem,
                               QWizard, QWizardPage, QVBoxLayout, QComboBox)

from extrap.comparison import matchers
from extrap.comparison.experiment_comparison import ComparisonExperiment
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
        self.setWizardStyle(QWizard.ModernStyle)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        if not experiment2:
            self.addPage(FileSelectionPage(self))
            self.addPage(FileLoadingPage(self))
        self.addPage(NamingPage(self))
        self.addPage(ModelSelectionPage(self))
        self.addPage(MatcherSelectionPage(self))
        self.comparing_page_id = self.addPage(ComparingPage(self))
        self.matcher = None
        self.exp_names = [name1, name2]
        self.experiment1 = experiment1
        self.experiment2 = experiment2
        self.experiment: Optional[ComparisonExperiment] = None
        self.model_mapping = {}
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
                                             file_mode=QFileDialog.Directory if reader.LOADS_FROM_DIRECTORY else None))
                layout.addWidget(btn)

            _(reader)
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.scroll_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))

    def open_file(self, reader, name):
        wizard: ComparisonWizard = self.wizard()
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
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))

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
        wizard: ComparisonWizard = self.wizard()
        wizard.experiment2 = wizard.file_reader().read_experiment(wizard.file_name, pbar)
        if wizard.file_reader.GENERATE_MODELS_AFTER_LOAD:
            from extrap.gui.MainWidget import DEFAULT_MODEL_NAME
            ModelGenerator(wizard.experiment2, name=DEFAULT_MODEL_NAME).model_all(pbar)


class ComparingPage(ProgressPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Comparing experiments')
        
    def cleanupPage(self) -> None:
        self._override_next_id = None
        super().cleanupPage()

    def do_process(self, pbar):
        wizard: ComparisonWizard = self.wizard()
        if wizard.matcher == InteractiveMatcher:
            matcher = wizard.matcher(wizard)
        else:
            matcher = wizard.matcher()
        wizard.experiment = ComparisonExperiment(wizard.experiment1, wizard.experiment2, matcher=matcher)
        wizard.experiment.experiment_names = wizard.exp_names
        wizard.experiment.modelers_match = wizard.model_mapping
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
        wizard: ComparisonWizard = self.wizard()
        self._tb_name1.setText(wizard.exp_names[0])
        self._tb_name2.setText(wizard.exp_names[1])

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = self.wizard()
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
        #self.experiment_name1 = QLabel("Experiment1 Model")
        #self.experiment_name2 = QLabel("Experiment2 Model")
        # layout.addWidget(self.experiment_name1)
        # layout.addWidget(self.model_list1)
        # layout.addWidget(self.experiment_name2)
        # layout.addWidget(self.model_list2)
        self.model_lists = []
        
    def initializePage(self) -> None:
        wizard: ComparisonWizard = self.wizard()
        # self.model_list1.clear()
        # self.model_list2.clear()
        # self.experiment_name1.setText(f'Choose Model for {wizard.exp_names[0]}')
        # self.model_list1.addItems([model.name for model in wizard.experiment1.modelers])
        # self.experiment_name2.setText(f'Choose Model for {wizard.exp_names[1]}')
        # self.model_list2.addItems([model.name for model in wizard.experiment2.modelers])
        for modeler in wizard.experiment1.modelers:
            if len(self.model_lists) < len(wizard.experiment1.modelers):
                model_list = QComboBox()
                model_list.addItems([model.name for model in wizard.experiment2.modelers])
                self.model_lists.append(model_list)
                self.layout.addRow(f"Compare {modeler.name} with: ", model_list)

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = self.wizard()
        exp1_models = wizard.experiment1.modelers
        exp2_models = wizard.experiment2.modelers
        wizard.model_mapping = {
            m.name: [m, exp2_models[self.model_lists[i].currentIndex()]] for i, m in enumerate(exp1_models)
        }

        return super().validatePage()
        
        
