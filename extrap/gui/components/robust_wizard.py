# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2025, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import json
import math
import warnings
from asyncio import Event
from functools import partial
from itertools import chain
from json import JSONDecodeError
from pathlib import Path
from typing import Optional, Type, cast

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QCommandLinkButton, QFileDialog, QFormLayout,
                               QLabel, QLineEdit, QSizePolicy, QSpacerItem,
                               QWizard, QWizardPage, QComboBox, QGridLayout, QButtonGroup, QHBoxLayout, QWidget,
                               QRadioButton, QListWidget, QGroupBox, QDoubleSpinBox, QPushButton, QSpinBox,
                               QListWidgetItem, QAbstractItemView, QVBoxLayout)

from extrap.fileio.experiment_io import ExperimentReader
from extrap.fileio.file_reader import FileReader, all_readers
from extrap.gui.components import file_dialog
from extrap.gui.components.dynamic_options import DynamicOptionsWidget
from extrap.gui.components.wizard_pages import ProgressPage, ScrollAreaPage
from extrap.util.dynamic_options import DynamicOptions
from extrap.modelers.robust import RobustModel
from extrap.entities.measurement import Measure

class RobustModelWizard(QWizard):
    file_reader: Type[FileReader]
    file_name: str

    def __init__(self, experiment1=None, experiment2=None, use_measure=Measure.MEAN):
        super().__init__()

        self.setWindowTitle("Create Robust Model With Time Experiment")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        self.exp_swc = experiment1
        self.exp_time = experiment2
        self.use_measure = use_measure

        self.is_cancelled = Event()

        self.file_reader = None
        self.rejected.connect(self.on_reject)

        if not experiment1:
            self.addPage(FileSelectionPage(self, is_swc=True))
            self.addPage(FileReaderOptionsPage(self))
            self.file_loading_page_id = self.addPage(FileLoadingPage(self, is_swc=True))
        if not experiment2:
            self.addPage(FileSelectionPage(self, is_swc=False))
            self.addPage(FileReaderOptionsPage(self))
            self.file_loading_page_id = self.addPage(FileLoadingPage(self, is_swc=False))

    def on_reject(self):
        self.is_cancelled.set()

    def back(self) -> None:
        self.restart()




class FileSelectionPage(ScrollAreaPage):
    def __init__(self, parent,is_swc):
        super().__init__(parent)
        if is_swc:
            self.setTitle('Select SWC experiment format')
        else:
            self.setTitle('Select time experiment format')
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
        wizard: RobustModelWizard = cast(RobustModelWizard, self.wizard())
        wizard.file_name = name
        wizard.file_reader = reader()
        wizard.next()

    def nextId(self) -> int:
        wizard: RobustModelWizard = cast(RobustModelWizard, self.wizard())
        if not isinstance(wizard.file_reader, DynamicOptions):
            return wizard.file_loading_page_id
        else:
            return super().nextId()

    def isComplete(self) -> bool:
        return False

class FileReaderOptionsPage(QWizardPage):
    def __init__(self, parent: RobustModelWizard):
        super().__init__(parent)
        self.setTitle('File import options')
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.dynamic_options_widget = DynamicOptionsWidget(self, None)
        layout.addWidget(self.dynamic_options_widget)

    def initializePage(self) -> None:
        wizard: RobustModelWizard = cast(RobustModelWizard, self.wizard())
        self.dynamic_options_widget.update_object_with_options(wizard.file_reader)

class FileLoadingPage(ProgressPage):
    def __init__(self, parent,is_swc):
        super().__init__(parent)
        self.setTitle('Loading experiment')
        self.is_swc = is_swc

    def do_process(self, pbar):
        wizard: RobustModelWizard = cast(RobustModelWizard, self.wizard())
        if self.is_swc:
            wizard.exp_swc = wizard.file_reader.read_experiment(wizard.file_name, pbar)
        else:
            wizard.exp_time = wizard.file_reader.read_experiment(wizard.file_name, pbar)



