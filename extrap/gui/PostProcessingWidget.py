# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
from typing import TYPE_CHECKING

from PySide6.QtCore import *
from PySide6.QtWidgets import *

from extrap.gui.components.ProgressWindow import ProgressWindow
from extrap.gui.components.dynamic_options import DynamicOptionsWidget
from extrap.modelers.postprocessing import all_post_processes, PostProcess

if TYPE_CHECKING:
    from extrap.gui.MainWidget import MainWidget


class PostProcessingWidget(QWidget):
    def __init__(self, main_widget: 'MainWidget', parent):
        super().__init__(parent)
        self.main_widget = main_widget

        layout = QFormLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.setLayout(layout)

        label = QLabel("Strategy", self)

        self._cb_strategy = QComboBox(self)
        self.on_experiment_changed(main_widget.getExperiment())
        self._cb_strategy.currentIndexChanged.connect(self._select_strategy)

        layout.addRow(label, self._cb_strategy)

        self._strategy_options = DynamicOptionsWidget(self, None)
        layout.addRow(self._strategy_options)

        _btn_post_process = QPushButton(text="Process", parent=self)
        _btn_post_process.clicked.connect(self.process)

        layout.addRow(_btn_post_process)

    def on_experiment_changed(self, experiment):
        self._cb_strategy.clear()
        for name, post_process in all_post_processes.items():
            self._cb_strategy.addItem(name, userData=post_process(experiment))

    @Slot()
    def process(self):

        model_generator = self.main_widget.get_current_model_gen()
        if model_generator:
            post_process: PostProcess = copy.copy(self._cb_strategy.currentData())
            with ProgressWindow(self, "Processing models") as pbar:
                model_generator.post_process(post_process, pbar)

                if post_process.modifies_experiment:
                    self.main_widget.set_experiment(post_process.experiment)

                self.main_widget.selector_widget.updateModelList()
                self.main_widget.selector_widget.selectLastModel()
                self.main_widget.updateMinMaxValue()

                # must happen before 'valuesChanged' to update the color boxes
                self.main_widget.selector_widget.tree_model.valuesChanged()
                self.main_widget.update()

    @Slot(int)
    def _select_strategy(self, index: int):
        self._strategy_options.update_object_with_options(self._cb_strategy.currentData())
