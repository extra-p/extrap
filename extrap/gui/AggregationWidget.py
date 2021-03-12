# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import TYPE_CHECKING

from PySide2.QtCore import *
from PySide2.QtWidgets import *

from extrap.gui.ProgressWindow import ProgressWindow
from extrap.modelers.aggregation import all_aggregations

if TYPE_CHECKING:
    from extrap.gui.MainWidget import MainWidget


class AggregationWidget(QWidget):
    def __init__(self, main_widget: 'MainWidget', parent):
        super().__init__(parent)
        self.main_widget = main_widget
        self.setWindowTitle("Aggregation")

        layout = QFormLayout(self)

        self.setLayout(layout)

        label = QLabel("Strategy", self)

        self._cb_strategy = QComboBox(self)
        for name, aggregation in all_aggregations.items():
            self._cb_strategy.addItem(name, userData=aggregation)

        layout.addRow(label, self._cb_strategy)

        _btn_aggregation = QPushButton(text="Aggregate", parent=self)
        _btn_aggregation.clicked.connect(self.aggregate)

        layout.addRow(_btn_aggregation)

    @Slot()
    def aggregate(self):

        model_generator = self.main_widget.get_current_model_gen()
        if model_generator:
            aggregation_type = self._cb_strategy.currentData()
            aggregation = aggregation_type()
            with ProgressWindow(self, "Aggregating models") as pbar:
                model_generator.aggregate(aggregation, pbar)

                self.main_widget.selector_widget.updateModelList()
                self.main_widget.selector_widget.selectLastModel()
                self.main_widget.updateMinMaxValue()

                # must happen before 'valuesChanged' to update the color boxes
                self.main_widget.selector_widget.tree_model.valuesChanged()
                self.main_widget.update()
