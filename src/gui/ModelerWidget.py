"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""
from PySide2.QtCore import Slot
from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport

import modelers
from gui.ExpanderWidget import ExpanderWidget
from gui.ModelerOptionsWidget import ModelerOptionsWidget
from modelers.abstract_modeler import AbstractModeler
from modelers.model_generator import ModelGenerator
from modelers import single_parameter
from modelers import multi_parameter


class ModelerWidget(QWidget):

    def __init__(self, mainWidget, parent):
        super(ModelerWidget, self).__init__(parent)
        self.main_widget = mainWidget
        # use mean or median measurement values to compute the models
        self.mean = True
        self._modeler: AbstractModeler = ...

        self._model_selector = QComboBox(self)
        self._options_container = ExpanderWidget(self, 'Advanced Options')
        self._model_button = QPushButton(self)
        self.initUI()

    def initUI(self):
        grid = QGridLayout(self)
        self.setLayout(grid)

        self.model_name_edit = QLineEdit(self)
        self.model_name_edit.setPlaceholderText("ModelName")
        self.model_name_edit.setText("New Model")

        label = QLabel(self)
        label.setText("Model Name:")
        grid.addWidget(label, 0, 0)
        grid.addWidget(self.model_name_edit, 0, 1)

        self.model_mean_radio = QRadioButton(self.tr("Model Mean"))
        self.model_mean_radio.setChecked(True)
        grid.addWidget(self.model_mean_radio, 1, 0)

        self.model_median_radio = QRadioButton(self.tr("Model Median"))
        grid.addWidget(self.model_median_radio, 1, 1)

        self.model_mean_median_radio_group = QButtonGroup(grid)
        self.model_mean_median_radio_group.addButton(self.model_mean_radio)
        self.model_mean_median_radio_group.addButton(self.model_median_radio)

        self._model_selector.currentIndexChanged.connect(self._modeler_selected)
        self._model_selector.setEnabled(False)

        self._options_container.setEnabled(False)

        self._model_button.setText("Generate Models")
        self._model_button.pressed.connect(self.remodel)
        self._model_button.setEnabled(False)

        grid.addWidget(QWidget(), 2, 0, 1, 2)
        grid.addWidget(QLabel(self.tr('Model Generator:')), 3, 0, 1, 2)
        grid.addWidget(self._model_selector, 4, 0, 1, 2)
        grid.addWidget(self._options_container, 5, 0, 1, 2)
        grid.addWidget(self._model_button, 6, 0, 1, 2)
        grid.addWidget(QWidget(), 99, 0)
        grid.setRowStretch(99, 1)
        grid.setColumnStretch(1, 1)

    def getName(self):
        return self.model_name_edit.text()

    @Slot()
    def _modeler_selected(self):
        modeler_class = self._model_selector.currentData()
        if modeler_class:
            self._modeler = modeler_class()
            self._options_container.setContent(ModelerOptionsWidget(self._options_container, self._modeler))
        else:
            self._options_container.setContent(QWidget())
            self._options_container.toggle(True)
            self._options_container.setEnabled(False)

    def experimentChanged(self):
        experiment = self.main_widget.getExperiment()
        if experiment is None:
            self._model_selector.setEnabled(False)
            self._options_container.setEnabled(False)
            return

        self._model_selector.clear()
        self._model_selector.setCurrentText(None)
        if len(experiment.parameters) == 1:
            for name, modeler in single_parameter.all_modelers.items():
                self._model_selector.addItem(name, modeler)
        else:
            for name, modeler in multi_parameter.all_modelers.items():
                self._model_selector.addItem(name, modeler)
        self._model_selector.setCurrentText('Default')

        self._model_selector.setEnabled(True)
        self._model_button.setEnabled(True)

    def remodel(self):
        # set the modeler options
        if (self.model_mean_radio.isChecked()):
            use_median = False
        elif (self.model_median_radio.isChecked()):
            use_median = True

        # initialize model generator
        experiment = self.main_widget.getExperiment()
        if experiment is None:
            return

        model_generator = ModelGenerator(experiment, use_median=use_median, modeler=self._modeler)

        model_generator.set_name(self.model_name_edit.text())

        # create models from data
        model_generator.model_all()

        self.main_widget.selector_widget.updateModelList()
        self.main_widget.selector_widget.selectLastModel()
        self.main_widget.updateMinMaxValue()

        # must happen before 'valuesChanged' to update the color boxes
        self.main_widget.selector_widget.tree_model.valuesChanged()
        self.main_widget.update()
