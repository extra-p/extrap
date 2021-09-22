# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from PySide2.QtCore import Slot, Qt
from PySide2.QtWidgets import *  # @UnusedWildImport

from extrap.entities.measurement import Measure
from extrap.gui.ModelerOptionsWidget import ModelerOptionsWidget
from extrap.gui.components.ExpanderWidget import ExpanderWidget
from extrap.gui.components.ProgressWindow import ProgressWindow
from extrap.modelers import multi_parameter
from extrap.modelers import single_parameter
from extrap.modelers.abstract_modeler import AbstractModeler
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.exceptions import RecoverableError


class ModelerWidget(QWidget):

    def __init__(self, mainWidget, parent):
        super(ModelerWidget, self).__init__(parent)
        self.main_widget = mainWidget
        # use mean or median measurement values to compute the models
        self.mean = True
        self._modeler: AbstractModeler = ...

        self._model_selector = QComboBox(self)
        self._options_container = ExpanderWidget(self, 'Advanced options')
        self._model_button = QPushButton(self)
        self.initUI()

    # noinspection PyAttributeOutsideInit
    def initUI(self):
        grid = QGridLayout(self)
        self.setLayout(grid)

        self.model_name_edit = QLineEdit(self)
        self.model_name_edit.setPlaceholderText("Model name")
        self.model_name_edit.setText("New Model")

        label = QLabel(self)
        label.setText("Model name:")
        grid.addWidget(label, 0, 0)
        grid.addWidget(self.model_name_edit, 0, 1)

        _measure_select_layout = QGridLayout()
        _measure_select_layout.setColumnStretch(3, 1)
        grid.addLayout(_measure_select_layout, 1, 0, 1, 2)

        self.model_mean_radio = QRadioButton("Model mean")
        self.model_mean_radio.setChecked(True)
        _measure_select_layout.addWidget(self.model_mean_radio, 1, 0)

        self.model_median_radio = QRadioButton("Model median")
        _measure_select_layout.addWidget(self.model_median_radio, 1, 1)

        self._model_other_radio = QRadioButton()
        _measure_select_layout.addWidget(self._model_other_radio, 1, 2)

        self._model_other_select = QComboBox()
        self._model_other_select.addItems(
            [m.name.title() for m in Measure.choices() if m != Measure.MEDIAN and m != Measure.MEAN])
        self._model_other_select.activated.connect(lambda _: self._model_other_radio.setChecked(True))
        _measure_select_layout.addWidget(self._model_other_select, 1, 3)

        self._model_measure_radio_group = QButtonGroup(grid)
        self._model_measure_radio_group.addButton(self.model_mean_radio)
        self._model_measure_radio_group.addButton(self.model_median_radio)
        self._model_measure_radio_group.addButton(self._model_other_radio)

        self._model_selector.currentIndexChanged.connect(self._modeler_selected)
        self._model_selector.setEnabled(False)

        self._options_container.setEnabled(False)

        self._model_button.setText("Generate models")
        self._model_button.clicked.connect(self.remodel)
        self._model_button.setEnabled(False)

        grid.addWidget(QWidget(), 2, 0, 1, 2)
        grid.addWidget(QLabel('Model generator:'), 3, 0, 1, 2)
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
        self._model_selector.setCurrentText('')
        if len(experiment.parameters) == 1:
            for i, (name, modeler) in enumerate(single_parameter.all_modelers.items()):
                self._model_selector.addItem(name, modeler)
                self._model_selector.setItemData(i, modeler.DESCRIPTION, Qt.ToolTipRole)
        else:
            for i, (name, modeler) in enumerate(multi_parameter.all_modelers.items()):
                self._model_selector.addItem(name, modeler)
                self._model_selector.setItemData(i, modeler.DESCRIPTION, Qt.ToolTipRole)
        self._model_selector.setCurrentText('Default')
        self._model_selector.currentData()

        self._model_selector.setEnabled(True)
        self._model_button.setEnabled(True)

    @Slot()
    def remodel(self):
        # set the modeler options
        if self.model_mean_radio.isChecked():
            measure = Measure.MEAN
        elif self.model_median_radio.isChecked():
            measure = Measure.MEDIAN
        elif self._model_other_radio.isChecked():
            measure = Measure.from_str(self._model_other_select.currentText())
        else:
            raise RecoverableError('No measure selected.')

        # initialize model generator
        experiment = self.main_widget.getExperiment()
        if experiment is None:
            return

        model_generator = ModelGenerator(experiment, use_measure=measure, modeler=self._modeler)

        model_generator.name = self.model_name_edit.text()
        # print(QCoreApplication.hasPendingEvents())
        with ProgressWindow(self.main_widget, 'Generating models') as pbar:
            # create models from data
            model_generator.model_all(pbar)

            self.main_widget.selector_widget.updateModelList()
            self.main_widget.selector_widget.selectLastModel()
            self.main_widget.updateMinMaxValue()

            # must happen before 'valuesChanged' to update the color boxes
            self.main_widget.selector_widget.tree_model.valuesChanged()
            self.main_widget.update()
