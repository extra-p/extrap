"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""


from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport
from modelers.model_generator import ModelGenerator
from modelers import single_parameter
from modelers import multi_parameter


class ModelerWidget(QWidget):

    def __init__(self, mainWidget, parent):
        super(ModelerWidget, self).__init__(parent)
        self.main_widget = mainWidget
        # use mean or median measurement values to compute the models
        self.mean = True
        self.initUI()

    def initUI(self):
        grid = QGridLayout(self)
        self.setLayout(grid)

        self.model_name_edit = QLineEdit(self)
        self.model_name_edit.setPlaceholderText("ModelName")
        self.model_name_edit.setText("New Model")

        label = QLabel(self)
        label.setText("Name new Model:")
        grid.addWidget(label, 0, 0)
        grid.addWidget(self.model_name_edit, 0, 1)

        self.model_mean_radio = QRadioButton(self.tr("Model mean"))
        self.model_mean_radio.setChecked(True)
        grid.addWidget(self.model_mean_radio, 1, 0)

        self.model_median_radio = QRadioButton(self.tr("Model median"))
        grid.addWidget(self.model_median_radio, 2, 0)

        self.model_mean_median_radio_group = QButtonGroup(grid)
        self.model_mean_median_radio_group.addButton(self.model_mean_radio)
        self.model_mean_median_radio_group.addButton(self.model_median_radio)

        model_button = QPushButton(self)
        model_button.setText("Generate models")
        model_button.pressed.connect(self.remodel)
        grid.addWidget(model_button, 3, 0)
        grid.addWidget(QWidget(),4,0)
        grid.setRowStretch(4,1)

    def getName(self):
        return self.model_name_edit.text()

    def remodel(self):
        # set the modeler options
        if (self.model_mean_radio.isChecked()):
            self.mean = True
        elif (self.model_median_radio.isChecked()):
            self.mean = False

        # initialize model generator
        experiment = self.main_widget.getExperiment()

        model_generator = ModelGenerator(experiment,use_median=self.median)

        model_generator.set_name(self.model_name_edit.text())

        # create models from data
        model_generator.model_all()

        self.main_widget.getExperiment().modelAll(
            modeler, self.main_widget.getExperiment(), self.options)
        self.main_widget.selector_widget.updateModelList()
        self.main_widget.selector_widget.selectLastModel()
        self.main_widget.updateMinMaxValue()

        # must happen before 'valuesChanged' to update the color boxes
        self.main_widget.selector_widget.tree_model.valuesChanged()
        self.main_widget.update()

