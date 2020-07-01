"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


try:
    from PyQt4.QtGui import *
except ImportError:
    from PySide2.QtGui import *  # @UnusedWildImport
    from PySide2.QtWidgets import *  # @UnusedWildImport


class MultiParameterSparseModeler(QWidget):

    def __init__(self, modelerWidget, parent):
        super(MultiParameterSparseModeler, self).__init__(parent)
        self.auto_select = False
        self.additional_points = False
        self.number_additional_points = 0
        self.number_single_points = 5
        self.single_option = 0
        self.multi_option = 0
        self.modeler_widget = modelerWidget
        self.initUI()

    def initUI(self):
        grid = QGridLayout(self)
        self.setLayout(grid)

        # new options
        auto_checkbox = QCheckBox(self)
        auto_checkbox.setText("Auto select")
        auto_checkbox.setChecked(False)
        auto_checkbox.stateChanged.connect(self.auto_select_option)
        grid.addWidget(auto_checkbox, 0, 0)

        # single parameter experiment options
        self.frame1 = QGroupBox(self)
        self.frame1.setTitle("Single Parameter Experiment Options")
        self.frame1.setContentsMargins(20, 25, 0, 15)
        grid.addWidget(self.frame1, 1, 0)

        grid2 = QGridLayout()

        label1 = QLabel(self)
        label1.setText("Minimum number of points:")
        grid2.addWidget(label1, 0, 0)

        self.spinner1 = QSpinBox(self)
        self.spinner1.setMinimum(4)
        self.spinner1.setMaximum(5)
        self.spinner1.setValue(5)
        self.spinner1.setFixedWidth(60)
        self.spinner1.setEnabled(True)
        grid2.addWidget(self.spinner1, 0, 1)

        self.first_points = QRadioButton(self.tr("Use first option"))
        self.first_points.setChecked(True)
        grid2.addWidget(self.first_points, 1, 0)

        self.max_points = QRadioButton(self.tr("Use max. number of points"))
        grid2.addWidget(self.max_points, 2, 0)

        self.expensive_points = QRadioButton(self.tr("Use expensive points"))
        grid2.addWidget(self.expensive_points, 3, 0)

        self.cheapest_points = QRadioButton(self.tr("Use cheap points"))
        grid2.addWidget(self.cheapest_points, 4, 0)

        self.radio_group1 = QButtonGroup(grid2)
        self.radio_group1.addButton(self.first_points)
        self.radio_group1.addButton(self.max_points)
        self.radio_group1.addButton(self.expensive_points)
        self.radio_group1.addButton(self.cheapest_points)

        self.frame1.setLayout(grid2)

        # multi parameter experiment options
        self.frame2 = QGroupBox(self)
        self.frame2.setTitle("Multi Parameter Experiment Options")
        self.frame2.setContentsMargins(20, 25, 0, 15)
        grid.addWidget(self.frame2, 2, 0)

        grid3 = QGridLayout()

        self.points_checkbox = QCheckBox(self)
        self.points_checkbox.setText("Use additional points")
        self.points_checkbox.setChecked(False)
        self.points_checkbox.stateChanged.connect(
            self.additional_points_options)
        grid3.addWidget(self.points_checkbox, 0, 0)

        label2 = QLabel(self)
        label2.setText("Number of additional points:")
        grid3.addWidget(label2, 1, 0)

        self.points_number_edit = QSpinBox(self)
        self.points_number_edit.setMinimum(1)
        self.points_number_edit.setMaximum(1000)
        self.points_number_edit.setValue(1)
        self.points_number_edit.setFixedWidth(60)
        self.points_number_edit.setEnabled(False)
        grid3.addWidget(self.points_number_edit, 1, 1)

        self.option1 = QRadioButton(self.tr("Increasing cost"))
        self.option1.setChecked(True)
        self.option1.setEnabled(False)
        grid3.addWidget(self.option1, 2, 0)

        self.option2 = QRadioButton(self.tr("Decreasing cost"))
        self.option2.setEnabled(False)
        grid3.addWidget(self.option2, 3, 0)

        self.radio_group2 = QButtonGroup(grid)
        self.radio_group2.addButton(self.option1)
        self.radio_group2.addButton(self.option2)

        self.frame2.setLayout(grid3)

        model_button = QPushButton(self)
        model_button.setText("Generate models")
        model_button.pressed.connect(self.remodel)
        grid.addWidget(model_button, 3, 0)

        widget = QWidget(self)
        widget.setMinimumHeight(self.height() - 120)
        grid.addWidget(widget, 4, 0)

    def getName(self):
        return "Multi-Parameter Sparse Model Generator"

    def remodel(self):
        pass
        # TODO: use the multi parmaeter modeler when done
        # call model generator
        #modeler = EXTRAP.MultiParameterSparseModelGenerator()

        # set modeler options
        #self.number_single_points = self.spinner1.value()
        #self.number_additional_points = self.points_number_edit.value()
        # self.single_options()
        # self.multi_options()

        # TODO: delete this
        # create new options object
        #options = EXTRAP.ModelGeneratorOptions()

        # set strategy single
        # if(self.single_option == 0):
        #    options.setSinglePointsStrategy(EXTRAP.FIRST_POINTS_FOUND)
        # if(self.single_option == 1):
        #    options.setSinglePointsStrategy(EXTRAP.MAX_NUMBER_POINTS)
        # if(self.single_option == 2):
        #    options.setSinglePointsStrategy(EXTRAP.MOST_EXPENSIVE_POINTS)
        # if(self.single_option == 3):
        #    options.setSinglePointsStrategy(EXTRAP.CHEAPEST_POINTS)

        # set strategy multi
        # if(self.multi_option == 0):
        #    options.setMultiPointsStrategy(EXTRAP.INCREASING_COST)
        # if(self.multi_option == 1):
        #    options.setMultiPointsStrategy(EXTRAP.DECREASING_COST)

        # options.setMinNumberPoints(self.number_single_points)
        # options.setUseAddPoints(self.additional_points)
        # options.setNumberAddPoints(self.number_additional_points)

        # set auto select strategy if enabled
        # options.setUseAutoSelect(self.auto_select)

        # call modeling method
        #self.modeler_widget.onGenerate(modeler, options)

    def additional_points_options(self):
        if self.additional_points == False:
            self.additional_points = True
            self.points_number_edit.setEnabled(True)
            self.option1.setEnabled(True)
            self.option2.setEnabled(True)
        else:
            self.additional_points = False
            self.points_number_edit.setEnabled(False)
            self.option1.setEnabled(False)
            self.option2.setEnabled(False)

    def single_options(self):
        if self.first_points.isChecked() == True:
            self.single_option = 0
        elif self.max_points.isChecked() == True:
            self.single_option = 1
        elif self.expensive_points.isChecked() == True:
            self.single_option = 2
        elif self.cheapest_points.isChecked() == True:
            self.single_option = 3

    def multi_options(self):
        if self.option1.isChecked() == True:
            self.multi_option = 0
        elif self.option2.isChecked() == True:
            self.multi_option = 1

    def deactivate_options(self):
        self.spinner1.setEnabled(False)
        self.first_points.setEnabled(False)
        self.max_points.setEnabled(False)
        self.expensive_points.setEnabled(False)
        self.cheapest_points.setEnabled(False)
        self.points_number_edit.setEnabled(False)
        self.points_checkbox.setEnabled(False)
        self.option1.setEnabled(False)
        self.option2.setEnabled(False)

    def activate_options(self):
        self.spinner1.setEnabled(True)
        self.first_points.setEnabled(True)
        self.max_points.setEnabled(True)
        self.expensive_points.setEnabled(True)
        self.cheapest_points.setEnabled(True)
        self.points_checkbox.setEnabled(True)
        if self.additional_points == True:
            self.points_number_edit.setEnabled(True)
            self.option1.setEnabled(True)
            self.option2.setEnabled(True)

    def auto_select_option(self):
        if self.auto_select == False:
            self.auto_select = True
            self.deactivate_options()
        else:
            self.auto_select = False
            self.activate_options()
