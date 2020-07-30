"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""

from PySide2.QtWidgets import *  # @UnusedWildImport


class PlotTypeSelector(QDialog):

    def __init__(self, parent, dataDisplay):
        super(PlotTypeSelector, self).__init__(parent)
        self.valid = False
        self.dataDisplay = dataDisplay
        self.init_UI()

    def init_UI(self):
        self.setWindowTitle("Select the Plots")
        self.setFixedWidth(350)

        plotTypes = ['Line graph', 'Selected models in same surface plot', 'Selected models in different surface plots',
                     'Dominating models in a 3D Scatter plot', 'Max z as a single surface plot',
                     'Dominating models and max z as heat map', ' Selected models in contour plot',
                     'Selected models in interpolated contour plots', 'Measurement points']

        self.checkBoxes = [QCheckBox(plotType, self) for plotType in plotTypes]

        y_cord = 10
        for checkBox in self.checkBoxes:
            checkBox.move(10, y_cord)
            y_cord = y_cord + 30

        self.setFixedHeight(y_cord + 70)

        ok_button = QPushButton(self)
        ok_button.setText("OK")
        ok_button.move(165, y_cord + 30)
        ok_button.setFixedWidth(110)
        ok_button.pressed.connect(self.ok_pressed)

        cancel_button = QPushButton(self)
        cancel_button.setText("Cancel")
        cancel_button.move(40, y_cord + 30)
        cancel_button.setFixedWidth(100)
        cancel_button.pressed.connect(self.close)

    def ok_pressed(self):
        numberOfCheckBox = len(self.checkBoxes)
        selectedCheckBoxesIndex = list()

        for count in range(0, numberOfCheckBox):
            if self.checkBoxes[count].isChecked():
                selectedCheckBoxesIndex.append(count)

        self.valid = True
        self.close()
        self.dataDisplay.reloadTabs(selectedCheckBoxesIndex)
