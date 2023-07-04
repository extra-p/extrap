# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from PySide6.QtWidgets import *  # @UnusedWildImport


class PlotTypeSelector(QDialog):

    def __init__(self, parent, dataDisplay):
        super(PlotTypeSelector, self).__init__(parent)
        self.dataDisplay = dataDisplay
        self.init_UI()

    def init_UI(self):
        self.setWindowTitle("Select the Plots")
        self.setFixedWidth(350)

        plotTypes = ['Line graph', 'Selected models in same surface plot', 'Selected models in different surface plots',
                     'Dominating models in a 3D Scatter plot', 'Max z as a single surface plot',
                     'Dominating models and max z as heat map', 'Selected models in contour plot',
                     'Selected models in interpolated contour plots', 'Measurement points']

        layout = QVBoxLayout()

        self.checkBoxes = []
        for plotType in plotTypes:
            check_box = QCheckBox(plotType, self)
            self.checkBoxes.append(check_box)
            layout.addWidget(check_box)

        dialog_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                                          self)
        dialog_buttons.accepted.connect(self.accept)
        dialog_buttons.rejected.connect(self.reject)
        layout.addWidget(dialog_buttons)

        self.setLayout(layout)

    def accept(self):
        numberOfCheckBox = len(self.checkBoxes)
        selectedCheckBoxesIndex = list()

        for count in range(0, numberOfCheckBox):
            if self.checkBoxes[count].isChecked():
                selectedCheckBoxesIndex.append(count)

        self.dataDisplay.reloadTabs(selectedCheckBoxesIndex)
        super().accept()
