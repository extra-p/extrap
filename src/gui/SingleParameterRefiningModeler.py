"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""

try:
    from PyQt4.QtGui import *
except ImportError:
    from PySide2.QtGui import *  # @UnusedWildImport
    from PySide2.QtWidgets import *  # @UnusedWildImport
from entities.modelgenerator import ModelGenerator


class SingleParameterRefiningModeler(QWidget):

    def __init__(self, modelerWidget, parent):
        super(SingleParameterRefiningModeler, self).__init__(parent)

        self.modeler_widget = modelerWidget

        self.initUI()

    def initUI(self):

        grid = QGridLayout(self)
        self.setLayout(grid)

        model_button = QPushButton(self)
        model_button.setText("Generate models")
        model_button.pressed.connect(self.remodel)

        grid.addWidget(model_button, 4, 0)

        widget = QWidget(self)
        widget.setMinimumHeight(self.height() - 120)
        grid.addWidget(widget, 5, 0)

    def getName(self):
        return "Single Parameter Model Generator"

    def remodel(self):
        # TODO: code not used anymore, just delete in the end

        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        experiment = model_generator.model_all(self.median)

        modeler = EXTRAP.SingleParameterRefiningModelGenerator()
        modeler.setEpsilon(0.01)

        #options = EXTRAP.ModelGeneratorOptions()

        # init the optins
        # options.setMinNumberPoints(5)
        # options.setUseAddPoints(False)
        # options.setNumberAddPoints(0)
        # options.setSinglePointsStrategy(EXTRAP.FIRST_POINTS_FOUND)
        # options.setMultiPointsStrategy(EXTRAP.INCREASING_COST)

        #self.modeler_widget.onGenerate(modeler, options)
