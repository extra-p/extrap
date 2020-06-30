"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


import math

try:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
except ImportError:
    from PyQt5.QtGui import *  # @UnusedWildImport
    from PyQt5.QtCore import *  # @UnusedWildImport
    from PyQt5.QtWidgets import *  # @UnusedWildImport


class ParameterValueSlider(QWidget):
    def __init__(self, selectorWidget, parameter, parent):
        super(ParameterValueSlider, self).__init__(parent)
        self.selector_widget = selectorWidget
        self.parameter = parameter
        self.slider_update = True
        self.initUI()

    def initUI(self):
        self.grid = QGridLayout(self)
        self.setLayout(self.grid)

        name_label = QLabel(self.parameter.get_name(), self)
        name_label.setMinimumWidth(60)

        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(1)
        spinbox_max_val = 1073741824
        self.spinbox.setMaximum(spinbox_max_val)
        self.spinbox.setValue(0)
        self.spinbox.valueChanged.connect(self.spinboxValueChanged)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(1)
        slider_max_val = 30000
        self.slider.setMaximum(slider_max_val)
        self.slider.setValue(0)
        self.slider.valueChanged[int].connect(self.sliderValueChanged)

        self.grid.addWidget(name_label, 0, 0)
        self.grid.addWidget(self.spinbox, 0, 1)
        self.grid.addWidget(self.slider, 0, 2)
        self.grid.setContentsMargins(QMargins(0, 0, 0, 0))

    def spinboxValueChanged(self):
        value = self.spinbox.value()
        slider_value = int(math.log(value, 2) * 1000)
        self.slider_update = False
        self.slider.setValue(slider_value)
        self.selector_widget.main_widget.updateMinMaxValue()
        self.selector_widget.tree_model.valuesChanged()

    def sliderValueChanged(self, value):
        if self.slider_update:
            spinbox_value = int(2 ** (float(value) / 1000))
            self.spinbox.setValue(spinbox_value)
        self.slider_update = True

    def getValue(self):
        return self.spinbox.value()

    def getParameter(self):
        return self.parameter

    def clearRowLayout(self):
        if self.grid is not None:
            while self.grid.count():
                item = self.grid.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())
