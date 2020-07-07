"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""

from matplotlib.figure import Figure

import numpy as np
import matplotlib.patches as mpatches
from abc import abstractmethod

from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtCore import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


#####################################################################


class AdvancedPlotWidget(QWidget):
    """This class represents a Widget that is used to show the graph on
       the main window.
    """

    #####################################################################

    def __init__(self, main_widget, parent, graphDisplayWindowClass):
        super(AdvancedPlotWidget, self).__init__(parent)
        self.main_widget = main_widget
        self.initUI()
        self.set_initial_value()
        self.setMouseTracking(True)
        self.graphDisplayWindowClass = graphDisplayWindowClass
        self.graphDisplayWindow = None

    def initUI(self):
        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(0)
        self.setMinimumWidth(300)
        self.setMinimumHeight(300)
        self.show()

    def setMax(self, axis, maxValue):
        """ This function sets the highest value of x and y that is being shown on x axis and y axis.
        """
        if axis == 0:
            self.main_widget.data_display.setMaxValue(0, maxValue)
            self.max_x = maxValue
        elif axis == 1:
            self.main_widget.data_display.setMaxValue(1, maxValue)
            self.max_y = maxValue

        else:
            print("[EXTRAP:] Error: Set maximum for axis other than X-axis.")

    def getMaxX(self):
        """ 
           This function returns the highest value of x that is being shown on x axis.
        """
        return self.max_x

    def setMaxY(self, maxY):
        """ This function sets the highest value of Y that is being shown on x axis.
        """
        self.max_y = maxY

    def getMaxY(self):
        """ 
           This function returns the highest value of Y that is being shown on x axis.
        """
        return self.max_y

    def setFontSize(self, font_size):
        """ This function sets the font size of the legend.
        """
        self.font_size = font_size

    def getFontSize(self):
        """ 
           This function returns the font size of the legend.
        """
        return self.font_size

    def set_initial_value(self):
        """ 
          This function sets the initial value for different parameters required for graph.
        """
        # Basic geometry constants
        self.max_x = 10
        self.max_y = 10
        self.font_size = 6

    def drawGraph(self):
        """ 
            This function is being called by paintEvent to draw the graph 
        """
        # Get the font size as selected by the user and set it
        self.font_size = int(self.main_widget.getFontSize())
        # Call the 3D function display window
        if self.graphDisplayWindow is not None:
            self.grid.removeWidget(self.graphDisplayWindow)
            self.grid.removeWidget(self.toolbar)
            self.graphDisplayWindow.deleteLater()
            self.toolbar.deleteLater()

        self.graphDisplayWindow = self.graphDisplayWindowClass(
            self, self.main_widget, width=5, height=4, dpi=100)
        self.toolbar = MyCustomToolbar(self.graphDisplayWindow, self)
        self.grid.addWidget(self.graphDisplayWindow)
        self.grid.addWidget(self.toolbar)

    def getNumAxis(self):
        """ 
          This function returns the number of axis. If its a 2-paramter model, it returns 2
        """
        return 2


class GraphDisplayWindow(FigureCanvas):
    def __init__(self, graphWidget, main_widget, width=5, height=4, dpi=100):
        self.graphWidget = graphWidget
        self.main_widget = main_widget
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        super().setSizePolicy(QSizePolicy.Expanding,
                              QSizePolicy.Expanding)
        super().updateGeometry()
        self.draw_figure()
        self.fig.tight_layout()

    @abstractmethod
    def draw_figure(self):
        ...

    def _calculate_grid_parameters(self, maxX, maxY):
        if (maxX < 10):
            numberOfPixels_x = 45
            pixelGap_x = self.getPixelGap(0, maxX, numberOfPixels_x)
        elif maxX >= 10 and maxX <= 1000:
            numberOfPixels_x = 40
            pixelGap_x = self.getPixelGap(0, maxX, numberOfPixels_x)
        elif maxX > 1000 and maxX <= 1000000000:
            numberOfPixels_x = 15
            pixelGap_x = self.getPixelGap(0, maxX, numberOfPixels_x)
        else:
            numberOfPixels_x = 5
            pixelGap_x = self.getPixelGap(0, maxX, numberOfPixels_x)
        if (maxY < 10):
            numberOfPixels_y = 45
            pixelGap_y = self.getPixelGap(0, maxY, numberOfPixels_y)
        elif maxY >= 10 and maxY <= 1000:
            numberOfPixels_y = 40
            pixelGap_y = self.getPixelGap(0, maxY, numberOfPixels_y)
        elif maxY > 1000 and maxY <= 1000000000:
            numberOfPixels_y = 15
            pixelGap_y = self.getPixelGap(0, maxY, numberOfPixels_y)
        else:
            numberOfPixels_y = 5
            pixelGap_y = self.getPixelGap(0, maxY, numberOfPixels_y)
        return pixelGap_x, pixelGap_y

    def getPixelGap(self, lowerlimit, upperlimit, numberOfPixels):
        """ 
           This function calculate the gap in pixels based on number of pixels and max value 
        """
        pixelGap = (upperlimit - lowerlimit) / numberOfPixels
        return pixelGap

    def calculate_z(self, x, y, function):
        """ 
           This function evaluates the function passed to it. 
        """
        parameter_value_list = self.main_widget.data_display.getValues()
        param1 = self.main_widget.data_display.getAxisParameter(0).id
        param2 = self.main_widget.data_display.getAxisParameter(1).id
        parameter_value_list[param1] = x
        parameter_value_list[param2] = y
        z_value = function.evaluate(parameter_value_list)
        return z_value


class MyCustomToolbar(NavigationToolbar):
    """This class represents a Toolbar that is used to show the x,y, z value
       and save the figure.
    """
    toolitems = [toolitem for toolitem in NavigationToolbar.toolitems if
                 toolitem[0] in ('Home1', 'Save')]
