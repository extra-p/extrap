# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from PySide6.QtWidgets import *  # @UnusedWildImport
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


#####################################################################


class AdvancedPlotWidget(QWidget):
    """This class represents a Widget that is used to show the graph on
       the extrap window.
    """

    #####################################################################

    def __init__(self, main_widget, parent, graphDisplayWindowClass):
        super(AdvancedPlotWidget, self).__init__(parent)
        self.grid = QGridLayout(self)
        self.main_widget = main_widget
        self.initUI()
        # Basic geometry constants
        self.max_x = 10
        self.max_y = 10
        self.font_size = 6
        self.setMouseTracking(True)
        self.graphDisplayWindowClass = graphDisplayWindowClass
        self.graphDisplayWindow = None
        self.toolbar = None

    def initUI(self):
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
            print("[EXTRA-P:] Error: Set maximum for axis other than X-axis.")

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
        if self.graphDisplayWindow is None:
            self.graphDisplayWindow = self.graphDisplayWindowClass(
                self, self.main_widget, width=5, height=4, dpi=100)
            self.toolbar = MyCustomToolbar(self.graphDisplayWindow, self)
            self.grid.addWidget(self.graphDisplayWindow)
            self.grid.addWidget(self.toolbar)
        else:
            self.graphDisplayWindow.redraw()

    @staticmethod
    def getNumAxis():
        """
          This function returns the number of axis. If its a 2-parameter model, it returns 2
        """
        return 2


class MyCustomToolbar(NavigationToolbar):
    """This class represents a Toolbar that is used to show the x,y, z value
       and save the figure.
    """
    toolitems = [toolitem for toolitem in NavigationToolbar.toolitems if
                 toolitem[0] in ('Home', 'Save')]
