"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


from matplotlib.figure import Figure

import numpy as np
import matplotlib.patches as mpatches


try:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from PyQt5.QtGui import *  # @UnusedWildImport
    from PyQt5.QtCore import *  # @UnusedWildImport
    from PyQt5.QtWidgets import *  # @UnusedWildImport
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

#####################################################################


class MeasurementPointsPlotWidget(QWidget):
    """This class represents a Widget that is used to show the graph on
       the main window.
    """
#####################################################################

    def __init__(self, main_widget, parent):
        super(MeasurementPointsPlotWidget, self).__init__(parent)
        self.main_widget = main_widget
        self.initUI(parent)
        self.set_initial_value()
        self.setMouseTracking(True)

    def initUI(self):
        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
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

    def setMaxX(self, maxX):
        """ This function sets the highest value of X that is being shown on x axis.
        """
        self.max_x = maxX

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
        graphDisplayWindow = GraphDisplayWindow(
            self, self.main_widget, width=5, height=4, dpi=100)
        self.toolbar = MyCustomToolbar(graphDisplayWindow, self)
        self.grid.addWidget(graphDisplayWindow, 0, 0)
        self.grid.addWidget(self.toolbar, 1, 0)

    def getNumAxis(self):
        """ 
            This function returns the number of axis. If its a 2-paramter model, it returns 2
        """
        return 2


class GraphDisplayWindow (FigureCanvas):
    def __init__(self, graphWidget, main_widget, width=5, height=4, dpi=100):

        self.graphWidget = graphWidget
        self.main_widget = main_widget
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.draw_figure()

    def draw_figure(self):
        """ 
          This function draws the graph
        """

        # Get data
# Get data
        selected_metric = self.main_widget.getSelectedMetric()
        selected_callpaths = self.main_widget.getSelectedCallpath()
        if not selected_callpaths:
            return
        model_list = list()
        for selected_callpath in selected_callpaths:
            model = self.main_widget.getCurrentModel(
                selected_metric, selected_callpath)
            if model != None:
                model_list.append(model)

        # Get font size for legend
        fontSize = self.graphWidget.getFontSize()

        # Get max x and max y value as a initial default value or a value provided by user
        maxX = self.graphWidget.getMaxX()
        maxY = self.graphWidget.getMaxY()

        # define min x and min y value
        lower_max = 2.0  # since we are drawing the plots with minimum axis value of 1 to avoid nan values , so the first max-value of parameter could be 2 to calcualte number of subdivisions
        if maxX < lower_max:
            maxX = lower_max
        if maxY < lower_max:
            maxY = lower_max

        # define grid parameters based on max x and max y value
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

        # Get the grid of the x and y values
        x = np.arange(1.0, maxX, pixelGap_x)
        y = np.arange(1.0, maxY, pixelGap_y)
        X, Y = np.meshgrid(x, y)

        if len(model_list) < 1:
            return

        # Get the z value for the x and y value
        z_List = list()
        Z_List = list()
        for model in model_list:
            function = model.hypothesis.function
            zs = np.array([self.calculate_z(x, y, function)
                           for x, y in zip(np.ravel(X), np.ravel(Y))])
            Z = zs.reshape(X.shape)
            z_List.append(zs)
            Z_List.append(Z)

        # Get the callpath color map
        dict_callpath_color = self.main_widget.get_callpath_color_map()

        # Set the x_label and y_label based on parameter selected.
        x_label = self.main_widget.data_display.getAxisParameter(0).name
        if x_label.startswith("_"):
            x_label = x_label[1:]
        y_label = self.main_widget.data_display.getAxisParameter(1).name
        if y_label.startswith("_"):
            y_label = y_label[1:]

        # 1 because we are going to show all the models in same plot
        number_of_subplots = 1

        # Draw all the selected models as surface plots and measuremnet point around them
        ax_all = self.fig.add_subplot(
            1, number_of_subplots, number_of_subplots, projection='3d')
        ax_all.mouse_init()
        ax_all.xaxis.major.formatter._useMathText = True
        ax_all.yaxis.major.formatter._useMathText = True
        ax_all.zaxis.major.formatter._useMathText = True
        ax_all.set_xlabel('\n' + x_label)
        ax_all.set_ylabel('\n' + y_label, linespacing=3.1)
        ax_all.set_zlabel(
            '\n' + self.main_widget.getSelectedMetric().name, linespacing=3.1)
        ax_all.set_title("Measurement Points")
        # ax_all.zaxis.set_major_locator(LinearLocator(10))
        # ax_all.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        for i in range(len(Z_List)):
            ax_all.plot_surface(X, Y, Z_List[i], color=dict_callpath_color[selected_callpaths[i]],
                                rstride=1, cstride=1, antialiased=False, alpha=0.1)

        # Gat base data for drawing points
        experiment = self.main_widget.getExperiment()
        parameter_x = self.main_widget.data_display.getAxisParameter(0)
        parameter_y = self.main_widget.data_display.getAxisParameter(1)

        for callpath in selected_callpaths:
            callpath_color = dict_callpath_color[callpath]
            points = experiment.getPoints(selected_metric, callpath)
            # for j in range( 0, points.size() ):
            for point in points:
                x = point.getParameterValue(parameter_x)
                y = point.getParameterValue(parameter_y)
                if x < maxX and y < maxY:
                    # Draw points
                    ax_all.scatter(x, y, point.getMean(),
                                   color=callpath_color,
                                   marker='o')
                    ax_all.scatter(x, y, point.getMedian(),
                                   color=callpath_color,
                                   marker='o')
                    ax_all.scatter(x, y, point.getMinimum(),
                                   color=callpath_color,
                                   marker='o')
                    ax_all.scatter(x, y, point.getMaximum(),
                                   color=callpath_color,
                                   marker='o')
                    # Draw connecting line
                    ax_all.plot([x, x], [y, y],
                                [point.getMinimum(), point.getMaximum()],
                                color=callpath_color)

        # draw legend
        patches = list()
        for key, value in dict_callpath_color.items():
            labelName = str(key.getRegion().name)
            if labelName.startswith("_"):
                labelName = labelName[1:]
            patch = mpatches.Patch(color=value, label=labelName)
            patches.append(patch)

        leg = ax_all.legend(handles=patches, fontsize=fontSize,
                            loc="upper right", bbox_to_anchor=(1, 1))
        if leg:
            leg.draggable()

    def getPixelGap(self, lowerlimit, upperlimit, numberOfPixels):
        """ 
           This function calculate the gap in pixels based on number of pixels and max value 
        """
        pixelGap = (upperlimit - lowerlimit)/numberOfPixels
        return pixelGap

    def calculate_z(self, x, y, function):
        """ 
           This function evaluates the function passed to it. 
        """
        parameter_value_list = self.main_widget.data_display.getValues()
        param1 = self.main_widget.data_display.getAxisParameter(0)
        param2 = self.main_widget.data_display.getAxisParameter(1)
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
