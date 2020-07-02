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
from gui.AdvancedPlotWidget import GraphDisplayWindow
# This class was developed as a first approach to show dominating models in heatmap
# not used any more
# HeatMapGraphWidget in used for that purpose
# but still keeping it since it has a different approach implemented

from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtCore import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

#####################################################################


class DominatingFunctionsAsSingleScatterPlot(GraphDisplayWindow):

    def __init__(self, graphWidget, main_widget, width=5, height=4, dpi=100):
        super().__init__(graphWidget, main_widget, width, height, dpi)

    def draw_figure(self):
        """ 
          This function draws the graph
        """
        # Get data
        selected_metric = self.main_widget.getSelectedMetric()
        selected_callpaths = self.main_widget.getSelectedCallpath()
        if not selected_callpaths:
            return
        model_set = self.main_widget.getCurrentModel().models
        model_list = list()
        for selected_callpath in selected_callpaths:
            model = model_set[selected_callpath.path, selected_metric]
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

        # Get the z value for the x and y value
        Z_List = list()
        z_List = list()
        for model in model_list:
            function = model.hypothesis.function
            zs = np.array([self.calculate_z(x, y, function)
                           for x, y in zip(np.ravel(X), np.ravel(Y))])
            Z = zs.reshape(X.shape)
            z_List.append(zs)
            Z_List.append(Z)

        # Get the callpath color map
        dict_callpath_color = self.main_widget.get_callpath_color_map()

        # calculate max_z value
        max_z_val = z_List[0][0]
        color_for_max_z = dict_callpath_color[selected_callpaths[0]]
        max_z_list = list()
        max_color_list = list()

        for i in range(len(z_List[0])):
            max_z_val = z_List[0][i]
            for j in range(len(model_list)):
                if(z_List[j][i] > max_z_val):
                    max_z_val = z_List[j][i]
                    #func_with_max_z = model_list[j]
                    color_for_max_z = dict_callpath_color[selected_callpaths[j]]
            max_z_list.append(max_z_val)
            max_color_list.append(color_for_max_z)

        max_Z_List = np.array(max_z_list).reshape(X.shape)
        max_Color_List = np.array(max_color_list).reshape(X.shape)

        # Set the x_label and y_label based on parameter selected.
        x_label = self.main_widget.data_display.getAxisParameter(0).name
        if x_label.startswith("_"):
            x_label = x_label[1:]
        y_label = self.main_widget.data_display.getAxisParameter(1).name
        if y_label.startswith("_"):
            y_label = y_label[1:]

        # Draw the graph showing the max z value
        number_of_subplots = 1
        ax = self.fig.add_subplot(
            1, number_of_subplots, number_of_subplots, projection='3d')
        ax.mouse_init()
        ax.xaxis.major.formatter._useMathText = True
        ax.yaxis.major.formatter._useMathText = True
        ax.zaxis.major.formatter._useMathText = True
        ax.get_xaxis().get_major_formatter().set_scientific(True)
        for (x, y, z, colour) in zip(X, Y, max_Z_List, max_Color_List):
            ax.scatter(x, y, z, c=colour)
        ax.set_xlabel('\n' + x_label)
        ax.set_ylabel('\n' + y_label, linespacing=3.1)
        ax.set_zlabel(
            '\n' + self.main_widget.getSelectedMetric().name, linespacing=3.1)
        ax.set_title(r'Dominating Functions')

        # draw legend
        patches = list()
        for key, value in dict_callpath_color.items():
            labelName = str(key.name)
            if labelName.startswith("_"):
                labelName = labelName[1:]
            patch = mpatches.Patch(color=value, label=labelName)
            patches.append(patch)

        leg = ax.legend(handles=patches, fontsize=fontSize,
                        loc="upper right", bbox_to_anchor=(1, 1))
        if leg:
            leg.set_draggable(True)
