"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""

import random
import sys

import matplotlib.ticker as ticker
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from extrap.gui.plots.BaseGraphWidget import BaseContourGraph


#####################################################################


class HeatMapGraph(BaseContourGraph):
    def __init__(self, graphWidget, main_widget, width=5, height=4, dpi=100):
        try:
            self.colormap = cm.get_cmap('viridis')
        except ValueError:
            self.colormap = cm.get_cmap('spectral')

        # initializing value to be used later in finding boundary points
        self.isLeft = 1
        self.isRight = -1
        self.isInSameLine = 0
        self.epsilon = sys.float_info.epsilon

        super().__init__(graphWidget, main_widget, width, height, dpi)

    def draw_figure(self):
        """ 
          This function draws the graph
        """
        # Get data
        model_list, selected_callpaths = self.get_selected_models()
        if model_list is None:
            return

        # Get max x and max y value as a initial default value or a value provided by user
        maxX, maxY = self.get_max()

        X, Y, Z_List, z_List = self.calculate_z_models(maxX, maxY, model_list)

        # Get the callpath color map
        dict_callpath_color = self.main_widget.get_callpath_color_map()

        # for each x,y value , calculate max z for all function and
        # get the associated model functions for which z is highest.
        # Also store the the associated z value.

        color_for_max_z = dict_callpath_color[selected_callpaths[0]]
        max_z_list = list()
        max_color_list = list()
        max_function_list = list()
        func_with_max_z = model_list[0]

        for i in range(len(z_List[0])):
            max_z_val = z_List[0][i]
            for j in range(len(model_list)):
                if z_List[j][i] > max_z_val:
                    max_z_val = z_List[j][i]
                    func_with_max_z = model_list[j]
                    color_for_max_z = dict_callpath_color[selected_callpaths[j]]
            max_z_list.append(max_z_val)
            max_function_list.append(func_with_max_z)
            max_color_list.append(color_for_max_z)

        # get the indices of the dominating model functions
        # indicesList = list()
        function_indices_map = {}
        for i in range(len(model_list)):
            indices = self.get_dominating_function_indices(
                model_list[i], max_function_list)
            if indices:
                function_indices_map[selected_callpaths[i]] = indices
                # function_indices_map[model_list[i]] = indices

        # reshape the Max Z and corresponding color to give plot as input
        max_Z_List = np.array(max_z_list).reshape(X.shape)
        # max_Color_List = np.array(max_color_list).reshape(X.shape)

        # Set the x_label and y_label based on parameter selected.
        x_label = self.main_widget.data_display.getAxisParameter(0).name
        if x_label.startswith("_"):
            x_label = x_label[1:]
        y_label = self.main_widget.data_display.getAxisParameter(1).name
        if y_label.startswith("_"):
            y_label = y_label[1:]

        # Draw the graph showing the max z value

        # Step1: Draw the max Z value
        ax = self.fig.add_subplot(1, 1, 1)
        ax.get_xaxis().get_major_formatter().set_scientific(True)
        ax.xaxis.major.formatter._useMathText = True
        ax.yaxis.major.formatter._useMathText = True
        ax.set_xlabel('\n' + x_label)
        ax.set_ylabel('\n' + y_label)
        ax.set_title(r'Max Z value')
        im = ax.scatter(X, Y, c=max_Z_List, cmap=self.colormap)
        self.fig.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.2, format=ticker.ScalarFormatter(useMathText=True))

        # Step 2 : Mark the dominating functions by drawing boundary lines
        # For each dominating function, extract the (x, y)pair in which they dominate using the indices we got in
        # function_indices_map and then find the boundary of these points and plot on the graph
        for function in function_indices_map:
            indices_per_function = function_indices_map[function]
            x_indices = [X.ravel()[index] for index in indices_per_function]
            y_indices = [Y.ravel()[index] for index in indices_per_function]
            x_y_indices = list(zip(x_indices, y_indices))
            boundaryPoints = self.findBoundaryPoints(x_y_indices)
            i = 0
            while i < len(boundaryPoints) - 1:
                ax.plot([boundaryPoints[i][0], boundaryPoints[i + 1][0]], [boundaryPoints[i]
                                                                           [1], boundaryPoints[i + 1][1]],
                        color=dict_callpath_color[function])
                i = i + 1

        # Step 3: Draw legend
        self.draw_legend(ax, dict_callpath_color)

    @staticmethod
    def getColorMap():
        """ 
           This function creates a color map and return it.
        """
        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        n_bin = 100
        cmap_name = 'my_list'
        colorMap = LinearSegmentedColormap.from_list(
            cmap_name, colors, N=n_bin)
        return colorMap

    @staticmethod
    def get_dominating_function_indices(function, functionList):
        """ 
           This function filters the functions which are in the functionList and return their indices. 
        """
        functionIndexList = list()
        functionIndex = -1
        while True:
            try:
                functionIndex = functionList.index(function, functionIndex + 1)
                functionIndexList.append(functionIndex)
            except ValueError:
                break
        return functionIndexList

    # The below code is adapted from Stack Overflow
    # Reference: https://stackoverflow.com/questions/25787637/python-plot-only-the-outermost-points-of-a-dataset

    def findBoundaryPoints(self, pointsList):
        """ 
           This function finds the bpoints(x,y) that forms the boundary line from a given set
        """
        points = pointsList[:]
        minimumPoint = min(points)
        boundary = [minimumPoint]
        point1, point2 = minimumPoint, None
        remainingPoints = [point for point in points if point not in boundary]
        while point2 != minimumPoint and remainingPoints:
            point2 = random.choice(remainingPoints)
            for point in points:
                first = (point2[0] - point1[0], point2[1] - point1[1])
                second = (point[0] - point2[0], point[1] - point2[1])
                if (point != point2 and (self.findRelativePosition(first, second) == self.isRight or (
                        self.findRelativePosition(first, second) == self.isInSameLine and self.isOnSamePlane(first,
                                                                                                             second)))):
                    point2 = point
            point1 = point2
            points.remove(point1)
            boundary.append(point1)
            try:
                remainingPoints.remove(point1)
            except ValueError:
                pass
        return boundary

    def findRelativePosition(self, first, second):
        """ 
           This function finds the relative position of given two points; (x1,y1) and (x2, y2)
        """
        first_x, first_y = first
        second_x, second_y = second
        product = first_x * second_y - second_x * first_y
        if product < -self.epsilon:
            return self.isRight
        elif product > self.epsilon:
            return self.isLeft
        else:
            return self.isInSameLine

    def isOnSamePlane(self, first, second):
        """ 
            This function checks if two points; (x1,y1) and (x2, y2) are on the same plane
        """
        first_x, first_y = first
        second_x, second_y = second
        product = first_x * second_x + first_y * second_y
        if product < self.epsilon:
            return False
        else:
            return True
