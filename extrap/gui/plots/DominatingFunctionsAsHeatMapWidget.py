# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from extrap.gui.plots.BaseGraphWidget import GraphDisplayWindow


#####################################################################


class DominatingFunctionsAsHeatMap(GraphDisplayWindow):
    def __init__(self, heatMapGraphWidget, main_widget, width=5, height=4, dpi=100):
        super().__init__(heatMapGraphWidget, main_widget, width, height, dpi)

    def draw_figure(self):

        Z_List = list()
        z_List = list()
        lowerlimit = 1.0

        maxX = self.heatMapGraphWidget.getMaxX()
        maxY = self.heatMapGraphWidget.getMaxY()

        # 100, 75, 50
        if maxX <= 1000:
            numberOfPixels_x = 100
        elif 1000 < maxX <= 1000000000:
            numberOfPixels_x = 75
        else:
            numberOfPixels_x = 50

        if maxY <= 1000:
            numberOfPixels_y = 100
        elif 1000 < maxY <= 1000000000:
            numberOfPixels_y = 75
        else:
            numberOfPixels_y = 50

        # print("max x:", maxX, "max y:", maxY)
        pixelGap_x = self.getPixelGap(lowerlimit, maxX, numberOfPixels_x)
        pixelGap_y = self.getPixelGap(lowerlimit, maxY, numberOfPixels_y)
        x = np.arange(1.0, maxX, pixelGap_x)
        y = np.arange(1.0, maxY, pixelGap_y)
        X, Y = np.meshgrid(x, y)

        # Hardcording as of now , later on will be populated with callpath from experiment
        functions = list()
        functions.append("y**2+2*x")
        functions.append("2*y**3-x**2")
        functions.append("5.4*x*y")
        functions.append("12.68+3.67*10**-2*x**(5/4)*y")
        # functions.append("1.95*10**4+81.8*np.log(x)*y**(7/4) +4.62*10**3*y**(7/4)")
        functions.append("9.82+9.62*10**-3*x*y**(3/2)")

        for i in range(len(functions)):
            func = functions[i]
            # print ( "func", functions[i])
            zs = np.array([self.calculate_z(x, y, eval(func))
                           for x, y in zip(np.ravel(X), np.ravel(Y))])
            Z = zs.reshape(X.shape)
            z_List.append(zs)
            Z_List.append(Z)

        # define a color map depending upon the number of functions we have
        # todo: as of now hardcoding
        # colors = ['r','g','b']
        # colors = ['c','m','y']
        colors = ['r', 'g', 'b', 'c', 'm', 'y']

        # map each function to a particular color
        dict_callpath_color = self.populateCallPathColorMap(functions, colors)

        # for each x,y value , calculate max z for all function and get the associated function
        # for which z is highest.
        # Also store the the associated z value.

        max_z_val = z_List[0][0]
        color_for_max_z = dict_callpath_color[functions[0]]
        max_z_list = list()
        max_color_list = list()
        max_function_list = list()

        for i in range(len(z_List[0])):
            for j in range(len(functions)):
                if z_List[j][i] > max_z_val:
                    max_z_val = z_List[j][i]
                    func_with_max_z = functions[j]
                    color_for_max_z = dict_callpath_color[func_with_max_z]
            max_z_list.append(max_z_val)
            max_function_list.append(func_with_max_z)
            max_color_list.append(color_for_max_z)

        # indicesList = list()
        function_indices_map = {}
        for i in range(len(functions)):
            indices = self.get_dominating_function_indices(
                functions[i], max_function_list)
            if indices:
                function_indices_map[functions[i]] = indices
        # print ("function_indices_map", function_indices_map)

        max_Z_List = np.array(max_z_list).reshape(X.shape)
        max_Color_List = np.array(max_color_list).reshape(X.shape)

        ax1 = self.fig.add_subplot(1, 1, 1)
        ax1.set_xlabel(r'X')
        ax1.set_ylabel(r'Y')
        ax1.set_title(r'Dominating Functions')
        for (x, y, colour) in zip(X, Y, max_Color_List):
            ax1.scatter(x, y, c=colour)

        numOfCurves = 12
        maxZ = max([max(row) for row in max_Z_List])
        levels = np.arange(0, maxZ, (1 / float(numOfCurves)) * maxZ)
        CS = ax1.contour(X, Y, max_Z_List, levels=levels)
        ax1.clabel(CS, CS.levels[::1], inline=True, fontsize=self.main_widget.plot_formatting_options.font_size * 0.8)

        # legend

        patches = list()
        for key, value in dict_callpath_color.items():
            patch = mpatches.Patch(color=value, label=key)
            patches.append(patch)

        leg = ax1.legend(handles=patches, loc="upper right",
                         bbox_to_anchor=(1, 1))
        if leg:
            leg.set_draggable(True)

    @staticmethod
    def calculate_z(x, y, functiontoEvaluate):
        return functiontoEvaluate

    @staticmethod
    def populateCallPathColorMap(callpaths, colors):
        dict_callpath_color = {}
        current_index = 0
        for callpath in callpaths:
            dict_callpath_color[callpath] = colors[current_index]
            current_index = current_index + 1
        return dict_callpath_color

    @staticmethod
    def getColorMap():
        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        n_bin = 100
        cmap_name = 'my_list'
        colorMap = LinearSegmentedColormap.from_list(
            cmap_name, colors, N=n_bin)
        return colorMap

    def get_callpath_color_map(self):
        return self.dict_callpath_color

    @staticmethod
    def get_dominating_function_indices(function, functionList):
        functionIndexList = list()
        functionIndex = -1
        while True:
            try:
                functionIndex = functionList.index(function, functionIndex + 1)
                functionIndexList.append(functionIndex)
            except ValueError:
                break
        return functionIndexList
