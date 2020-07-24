"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""

import matplotlib.ticker as ticker

import numpy as np

from matplotlib import cm

from matplotlib.colors import LinearSegmentedColormap

#####################################################################
from gui.plots.BaseGraphWidget import BaseContourGraph


class InterpolatedContourDisplay(BaseContourGraph):
    def __init__(self, graphWidget, main_widget, width=5, height=4, dpi=100):

        try:
            self.colormap = cm.get_cmap('viridis')
        except:
            self.colormap = cm.get_cmap('spectral')
        super().__init__(graphWidget, main_widget, width, height, dpi)

    def draw_figure(self):
        """ 
          This function draws the graph
        """

        # Get data
        model_list, selected_callpaths = self.get_selected_models()

        # Get font size for legend
        fontSize = self.graphWidget.getFontSize()

        # Get max x and max y value as a initial default value or a value provided by user
        maxX, maxY = self.get_max()

        X, Y, Z_List, z_List = self.calculate_z_models(maxX, maxY, model_list)

        # Get the callpath color map
        # dict_callpath_color = self.main_widget.get_callpath_color_map()

        # define the number of subplots
        number_of_subplots = 1
        if (len(Z_List) > 1):
            number_of_subplots = len(Z_List)

        # Adjusting subplots in order to avoid overlapping of labels
        # Reference : https://stackoverflow.com/questions/2418125/matplotlib-subplots-adjust-hspace-so-titles-and-xlabels-dont-overlap
        left = 0.1  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.2  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.5  # the amount of width reserved for blank space between subplots
        hspace = 0.2
        self.fig.subplots_adjust(
            left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        # Set the x_label and y_label based on parameter selected.
        x_label = self.main_widget.data_display.getAxisParameter(0).name
        if x_label.startswith("_"):
            x_label = x_label[1:]
        y_label = self.main_widget.data_display.getAxisParameter(1).name
        if y_label.startswith("_"):
            y_label = y_label[1:]

        numOfCurves = 15
        # cm = self.getColorMap()
        # cm ='viridis'
        # cm='hot'

        for i in range(len(Z_List)):
            maxZ = max([max(row) for row in Z_List[i]])
            levels = np.arange(0, maxZ, (1 / float(numOfCurves)) * maxZ)
            ax = self.fig.add_subplot(1, number_of_subplots, i + 1)
            ax.xaxis.major.formatter._useMathText = True
            ax.yaxis.major.formatter._useMathText = True
            CM = ax.pcolormesh(X, Y, Z_List[i], cmap=self.colormap)
            self.fig.colorbar(CM, ax=ax, orientation="horizontal",
                              pad=0.2, format=ticker.ScalarFormatter(useMathText=True))
            try:
                CS = ax.contour(X, Y, Z_List[i], colors="white", levels=levels)
                ax.clabel(CS, CS.levels[::1], inline=True, fontsize=8)
            except ValueError:  # raised if function selected is constant
                pass
            ax.set_xlabel('\n' + x_label)
            ax.set_ylabel('\n' + y_label)

            titleName = selected_callpaths[i].name
            if titleName.startswith("_"):
                titleName = titleName[1:]
            ax.set_title(titleName)
            for item in ([ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(10)
            for item in ([ax.title]):
                item.set_fontsize(fontSize)

    def getColorMap(self):
        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        n_bin = 100
        cmap_name = 'my_list'
        colorMap = LinearSegmentedColormap.from_list(
            cmap_name, colors, N=n_bin)
        return colorMap
