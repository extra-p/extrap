# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import matplotlib.ticker as ticker
import numpy as np
from matplotlib import cm

from extrap.gui.plots.BaseGraphWidget import GraphDisplayWindow


#####################################################################


class MaxZAsSingleSurfacePlot(GraphDisplayWindow):
    def __init__(self, graphWidget, main_widget, width=5, height=4, dpi=100):
        try:
            self.colormap = cm.get_cmap('viridis')
        except ValueError:
            self.colormap = cm.get_cmap('spectral')

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

        # calculate max_z value
        if len(model_list) == 1:
            max_Z_List = Z_List[0]

        else:
            # for each x,y value , calculate max z for all function
            max_z_list = list()
            for i in range(len(z_List[0])):
                max_z_val = z_List[0][i]
                for j in range(len(model_list)):
                    if z_List[j][i] > max_z_val:
                        max_z_val = z_List[j][i]
                max_z_list.append(max_z_val)

            max_Z_List = np.array(max_z_list).reshape(X.shape)

        # Get the callpath color map
        # dict_callpath_color = self.main_widget.get_callpath_color_map()
        # Set the x_label and y_label based on parameter selected.
        x_label = self.main_widget.data_display.getAxisParameter(0).name
        if x_label.startswith("_"):
            x_label = x_label[1:]
        y_label = self.main_widget.data_display.getAxisParameter(1).name
        if y_label.startswith("_"):
            y_label = y_label[1:]

        # Draw plot showing max z value considering all the selected models
        number_of_subplots = 1
        ax = self.fig.add_subplot(
            1, number_of_subplots, number_of_subplots, projection='3d')
        ax.mouse_init()
        ax.xaxis.major.formatter._useMathText = True
        ax.yaxis.major.formatter._useMathText = True
        ax.zaxis.major.formatter._useMathText = True
        im = ax.plot_surface(X, Y, max_Z_List, cmap=self.colormap)

        ax.set_xlabel('\n' + x_label, linespacing=3.2)
        ax.set_ylabel('\n' + y_label, linespacing=3.1)
        ax.set_zlabel(
            '\n' + self.main_widget.getSelectedMetric().name, linespacing=3.1)
        ax.set_title(r'Max Z value')
        self.fig.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.2, format=ticker.ScalarFormatter(useMathText=True))
        # self.fig.tight_layout()
