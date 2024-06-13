# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import numpy as np

from extrap.gui.plots.BaseGraphWidget import GraphDisplayWindow


# This class was developed as a first approach to show dominating models in heatmap
# not used any more
# HeatMapGraphWidget in used for that purpose
# but still keeping it since it has a different approach implemented


#####################################################################


class DominatingFunctionsAsSingleScatterPlot(GraphDisplayWindow):

    def __init__(self, graphWidget, main_widget, width=5, height=4, dpi=100):
        super().__init__(graphWidget, main_widget, width, height, dpi)

    def draw_figure(self):
        """ 
          This function draws the graph
        """
        # Get data
        model_list, selected_callpaths = self.main_widget.get_selected_models()
        if model_list is None:
            return

        # Get max x and max y value as a initial default value or a value provided by user
        maxX, maxY = self.get_max()

        X, Y, Z_list, z_List = self.calculate_z_models(maxX, maxY, model_list)

        # Get the callpath color map
        widget = self.main_widget
        dict_callpath_color = widget.model_color_map

        # calculate max_z value

        max_z_list = list()
        max_color_list = list()

        for i in range(len(z_List[0])):
            max_z_val = z_List[0][i]
            color_for_max_z = dict_callpath_color[selected_callpaths[0]]
            for j in range(len(model_list)):
                if z_List[j][i] > max_z_val:
                    max_z_val = z_List[j][i]
                    # func_with_max_z = model_list[j]
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
        ax = self.fig.add_subplot(1, number_of_subplots, number_of_subplots, projection='3d')
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
            '\n' + self.main_widget.get_selected_metric().name, linespacing=3.1)
        ax.set_title(r'Dominating Functions')

        self.draw_legend(ax, dict_callpath_color)
