# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extrap.gui.plots.BaseGraphWidget import GraphDisplayWindow


#####################################################################


class AllFunctionsAsOneSurfacePlot(GraphDisplayWindow):

    def __init__(self, graphWidget, main_widget, width=5, height=4, dpi=100):
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
        widget = self.main_widget
        dict_callpath_color = widget.model_color_map

        # colors = ['r','g','b','c','m','y']
        # colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        # Set the x_label and y_label based on parameter selected.
        x_label = self.main_widget.data_display.getAxisParameter(0).name
        if x_label.startswith("_"):
            x_label = x_label[1:]
        y_label = self.main_widget.data_display.getAxisParameter(1).name
        if y_label.startswith("_"):
            y_label = y_label[1:]

        # 1 because we are going to show all the models in same plot
        number_of_subplots = 1

        # Draw all the selected models
        ax_all = self.fig.add_subplot(
            1, number_of_subplots, 1, projection='3d')
        ax_all.mouse_init()
        ax_all.get_xaxis().get_major_formatter().set_scientific(True)
        ax_all.xaxis.major.formatter._useMathText = True
        ax_all.yaxis.major.formatter._useMathText = True
        ax_all.zaxis.major.formatter._useMathText = True
        ax_all.set_xlabel('\n' + x_label)
        ax_all.set_ylabel('\n' + y_label, linespacing=3.1)
        ax_all.set_zlabel(
            '\n' + self.main_widget.getSelectedMetric().name, linespacing=3.1)
        for i in range(len(Z_List)):
            ax_all.plot_surface(
                X, Y, Z_List[i], color=dict_callpath_color[selected_callpaths[i]])

        # draw legend
        self.draw_legend(ax_all, dict_callpath_color)
