# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2025, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extrap.gui.plots.BaseGraphWidget import GraphDisplayWindow


#####################################################################


class AllFunctionsAsDifferentSurfacePlot(GraphDisplayWindow):
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

        X, Y, Z_List, z_List = self.calculate_z_models(maxX, maxY, model_list)

        # Get the callpath color map
        widget = self.main_widget
        dict_callpath_color = widget.model_color_map

        # Create subplots based on the number of functions
        number_of_subplots = len(Z_List)

        # Adjusting subplots in order to avoid overlapping of labels
        # Reference : https://stackoverflow.com/questions/2418125/matplotlib-subplots-adjust-hspace-so-titles-and-xlabels-dont-overlap

        left = 0.1
        right = 0.9
        bottom = 0.2
        top = 0.9
        wspace = 0.3
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

        opacity = widget.plot_formatting_options.surface_opacity

        # Draw the graphs in subplots
        for i in range(len(Z_List)):
            ax = self.fig.add_subplot(
                1, number_of_subplots, i + 1, projection='3d')
            ax.mouse_init()
            ax.get_xaxis().get_major_formatter().set_scientific(True)
            ax.xaxis.major.formatter._useMathText = True
            ax.yaxis.major.formatter._useMathText = True
            ax.zaxis.major.formatter._useMathText = True
            ax.plot_surface(
                X, Y, Z_List[i], color=dict_callpath_color[selected_callpaths[i]], alpha=opacity)
            ax.set_xlabel('\n' + x_label)
            ax.set_ylabel('\n' + y_label, linespacing=3.1)
            ax.set_zlabel(
                '\n' + self.main_widget.get_selected_metric().name, linespacing=3.1)

        # draw legend
        self.draw_legend(ax, dict_callpath_color)

        self.fig.tight_layout()
