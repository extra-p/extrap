# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import numpy as np

from extrap.gui.plots.BaseGraphWidget import GraphDisplayWindow


#####################################################################


class MeasurementPointsPlot(GraphDisplayWindow):
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

        if len(model_list) < 1:
            return

        X, Y, Z_List, z_List = self.calculate_z_models(maxX, maxY, model_list)

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
        parameter_x = self.main_widget.data_display.getAxisParameter(0)
        parameter_y = self.main_widget.data_display.getAxisParameter(1)

        for model, callpath in zip(model_list, selected_callpaths):
            callpath_color = dict_callpath_color[callpath]
            points = model.measurements
            mean = np.array([m.mean for m in points])
            median = np.array([m.median for m in points])
            minimum = np.array([m.minimum for m in points])
            maximum = np.array([m.maximum for m in points])
            xs = np.array([m.coordinate[parameter_x.id] for m in points])
            ys = np.array([m.coordinate[parameter_y.id] for m in points])

            # Draw points
            ax_all.scatter(xs, ys, mean, color=callpath_color, marker='x')
            ax_all.scatter(xs, ys, median, color=callpath_color, marker='+')
            ax_all.scatter(xs, ys, minimum, color=callpath_color, marker='_')
            ax_all.scatter(xs, ys, maximum, color=callpath_color, marker='_')
            # Draw connecting line
            line_x, line_y, line_z = [], [], []
            for x, y, min_v, max_v in zip(xs, ys, minimum, maximum):
                line_x.append(x), line_x.append(x)
                line_y.append(y), line_y.append(y)
                line_z.append(min_v), line_z.append(max_v)
                line_x.append(np.nan), line_y.append(np.nan), line_z.append(np.nan)

            ax_all.plot(line_x, line_y, line_z, color=callpath_color)

        self.draw_legend(ax_all, dict_callpath_color)
