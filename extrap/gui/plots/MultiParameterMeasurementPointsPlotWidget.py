# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2025, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import numpy as np
from PySide6.QtGui import QColor
from matplotlib import patches as mpatches

from extrap.gui.plots.BaseGraphWidget import GraphDisplayWindow
from extrap.util.formatting_helper import replace_method_parameters


#####################################################################


class MultiParameterMeasurementPointsPlot(GraphDisplayWindow):
    """
    Shows the same information as the "Measurement points" plot (measurement
    points and model surface for two parameters shown on the X- and Y-axis),
    but instead of evaluating the model for a single, fixed value of the
    remaining parameter, it draws one surface (and its measurement points)
    for every discrete value of that parameter that occurs in the
    measurements. This is intended for experiments with three parameters.
    """

    def __init__(self, graphWidget, main_widget, width=5, height=4, dpi=100):
        super().__init__(graphWidget, main_widget, width, height, dpi)

    def draw_figure(self):
        """
          This function draws the graph
        """

        # Get data
        model_list, selected_callpaths = self.main_widget.get_selected_models()
        if model_list is None or len(model_list) < 1:
            return

        # Get max x and max y value as a initial default value or a value provided by user
        maxX, maxY = self.get_max()

        # Get the callpath color map
        widget = self.main_widget
        dict_callpath_color = widget.model_color_map

        # Get base data for drawing points
        parameter_x = self.main_widget.data_display.getAxisParameter(0)
        parameter_y = self.main_widget.data_display.getAxisParameter(1)
        parameters = self.main_widget.data_display.parameters

        # The third parameter is the first parameter that is not shown on an axis.
        # Note: parameter_x.id/parameter_y.id are *positions* within `parameters`
        # (as assigned by the axis combo boxes), not the parameters' own global ids,
        # so the series parameter and its coordinate index must be found by position too.
        series_parameter = None
        series_parameter_index = None
        for i, p in enumerate(parameters):
            if i != parameter_x.id and i != parameter_y.id:
                series_parameter = p
                series_parameter_index = i
                break

        # Set the x_label and y_label based on parameter selected.
        x_label = parameter_x.name
        if x_label.startswith("_"):
            x_label = x_label[1:]
        y_label = parameter_y.name
        if y_label.startswith("_"):
            y_label = y_label[1:]

        ax_all = self.fig.add_subplot(1, 1, 1, projection='3d')

        if series_parameter is None:
            ax_all.text2D(0.5, 0.5, "This plot requires an experiment with at least 3 parameters",
                          transform=ax_all.transAxes, ha='center')
            return

        # Determine the discrete values of the third parameter that occur in the measurements
        series_values = sorted({
            m.coordinate[series_parameter_index]
            for model in model_list for m in model.measurements
        })

        max_z = 0
        for model, callpath in zip(model_list, selected_callpaths):
            base_color = QColor(dict_callpath_color[callpath])
            points = model.measurements
            for value in series_values:
                value_points = [m for m in points if m.coordinate[series_parameter_index] == value]
                if not value_points:
                    continue
                color = self._shade_color(base_color, value, series_values)

                if parameter_x.id >= 0:
                    xs = np.array([m.coordinate[parameter_x.id] for m in value_points])
                else:
                    xs = np.zeros(len(value_points))
                if parameter_y.id >= 0:
                    ys = np.array([m.coordinate[parameter_y.id] for m in value_points])
                else:
                    ys = np.full(len(value_points), max(min(np.min(xs), 0), 0))
                in_range = (xs <= maxX) & (ys <= maxY)
                xs, ys = xs[in_range], ys[in_range]

                mean = np.array([m.mean for m in value_points])[in_range]
                median = np.array([m.median for m in value_points])[in_range]
                minimum = np.array([m.minimum for m in value_points])[in_range]
                maximum = np.array([m.maximum for m in value_points])[in_range]
                if len(maximum) > 0:
                    max_z = max(max_z, max(maximum))

                # Draw points
                ax_all.scatter(xs, ys, mean, color=color, marker='x')
                ax_all.scatter(xs, ys, median, color=color, marker='+')
                ax_all.scatter(xs, ys, minimum, color=color, marker='_')
                ax_all.scatter(xs, ys, maximum, color=color, marker='_')
                # Draw connecting line
                line_x, line_y, line_z = [], [], []
                for x, y, min_v, max_v in zip(xs, ys, minimum, maximum):
                    line_x.append(x), line_x.append(x)
                    line_y.append(y), line_y.append(y)
                    line_z.append(min_v), line_z.append(max_v)
                    line_x.append(np.nan), line_y.append(np.nan), line_z.append(np.nan)

                ax_all.plot(line_x, line_y, line_z, color=color)

        # plot surfaces: one per model and per discrete value of the third parameter
        pixel_gap_x, pixel_gap_y = self._calculate_grid_parameters(maxX, maxY)
        x = np.arange(1.0, maxX, pixel_gap_x)
        y = np.arange(1.0, maxY, pixel_gap_y)
        X, Y = np.meshgrid(x, y)

        with np.errstate(invalid='ignore', divide='ignore'):
            for model, callpath in zip(model_list, selected_callpaths):
                function = model.hypothesis.function
                base_color = QColor(dict_callpath_color[callpath])
                for value in series_values:
                    zs = self.calculate_z_optimized(X, Y, function, {series_parameter_index: value})
                    finite = zs[np.logical_not(np.isinf(zs))]
                    if finite.size:
                        max_z = max(max_z, np.max(finite))
                    zs[np.isinf(zs)] = max_z
                    Z = zs.reshape(X.shape)
                    color = self._shade_color(base_color, value, series_values)
                    ax_all.plot_surface(X, Y, Z, color=color,
                                        rstride=1, cstride=1, antialiased=False, alpha=0.1)

        ax_all.mouse_init()
        ax_all.xaxis.major.formatter._useMathText = True
        ax_all.yaxis.major.formatter._useMathText = True
        ax_all.zaxis.major.formatter._useMathText = True
        ax_all.set_xlabel('\n' + x_label)
        ax_all.set_ylabel('\n' + y_label, linespacing=3.1)
        ax_all.set_zlabel(
            '\n' + self.main_widget.get_selected_metric().name, linespacing=3.1)
        ax_all.set_title("Measurement points")

        self._draw_legend(ax_all, dict_callpath_color, series_parameter, series_values)

    @staticmethod
    def _shade_color(base_color: QColor, value, series_values):
        """ Shades the callpath's base color from darker (lowest value of the
            third parameter) to lighter (highest value), so that surfaces for
            the same callpath but different parameter values stay visually
            distinguishable while remaining recognizable as the same callpath.
        """
        if len(series_values) <= 1:
            return base_color.name()
        position = series_values.index(value) / (len(series_values) - 1)
        factor = 60 + position * 120  # 60 (darker) .. 180 (lighter)
        return base_color.lighter(int(factor)).name()

    def _draw_legend(self, ax_all, dict_callpath_color, series_parameter, series_values):
        patches = []
        for callpath, color in dict_callpath_color.items():
            base_color = QColor(color)
            for value in series_values:
                shade = self._shade_color(base_color, value, series_values)
                label = str(callpath.name)
                if label.startswith("_"):
                    label = label[1:]
                label = replace_method_parameters(label)
                label = f"{label} ({series_parameter.name}={value:g})"
                patches.append(mpatches.Patch(color=shade, label=label))
        leg = ax_all.legend(handles=patches, fontsize=self.main_widget.plot_formatting_options.legend_font_size,
                            loc="upper right", bbox_to_anchor=(1, 1))
        if leg:
            leg.set_draggable(True)
