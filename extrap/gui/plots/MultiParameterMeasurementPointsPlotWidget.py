# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2025, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QCheckBox, QPushButton, QComboBox
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
        # Name of the parameter currently used to split the data into separate surfaces;
        # None means "not chosen yet", in which case draw_figure() defaults to the first
        # parameter that isn't shown on the X/Y axis.
        self._series_parameter_name = None
        # Which values of the chosen parameter to show; missing keys default to visible.
        self._value_visible = {}
        self._controls_widget = None
        self._controls_layout = None
        super().__init__(graphWidget, main_widget, width, height, dpi)

    def get_controls_widget(self):
        """ Returns the widget with the parameter selector, the value checkboxes, and the
            fit-to-data button, or None if it hasn't been built yet (before the first draw).
        """
        return self._controls_widget

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

        # Get base data for drawing points
        parameter_x = self.main_widget.data_display.getAxisParameter(0)
        parameter_y = self.main_widget.data_display.getAxisParameter(1)
        parameters = self.main_widget.data_display.parameters

        # Set the x_label and y_label based on parameter selected.
        x_label = parameter_x.name
        if x_label.startswith("_"):
            x_label = x_label[1:]
        y_label = parameter_y.name
        if y_label.startswith("_"):
            y_label = y_label[1:]

        ax_all = self.fig.add_subplot(1, 1, 1, projection='3d')

        # Candidates for the "surfaces by" parameter: everything not already shown on the
        # X/Y axis. Like parameter_x.id/parameter_y.id, each index here is a *position*
        # within `parameters` (as assigned by the axis combo boxes), not the parameter's
        # own global id, so the coordinate index must be read the same way.
        candidate_parameters = [(i, p) for i, p in enumerate(parameters)
                                if i != parameter_x.id and i != parameter_y.id]

        if not candidate_parameters:
            ax_all.text2D(0.5, 0.5, "This plot requires an experiment with at least 3 parameters",
                          transform=ax_all.transAxes, ha='center')
            if self._controls_widget is not None:
                self._controls_widget.setVisible(False)
            return

        # Which parameter is used to split the data into surfaces is chosen via our own
        # dropdown (see _update_controls()), not via the Graph limits panel - its value
        # has no meaning here, since every measured value of it is shown at once, so a
        # fixed-value selector there wouldn't make sense. Default to the first candidate
        # until the user picks one, and fall back to it if the chosen name is no longer
        # a candidate (e.g. it just got assigned to the X or Y axis instead).
        candidate_names = [p.name for _, p in candidate_parameters]
        if self._series_parameter_name not in candidate_names:
            self._series_parameter_name = candidate_names[0]
        series_parameter_index, series_parameter = next(
            (i, p) for i, p in candidate_parameters if p.name == self._series_parameter_name)

        # The Graph limits panel shows a fixed-value selector for every parameter not on
        # an axis, but the one currently chosen above is ignored by this plot (every
        # measured value of it is shown at once), so its value field would be misleading.
        # Hide just that one; any other leftover parameter's value selector still matters
        # (it's used to evaluate the model) and stays visible.
        for value_selection in self.main_widget.data_display.value_selections:
            value_selection.setVisible(value_selection.parameter != series_parameter_index)

        # Determine the discrete values of the chosen parameter that occur in the measurements
        series_values = sorted({
            m.coordinate[series_parameter_index]
            for model in model_list for m in model.measurements
        })

        self._update_controls(candidate_parameters, series_parameter, series_values)
        # Only the values checked in the controls widget are actually drawn, but colors
        # are assigned over the full set (see below) so shades don't shift as values are
        # toggled on/off.
        displayed_values = [v for v in series_values if self._value_visible.get(v, True)]

        # Preserve plotting order (and de-duplicate) so every plotted callpath gets its
        # own color family, regardless of how many models are selected.
        plotted_callpaths = list(dict.fromkeys(selected_callpaths))

        # Every (callpath, value) combination gets its own, never-repeated color: the
        # callpath picks a color family (e.g. "Blues"), the value picks a shade within
        # it. This way selecting several models never reuses colors between surfaces,
        # while surfaces from the same model still read as a related family.
        surface_colors = self._compute_surface_colors(plotted_callpaths, series_values)

        max_z = 0
        for model, callpath in zip(model_list, selected_callpaths):
            points = model.measurements
            for value in displayed_values:
                value_points = [m for m in points if m.coordinate[series_parameter_index] == value]
                if not value_points:
                    continue
                color = surface_colors[(callpath, value)]

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

                # Draw points (larger than in the single-value "Measurement points" plot,
                # since here they compete with several overlapping surfaces instead of one)
                ax_all.scatter(xs, ys, mean, color=color, marker='x', s=60, linewidth=1.5)
                ax_all.scatter(xs, ys, median, color=color, marker='+', s=60, linewidth=1.5)
                ax_all.scatter(xs, ys, minimum, color=color, marker='_', s=60, linewidth=1.5)
                ax_all.scatter(xs, ys, maximum, color=color, marker='_', s=60, linewidth=1.5)
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
                for value in displayed_values:
                    zs = self.calculate_z_optimized(X, Y, function, {series_parameter_index: value})
                    finite = zs[np.logical_not(np.isinf(zs))]
                    if finite.size:
                        max_z = max(max_z, np.max(finite))
                    zs[np.isinf(zs)] = max_z
                    Z = zs.reshape(X.shape)
                    color = surface_colors[(callpath, value)]
                    # No edgecolor: a colored wireframe (rstride=cstride=1 means one
                    # line per grid cell) turns into dense clutter that buries the
                    # sparse scatter points, which is much worse with several
                    # overlapping surfaces (one per value) than with the single
                    # surface per model the original "Measurement points" plot draws.
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
        ax_all.set_title("Measurement points (3 parameters)")

        self._draw_legend(ax_all, plotted_callpaths, series_parameter, displayed_values, surface_colors)

    def _compute_surface_colors(self, plotted_callpaths, series_values):
        """ Assigns every (callpath, value) combination its own, unique color, drawn from
            Extra-P's currently selected colormap (Plot formatting options -> Colormap)
            instead of a separate hardcoded palette, so this plot's colors stay consistent
            with the rest of the application. Colors are handed out across all (callpath,
            value) pairs together (not one color per callpath, reused across values), so
            no two surfaces are ever assigned the same color; if there are more pairs than
            colors in the palette, it wraps around with progressively lighter shades, the
            same way ModelColorMap extends its palette for extra callpaths elsewhere.
        """
        color_list = self.main_widget.model_color_map.color_list
        pairs = [(callpath, value) for callpath in plotted_callpaths for value in series_values]

        surface_colors = {}
        for i, pair in enumerate(pairs):
            if i < len(color_list):
                surface_colors[pair] = color_list[i]
            else:
                offset = i % len(color_list)
                multiple = i // len(color_list)
                surface_colors[pair] = QColor(color_list[offset]).lighter(100 + 20 * multiple).name()
        return surface_colors

    def _update_controls(self, candidate_parameters, series_parameter, series_values):
        """ (Re-)builds the controls row: a dropdown to choose which parameter is split
            into separate surfaces, checkboxes to show/hide its individual values
            (keeping the on/off state of values that still occur in the data and
            defaulting newly seen values to visible), and a button to fit the view back
            to the currently shown data.
        """
        if self._controls_widget is None:
            self._controls_widget = QWidget()
            self._controls_layout = QHBoxLayout(self._controls_widget)
            self._controls_layout.setContentsMargins(4, 2, 4, 2)
        else:
            self._controls_widget.setVisible(True)
            while self._controls_layout.count():
                item = self._controls_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        self._controls_layout.addWidget(QLabel("Surfaces by:"))
        series_combo = QComboBox()
        for _, p in candidate_parameters:
            series_combo.addItem(p.name)
        series_combo.setCurrentText(series_parameter.name)
        series_combo.currentTextChanged.connect(self._on_series_parameter_changed)
        self._controls_layout.addWidget(series_combo)

        self._controls_layout.addWidget(QLabel(f"Show {series_parameter.name}:"))
        for value in series_values:
            checkbox = QCheckBox(f"{value:g}")
            checkbox.setChecked(self._value_visible.get(value, True))
            checkbox.toggled.connect(lambda checked, v=value: self._on_value_toggled(v, checked))
            self._controls_layout.addWidget(checkbox)
        self._controls_layout.addStretch(1)
        fit_button = QPushButton("Fit view to data")
        fit_button.setToolTip("Reset the zoom/pan and rescale the axes to fit the currently shown data")
        fit_button.clicked.connect(self._on_fit_to_data)
        self._controls_layout.addWidget(fit_button)

    def _on_series_parameter_changed(self, name):
        if name == self._series_parameter_name:
            return
        self._series_parameter_name = name
        # Visibility state belongs to the previous parameter's values; a same-numbered
        # value of the newly chosen parameter isn't necessarily related to it.
        self._value_visible = {}
        # Switching which parameter creates the surfaces changes the data range
        # entirely, so let the axes auto-scale to the new data instead of preserving
        # the old view.
        self.redraw()

    def _on_value_toggled(self, value, checked):
        self._value_visible[value] = checked
        # Showing/hiding a value doesn't change the underlying data range, so keep
        # whatever axis limits are currently shown (e.g. from a manual zoom) instead of
        # letting matplotlib auto-rescale to only what's now drawn.
        self.redraw_preserving_limits()

    def _on_fit_to_data(self):
        # A plain redraw() (unlike redraw_preserving_limits()) lets the axes auto-scale
        # to whatever is currently drawn, i.e. fits the view to the visible data again.
        self.redraw()

    def _draw_legend(self, ax_all, plotted_callpaths, series_parameter, series_values, surface_colors):
        patches = []
        for callpath in plotted_callpaths:
            label = str(callpath.name)
            if label.startswith("_"):
                label = label[1:]
            label = replace_method_parameters(label)
            for value in series_values:
                value_label = f"{label} ({series_parameter.name}={value:g})"
                patches.append(mpatches.Patch(color=surface_colors[(callpath, value)], label=value_label))
        leg = ax_all.legend(handles=patches, fontsize=self.main_widget.plot_formatting_options.legend_font_size,
                            loc="upper right", bbox_to_anchor=(1, 1))
        if leg:
            leg.set_draggable(True)
