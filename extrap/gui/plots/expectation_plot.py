# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import TYPE_CHECKING

import numpy as np
import pyvista
from PySide6.QtGui import QResizeEvent, QFont, QFontMetrics
from pyvista.plotting import parse_font_family
from pyvistaqt import QtInteractor
from vtkmodules.vtkCommonCore import vtkStringArray

from extrap.comparison.entities.comparison_model import ComparisonModel
from extrap.entities.metric import Metric
from extrap.gui.plots.AbstractPlotWidget import AbstractPlotWidget
from extrap.util.formatting_helper import format_number_ascii

if TYPE_CHECKING:
    from extrap.gui.MainWidget import MainWidget


class ExpectationPlot3D(QtInteractor, AbstractPlotWidget):

    def __init__(self, main_widget, parent, width=5, height=4, dpi=100):
        self.main_widget: MainWidget = main_widget

        self.max_values = [2, 2]
        super().__init__(parent)

    def setMax(self, axis, maxValue):
        self.max_values[axis] = maxValue

    def getMax(self, axis):
        return self.max_values[axis]

    def set_initial_value(self):
        self.max_values = [2, 2]

    def drawGraph(self):
        self.suppress_rendering = True
        self.enable_anti_aliasing()
        # Get data

        model_list1, selected_call_nodes1 = self.main_widget.get_selected_models()
        if not model_list1:
            return None, None
        estimation_id = 0
        model_list_estimation = None
        original_metric_name = self.main_widget.get_selected_metric().lookup_tag("projection__original_metric")
        if original_metric_name:
            original_metric = Metric(original_metric_name)
            target_id = self.main_widget.get_selected_metric().lookup_tag('projection__target_id')
            assert (target_id <= 1)
            estimation_id = abs(target_id - 1)
            model_list_estimation = [model.models[estimation_id] for model in model_list1]
            models = self.main_widget.get_current_model_gen().models
            model_list1 = [models[node.path, original_metric] for node in selected_call_nodes1]

        model_list = []
        selected_call_nodes = []
        for i, (model, call_node) in enumerate(zip(model_list1, selected_call_nodes1)):
            if isinstance(model, ComparisonModel):
                for m in model.models:
                    model_list.append(m)
                    selected_call_nodes.append(call_node)
            else:
                model_list.append(model)
                selected_call_nodes.append(call_node)

        self.clear()

        if model_list is None:
            return

        # Get max x and max y value as a initial default value or a value provided by user
        maxX, maxY = self.max_values

        if len(model_list) < 1:
            return

        # Get the callpath color map
        widget = self.main_widget
        dict_callpath_color = widget.model_color_map

        # Get base data for drawing points
        parameter_x = self.main_widget.data_display.getAxisParameter(0)
        parameter_y = self.main_widget.data_display.getAxisParameter(1)

        # Set the x_label and y_label based on parameter selected.
        x_label = parameter_x.name
        if x_label.startswith("_"):
            x_label = x_label[1:]
        y_label = parameter_y.name
        if y_label.startswith("_"):
            y_label = y_label[1:]

        max_z = 0

        # for model, callpath in zip(model_list, selected_call_nodes):
        #     callpath_color = dict_callpath_color[callpath]
        #     points = model.measurements
        #     if not points:
        #         continue
        #     if parameter_x.id >= 0:
        #         xs = np.array([m.coordinate[parameter_x.id] for m in points])
        #     else:
        #         xs = np.zeros(len(points))
        #     if parameter_y.id >= 0:
        #         ys = np.array([m.coordinate[parameter_y.id] for m in points])
        #     else:
        #         ys = np.full(len(points), max(min(np.min(xs), 0), 0))
        #     in_range = (xs <= maxX) & (ys <= maxY)
        #     xs, ys = xs[in_range], ys[in_range]
        #
        #     if not any(in_range):
        #         continue
        #
        #     mean = np.array([m.mean for m in points])[in_range]
        #     median = np.array([m.median for m in points])[in_range]
        #     minimum = np.array([m.minimum for m in points])[in_range]
        #     maximum = np.array([m.maximum for m in points])[in_range]
        #     if len(maximum) > 0:
        #         max_z = max(max_z, max(maximum))
        #
        #     # Draw points
        #
        #     minimum_points = np.array([xs, ys, minimum]).T
        #     maximum_points = np.array([xs, ys, maximum]).T
        #
        #     self.add_points(np.array([xs, ys, mean]).T, color=callpath_color)
        #     self.add_points(np.array([xs, ys, median]).T, color=callpath_color)
        #     self.add_points(minimum_points, color=callpath_color)
        #     self.add_points(maximum_points, color=callpath_color)
        #
        #     lines = np.ndarray((2 * minimum_points.shape[0], 3))
        #     lines[0::2] = minimum_points
        #     lines[1::2] = maximum_points
        #
        #     self.add_lines(lines, color=callpath_color)

        # Get the grid of the x and y values
        x = np.linspace(1.0, max(maxX, 2), 50)
        y = np.linspace(1.0, max(maxY, 2), 50)
        X, Y = np.meshgrid(x, y)
        xs, ys = X.reshape(-1), Y.reshape(-1)
        # Get the z value for the x and y value
        z_List = list()
        z_volumes_upper, z_volumes_lower = [], []
        previous = np.seterr(invalid='ignore', divide='ignore')
        for model in model_list1:
            if isinstance(model, ComparisonModel):
                for i, m in enumerate(model.models):
                    function = m.hypothesis.function
                    zs = self.calculate_z_optimized(X, Y, function)
                    z_List.append((zs, i))
                    max_z = max(max_z, np.max(zs[np.logical_not(np.isinf(zs))]))
            else:
                function = model.hypothesis.function
                zs = self.calculate_z_optimized(X, Y, function)
                z_List.append((zs, 0))
                max_z = max(max_z, np.max(zs[np.logical_not(np.isinf(zs))]))

        if model_list_estimation:
            for i, model_est in enumerate(model_list_estimation):
                function = model_est.hypothesis.function
                zs = self.calculate_z_optimized(X, Y, function)
                original_zs = z_List[i * 2 + estimation_id][0]
                zs_delta = np.abs(zs - original_zs) / 10
                zs_upper = zs + zs_delta
                zs_lower = zs - zs_delta
                z_volumes_upper.append(zs_upper)
                z_volumes_lower.append(zs_lower)
                max_z = max(max_z, np.max(zs_upper[np.logical_not(np.isinf(zs_upper))]))
                max_z = max(max_z, np.max(zs_lower[np.logical_not(np.isinf(zs_lower))]))
        np.seterr(**previous)
        for z, model_index in z_List:
            z[np.isinf(z)] = max_z
        for z in z_volumes_upper:
            z[np.isinf(z)] = max_z
        for z in z_volumes_lower:
            z[np.isinf(z)] = max_z

        if 10 ** -3 < abs(max(maxX, maxY, max_z)) < 10 ** 4:
            self.theme.font.fmt = '%.3f'
        else:
            self.theme.font.fmt = '%.3e'

        labels = []

        for i, (z, model_index) in enumerate(z_List):
            mesh = pyvista.StructuredGrid(X / maxX * 1000, Y / maxY * 1000, z.reshape(X.shape) / max_z * 1000)
            node = selected_call_nodes[i]
            color = dict_callpath_color.get_rgb(node)
            if model_index == 0:
                self.add_mesh(mesh, color=color,
                              opacity=0.5)
                labels.append(('  ' + node.name, color))
            else:
                mesh.texture_map_to_plane(inplace=True)
                mesh.active_t_coords *= 50
                texture = pyvista.Texture(np.array([[[255, 255, 255], color],
                                                    [color, color]], dtype=np.uint8))
                texture.repeat = True
                self.add_mesh(mesh, opacity=0.5, texture=texture)
        for i, (z_upper, z_lower) in enumerate(zip(z_volumes_upper, z_volumes_lower)):
            mesh = pyvista.StructuredGrid(np.stack([X, X], axis=-1) / maxX * 1000,
                                          np.stack([Y, Y], axis=-1) / maxY * 1000,
                                          np.stack([z_upper.reshape(X.shape), z_lower.reshape(X.shape)],
                                                   axis=-1) / max_z * 1000)
            self.add_mesh(mesh, color='yellow', lighting=True,
                          opacity=0.3, show_edges=True, edge_color='white')

            labels.append(('  Expectation', 'yellow'))

        self.add_axes_cube(maxX, maxY, max_z, x_label, y_label)

        self.show_axes()

        fm = QFontMetrics(QFont('Arial', self.main_widget.plot_formatting_options.legend_font_size))
        legend_height = fm.height() * 2 * len(labels) / self.height() + 5 * len(labels) / self.window_size[1]
        legend_with = (max(fm.horizontalAdvance(l) for l, _ in labels) + 2 * fm.height()) / self.width()
        legend_size = (legend_with, legend_height)
        legend = self.add_legend(labels, face='rectangle', border=True, bcolor=(255, 255, 255), size=legend_size)
        legend.SetBackgroundOpacity(0.5)
        legend.SetPadding(5)
        legend.SetPosition(1 - legend_with - 10 / self.height(), 1 - legend_height - 10 / self.height())
        for i, _ in enumerate(labels):
            legend.GetEntryTextProperty().SetColor(0.0, 0.0, 0.0)

        self.suppress_rendering = False
        self.render()

    def add_axes_cube(self, maxX, maxY, max_z, x_label, y_label):
        self.remove_bounds_axes()
        cube_axes_actor = pyvista.CubeAxesActor(
            self.camera,
            minor_ticks=False,
            tick_location='both',
            x_title=x_label,
            y_title=y_label,
            z_title=self.main_widget.get_selected_metric().name,
            x_axis_visibility=True,
            y_axis_visibility=True,
            z_axis_visibility=True,
            x_label_visibility=True,
            y_label_visibility=True,
            z_label_visibility=True,
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=5,
        )
        cube_axes_actor.SetXAxisRange(0.0, maxX)
        cube_axes_actor.SetYAxisRange(0.0, maxY)
        cube_axes_actor.SetZAxisRange(0.0, max_z)
        cube_axes_actor.SetBounds(0, 1000, 0, 1000, 0, 1000)
        string_array_x = vtkStringArray()
        for v in np.linspace(0, maxX, 5):
            string_array_x.InsertNextValue(format_number_ascii(v))
        cube_axes_actor.SetAxisLabels(0, string_array_x)
        string_array_y = vtkStringArray()
        for v in np.linspace(0, maxY, 5):
            string_array_y.InsertNextValue(format_number_ascii(v))
        cube_axes_actor.SetAxisLabels(1, string_array_y)
        string_array_z = vtkStringArray()
        for v in np.linspace(0, max_z, 5):
            string_array_z.InsertNextValue(format_number_ascii(v))
        cube_axes_actor.SetAxisLabels(2, string_array_z)

        cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_FURTHEST)
        cube_axes_actor.SetDrawXGridlines(True)
        cube_axes_actor.SetDrawYGridlines(True)
        cube_axes_actor.SetDrawZGridlines(True)
        # Set the colors
        cube_axes_actor.GetXAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)
        cube_axes_actor.GetYAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)
        cube_axes_actor.GetZAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)

        cube_axes_actor.GetXAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
        cube_axes_actor.GetYAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
        cube_axes_actor.GetZAxesLinesProperty().SetColor(0.0, 0.0, 0.0)

        cube_axes_actor.SetXTitle(cube_axes_actor.x_title)
        cube_axes_actor.SetYTitle(cube_axes_actor.y_title)
        cube_axes_actor.SetZTitle(cube_axes_actor.z_title)

        cube_axes_actor.SetUseTextActor3D(False)
        props = [
            cube_axes_actor.GetTitleTextProperty(0),
            cube_axes_actor.GetTitleTextProperty(1),
            cube_axes_actor.GetTitleTextProperty(2),
            cube_axes_actor.GetLabelTextProperty(0),
            cube_axes_actor.GetLabelTextProperty(1),
            cube_axes_actor.GetLabelTextProperty(2),
        ]
        for prop in props:
            prop.SetColor(0.0, 0.0, 0.0)
            prop.SetFontFamily(parse_font_family('Arial'))
            prop.SetBold(False)
            # prop.SetFontSize(50)

        cube_axes_actor.SetScreenSize(self.main_widget.plot_formatting_options.font_size / 12 * 10.0)
        self.add_actor(cube_axes_actor)
        # bounding_box = self.add_bounding_box()
        # bounding_box.GetMapper().GetInput().SetBounds(0, 1000, 0, 1000, 0, 1000)
        # bounding_box.GetMapper().GetInput().Update()

    @staticmethod
    def getNumAxis():
        return 2

    def resizeEvent(self, event: QResizeEvent):
        if len(self.main_widget.data_display.axis_selections) >= self.getNumAxis():
            self.drawGraph()
        super().resizeEvent(event)

    def calculate_z_optimized(self, X, Y, function):
        """
           This function evaluates the function passed to it.
        """
        xs, ys = X.reshape(-1), Y.reshape(-1)
        points = np.ndarray((len(self.main_widget.data_display.parameters), len(xs)))

        parameter_value_list = self.main_widget.data_display.getValues()
        for p, v in parameter_value_list.items():
            points[p] = v
        param1 = self.main_widget.data_display.getAxisParameter(0).id
        param2 = self.main_widget.data_display.getAxisParameter(1).id
        if param1 >= 0:
            points[param1] = xs
        if param2 >= 0:
            points[param2] = ys

        z_value = function.evaluate(points)
        return z_value
