# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import importlib.resources
import itertools
import math
import typing

import numpy
from PySide6.QtCore import *  # @UnusedWildImport
from PySide6.QtGui import *  # @UnusedWildImport
from PySide6.QtWidgets import *  # @UnusedWildImport

from extrap.gui.Utils import formatFormula
from extrap.gui.Utils import formatNumber
from extrap.gui.plots.AbstractPlotWidget import AbstractPlotWidget

if typing.TYPE_CHECKING:
    from extrap.gui.MainWidget import MainWidget


#####################################################################


class GraphWidget(QWidget):
    """This class represents a Widget that is used to show the graph on
       the extrap window.
    """

    #####################################################################

    def __init__(self, main_widget: MainWidget, parent):
        super(GraphWidget, self).__init__(parent)

        self.main_widget = main_widget
        self.initUI()
        self.set_initial_value()
        self.setMouseTracking(True)

    def initUI(self):
        self.setMinimumWidth(300)
        self.setMinimumHeight(300)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.show()

    def setMax(self, axis, maxX):
        """ This function sets the highest value of x that is being shown on x axis.
        """
        if axis == 0:
            self.main_widget.data_display.setMaxValue(0, maxX)
            self.max_x = maxX

        else:
            print("[EXTRAP:] Error: Set maximum for axis other than X-axis.")

    def logicalXtoPixel(self, lValue):
        """
            This function converts an X-value from logical into pixel coordinates.
        """
        return self.left_margin + lValue * self.graph_width / self.max_x

    def pixelXtoLogical(self, pValue):
        """
            This function converts an X-value from pixel into logical coordinates.
        """
        return (pValue - self.left_margin) * self.max_x / self.graph_width

    def logicalYtoPixel(self, lValue):
        """
            This function converts an Y-value from logical into pixel coordinates.
        """
        return self.top_margin + (self.max_y - lValue) * self.graph_height / self.max_y

    def pixelYtoLogical(self, pValue):
        """
            This function converts an Y-value from pixel into logical coordinates.
        """
        return (self.graph_height + self.top_margin - pValue) * self.max_y / self.graph_height

    def paintEvent(self, event):
        paint = QPainter()
        paint.begin(self)
        paint.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.drawGraph(paint)
        paint.end()

    # noinspection PyAttributeOutsideInit
    def set_initial_value(self):
        """
          This function sets the initial value for different parameters required for graph.
        """
        # Basic geometry constants
        self.max_x = 40
        self.left_margin = 80
        self.bottom_margin = 80
        self.top_margin = 20
        self.right_margin = 20
        self.legend_x = 100  # X-coordinate of the upper left corner of the legend
        self.legend_y = 20  # Y-coordinate of the upper left corner of the legend

        # the actual value for below 3 variables will be set later in the code
        self.max_y = 0
        self.graph_height = 0
        self.graph_width = 0
        self.legend_width = 0
        self.legend_height = 0
        self.clicked_x_pos = None
        self.clicked_y_pos = None

        # colors
        self.BACKGROUND_COLOR = QColor("white")
        self.TEXT_COLOR = QColor("black")
        self.AXES_COLOR = QColor("black")
        self.AGGREGATE_MODEL_COLOR = QColor(self.main_widget.model_color_map.default_color)
        self.DATA_POINT_COLOR = QColor(self.main_widget.model_color_map.default_color).darker(200)
        self.DATA_RANGE_COLOR = QColor(self.main_widget.model_color_map.default_color).darker(150)

        self.minimum_number_points_marked = 2
        self.aggregate_callpath = False
        self.datapoints_type = ""

        self.datapointType_Int_Map = {
            'min': 1, 'mean': 2, 'max': 3, 'median': 4, 'standardDeviation': 5, 'outlier': 6}

    @property
    def show_datapoints(self):
        return not self.combine_all_callpath and self.datapoints_type != ''

    @property
    def combine_all_callpath(self):
        return self.aggregate_callpath and len(self.main_widget.get_selected_call_tree_nodes()) > 1

    def create_context_menu(self) -> typing.Optional[QMenu]:
        # selected_metric = self.main_widget.get_selected_metric()
        selected_callpaths = self.main_widget.get_selected_call_tree_nodes()

        if not selected_callpaths:
            return None

        menu = QMenu(self)
        points_group = QActionGroup(self)
        points_group.setEnabled(not self.combine_all_callpath)

        data_point_selection = [
            # To show data points for mean values
            ("Show Mean Points", 'mean'),
            # To show data points for min values
            ("Show Minimum Points", 'min'),
            # To show data points for max values
            ("Show Maximum Points", 'max'),
            # To show data points for median values
            ("Show Median Points", 'median'),
            # To show data points for standard deviation values
            ("Show Standard Deviation Points", 'standardDeviation'),
            # To show outlier points
            ("Show all data points", 'outlier'),
            # Hiding any datapoints
            ("Hide Points", ''),
        ]
        for name, type in data_point_selection:
            action = points_group.addAction(name)
            action.setCheckable(True)
            action.setData(type)
            action.setChecked(self.datapoints_type == type)
            action.triggered.connect(self.selectDataPoints)
            menu.addAction(action)

        # Combining and disjoing Multiple callpaths
        menu.addSeparator()
        group = QActionGroup(self)
        group.setEnabled(len(selected_callpaths) > 1)
        group.setExclusive(True)

        action = group.addAction("Combine callpaths")
        action.setCheckable(True)
        action.setChecked(self.aggregate_callpath)
        action.triggered.connect(self.combineCallPaths)
        menu.addAction(action)

        action1 = group.addAction("Show all callpaths")
        action1.setCheckable(True)
        action1.setChecked(not self.aggregate_callpath)
        action1.triggered.connect(self.showAllCallPaths)
        menu.addAction(action1)

        # Export
        menu.addSeparator()
        exportDataAction = menu.addAction("Export Data")
        exportDataAction.triggered.connect(self.exportData)
        screenshotAction = menu.addAction("Screenshot")
        screenshotAction.triggered.connect(self.screenshot)

        return menu

    @Slot(QPoint)
    def showContextMenu(self, point):
        """
          This function takes care of different options and their visibility in the context menu.
        """

        menu = self.create_context_menu()
        if not menu:
            return

        menu.exec(self.mapToGlobal(point))

    @Slot()
    def selectDataPoints(self):
        """
          This function hides all the datapoints that is being shown on graph.
        """
        self.datapoints_type = QObject.sender(self).data()
        self.update()

    @Slot()
    def combineCallPaths(self):
        """
          This function combines all callpaths shown on graph.
        """
        self.aggregate_callpath = True
        self.update()

    @Slot()
    def showAllCallPaths(self):
        """
          This function shows all callpaths.
        """
        self.aggregate_callpath = False
        self.update()

    @Slot()
    def screenshot(self):
        selected_callpaths = self.main_widget.get_selected_call_tree_nodes()
        selected_metric = self.main_widget.get_selected_metric()

        name_addition = "-"
        if selected_metric:
            name_addition = f"-{selected_metric}-"
        if selected_callpaths:
            name_addition += ','.join((c.name for c in selected_callpaths))

        self.main_widget.screenshot(target=self, name_addition=name_addition)

    @Slot()
    def exportData(self):
        """
          This function allows to export the currently shown points and functions in a text format
        """

        text = ''
        models, _ = self.main_widget.get_selected_models()
        if models is None:
            return

        for model in models:
            callpath_name = model.callpath.name
            data_points = [p for (_, p) in self.calculateDataPoints(model, True)]
            parameters = self.main_widget.getExperiment().parameters
            model_function_text = 'Model: ' + formatFormula(model.hypothesis.function.to_string(*parameters))

            data_points_text = '\n'.join(
                ('(' + str(x) + ', ' + str(y) + ')') for (x, y) in data_points)
            text += callpath_name + '\n' + data_points_text + '\n' + model_function_text + '\n\n'

        msg = QDialog()
        msg.setWindowTitle("Export Data")
        msg.setFixedSize(600, 400)
        layout = QGridLayout()
        layout.addWidget(QLabel("Exported data (text can be copied to the clipboard using the context menu):"))

        info_text = QTextEdit()
        info_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        info_text.setText(text)
        layout.addWidget(info_text)

        btn = QPushButton('OK', msg)
        btn.setDefault(True)
        btn.clicked.connect(msg.accept)
        layout.addWidget(btn)
        msg.setLayout(layout)
        msg.exec()

    def drawGraph(self, paint):
        """
            This function is being called by paintEvent to draw the graph
        """

        # Get data
        model_list, selected_call_nodes = self.main_widget.get_selected_models()
        if not model_list:
            return

        plot_options = self.main_widget.plot_formatting_options
        paint.setFont(QFont(plot_options.font_family, plot_options.font_size))

        # Calculate geometry constraints
        self.graph_width = self.frameGeometry().width() - self.left_margin - self.right_margin
        self.graph_height = self.frameGeometry().height() - self.top_margin - self.bottom_margin
        y = self.calculateMaxY(model_list) * 1.2
        self.max_y = y

        # Draw coordinate system
        self.drawAxis(paint, self.main_widget.get_selected_metric())

        # Draw functionss
        index_indicator = 0
        if not self.combine_all_callpath:
            for model, call_node in zip(model_list, selected_call_nodes):
                color = self.main_widget.model_color_map[call_node]
                self.drawModel(paint, model, color)
        else:
            # main_widget = self.main_widget
            # color = main_widget.model_color_map[selected_call_nodes[0]]
            self.drawAggregratedModel(paint, model_list)

        # Draw data points
        self.drawDataPoints(paint, model_list)

        # Draw legend
        paint.setFont(QFont(plot_options.font_family, plot_options.legend_font_size))
        self.drawLegend(paint)

    def drawDataPoints(self, paint, selected_models):
        if self.show_datapoints is True:
            pen = QPen(self.DATA_POINT_COLOR)
            pen.setWidth(4)
            paint.setPen(pen)
            # data_points_list = list()
            for selected_model in selected_models:
                if self.datapoints_type == "outlier":
                    self.showOutlierPoints(paint, selected_model)

                else:
                    data_points = self.calculateDataPoints(selected_model)
                    self.plotPointsOnGraph(paint, data_points)

    def drawLegend(self, paint):
        # drawing the graph legend

        widget = self.main_widget
        callpath_color_dict = widget.model_color_map
        dict_size = len(callpath_color_dict)
        paint.setBrush(self.BACKGROUND_COLOR)
        font_metrics = paint.fontMetrics()
        pen = QPen(self.TEXT_COLOR)
        pen.setWidth(1)
        paint.setPen(pen)

        px_between = 2
        font_height = font_metrics.height()
        counter_increment = font_height + px_between
        line_offset = font_height / 2
        left_margin_text = self.legend_x + 45

        if self.combine_all_callpath is False:
            text_len = 0
            for callpath, color in callpath_color_dict.items():
                text_len = max(text_len, font_metrics.horizontalAdvance(callpath.name))
            self.legend_width = 55 + text_len
            self.legend_height = counter_increment * dict_size + 3 * px_between

            paint.drawRect(self.legend_x,
                           self.legend_y,
                           self.legend_width,
                           self.legend_height)
            counter = 2 * px_between
            for callpath, color in callpath_color_dict.items():
                pen = QPen(QColor(color))
                pen.setWidth(2)
                paint.setPen(pen)
                paint.drawLine(QPoint(self.legend_x + 5,
                                      self.legend_y + counter + line_offset),
                               QPoint(self.legend_x + 35,
                                      self.legend_y + counter + line_offset))
                paint.setPen(self.TEXT_COLOR)
                paint.drawText(QRect(left_margin_text, self.legend_y + counter,
                                     text_len, font_height),
                               Qt.TextFlag.TextDontClip, callpath.name)

                counter = counter + counter_increment

        else:

            aggregated_callpath_name = ' + '.join(callpath.name for callpath, color in callpath_color_dict.items())

            bounding_rect_text = font_metrics.boundingRect(
                QRect(left_margin_text, self.legend_y + 2 * px_between, self.graph_width - left_margin_text,
                      self.graph_height - self.legend_y + 2 * px_between),
                Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextDontClip, aggregated_callpath_name)

            text_len = bounding_rect_text.width()
            self.legend_width = 55 + text_len

            self.legend_height = bounding_rect_text.height() + 4 * px_between

            paint.drawRect(self.legend_x,
                           self.legend_y,
                           self.legend_width,
                           self.legend_height)
            pen = QPen(self.AGGREGATE_MODEL_COLOR)
            pen.setWidth(2)
            paint.setPen(pen)
            paint.drawLine(self.legend_x + 5,
                           self.legend_y + 2 * px_between + line_offset,
                           self.legend_x + 35,
                           self.legend_y + 2 * px_between + line_offset)
            paint.setPen(self.TEXT_COLOR)
            paint.drawText(bounding_rect_text,
                           Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextDontClip, aggregated_callpath_name)

    def drawModel(self, paint, model, color):
        function = model.hypothesis.function

        cord_lists = self.calculate_function(function, self.graph_width)

        pen = QPen(QColor(color))
        pen.setWidth(2)
        paint.setPen(pen)

        for cord_list in cord_lists:
            points = [
                QPointF(self.logicalXtoPixel(x), self.logicalYtoPixel(y)) for x, y in cord_list
            ]
            paint.drawPolyline(points)

    def drawAggregratedModel(self, paint, model_list):
        functions = list()
        for model in model_list:
            function = model.hypothesis.function
            functions.append(function)
        cord_lists = self.calculate_aggregate_callpath_function(
            functions, self.graph_width)
        pen = QPen(self.AGGREGATE_MODEL_COLOR)
        pen.setWidth(2)
        paint.setPen(pen)
        for cord_list in cord_lists:
            points = [
                QPointF(self.logicalXtoPixel(x), self.logicalYtoPixel(y)) for x, y in cord_list
            ]
            paint.drawPolyline(points)

    def drawAxis(self, paint, selectedMetric):
        # Determing the number of divisions to be marked on x axis such that there is a minimum distance of 100 pixels
        # between two of them based on that, then calculating distance between two marks on y axis.
        x_offset = 100
        x_origin = self.left_margin
        y_origin = self.top_margin + self.graph_height
        y_other_end = self.top_margin
        x_other_end = self.graph_width + self.left_margin

        num_points_marked_on_x_axis = int(self.graph_width / x_offset)
        if num_points_marked_on_x_axis < self.minimum_number_points_marked:
            num_points_marked_on_x_axis = self.minimum_number_points_marked
        x_offset = self.graph_width / num_points_marked_on_x_axis

        num_points_marked_on_y_axis = int(self.graph_height / x_offset)
        if num_points_marked_on_y_axis == 0:
            num_points_marked_on_y_axis = 1
        y_offset = self.graph_height / num_points_marked_on_y_axis

        x_to_mark_cal = self.get_axis_mark_list(
            self.max_x, num_points_marked_on_x_axis)
        y_to_mark_cal = self.get_axis_mark_list(
            self.max_y, num_points_marked_on_y_axis)

        x_to_mark = self.format_numbers_to_be_displayed(x_to_mark_cal)
        y_to_mark = self.format_numbers_to_be_displayed(y_to_mark_cal)

        # setting number of sub division to be marked on x axis and y axis
        number_of_intermediate_points_on_x = 2
        number_of_intermediate_points_on_y = 4

        # drawing the rectangular region that would contain the graph
        paint.setPen(self.BACKGROUND_COLOR)
        paint.setBrush(self.BACKGROUND_COLOR)
        paint.drawRect(self.frameGeometry())
        # drawing x axis and y axis
        paint.setPen(self.AXES_COLOR)
        paint.drawLine(x_origin, y_origin, x_other_end, y_origin)
        paint.drawLine(x_origin, y_other_end, x_origin, y_origin)

        # marking divions and subdivisons on x axis
        y = y_origin
        paint.drawText(self.left_margin - 5, y + 30, "0")
        if x_to_mark[0] != 0:
            intermediate_x_offset = x_offset / \
                                    (number_of_intermediate_points_on_x + 1)
            intermediate_x = self.left_margin + intermediate_x_offset
            for _ in range(0, number_of_intermediate_points_on_x, +1):
                paint.drawLine(intermediate_x, y - 3, intermediate_x, y)
                intermediate_x = intermediate_x + intermediate_x_offset
            # x_last_position = self.left_margin

        # removing the  "_" sign form beginning if x_label has
        x_label = self.main_widget.data_display.getAxisParameter(0).name
        if x_label.startswith("_"):
            x_label = x_label[1:]

        for i in range(len(x_to_mark)):
            x = self.logicalXtoPixel(x_to_mark_cal[i])
            if i == len(x_to_mark) - 1:
                x = round(x)

            if i == (int(len(x_to_mark) / 2)):
                paint.drawText(x, y + 50, x_label)

            intermediate_x_offset = x_offset / (number_of_intermediate_points_on_x + 1)
            intermediate_x = x + intermediate_x_offset
            for _ in range(0, number_of_intermediate_points_on_x, +1):
                paint.drawLine(intermediate_x, y - 3, intermediate_x, y)
                intermediate_x = intermediate_x + intermediate_x_offset
            paint.drawLine(x, y - 6, x, y)
            paint.drawText(x - 15, y + 30, str(x_to_mark[i]))
            # x_last_position = x

            for y_value in range((y_origin - 7), y_other_end, -3):
                paint.drawPoint(x, y_value)

        # marking divions and subdivisons on y axis
        x = self.left_margin

        if y_to_mark[0] != 0:
            intermediate_y_offset = y_offset / \
                                    (number_of_intermediate_points_on_y + 1)
            intermediate_y = y_origin - intermediate_y_offset
            for _ in range(0, number_of_intermediate_points_on_y, +1):
                paint.drawLine(x_origin, intermediate_y,
                               x_origin + 3, intermediate_y)
                intermediate_y = intermediate_y - intermediate_y_offset
            # y_last_position = y_origin

        for j in range(len(y_to_mark)):
            y = self.logicalYtoPixel(y_to_mark_cal[j])

            if j + 1 == (int(len(y_to_mark) / 2)):
                paint.drawText(5, y - (y_offset / 2), selectedMetric.name)

            intermediate_y_offset = y_offset / (number_of_intermediate_points_on_y + 1)
            intermediate_y = y - intermediate_y_offset
            for _ in range(0, number_of_intermediate_points_on_y, +1):
                paint.drawLine(x_origin, intermediate_y,
                               x_origin + 3, intermediate_y)
                intermediate_y = intermediate_y - intermediate_y_offset
            paint.drawLine(x_origin, y, x_origin + 6, y)
            paint.drawText(x_origin - 70, y + 5, str(y_to_mark[j]))
            # y_last_position = y
            for x_value in range((x_origin + 7), x_other_end, +3):
                paint.drawPoint(x_value, y)

    def calculate_function(self, function, length_x_axis, x_min=None, x_max=None):
        """
         This function calculates the x values,
         based on the range that were provided &
         then it uses ExtraP_Function_Generic to calculate the correspoding y value
        """

        # m_x_lower_bound = 1
        if x_min != None and x_max != None:
            number_of_x_points, x_list, x_values = self._calculate_evaluation_points(length_x_axis, x_min, x_max)
        else:
            number_of_x_points, x_list, x_values = self._calculate_evaluation_points(length_x_axis)
        previous = numpy.seterr(invalid='ignore', divide='ignore')
        y_list = function.evaluate(x_list).reshape(-1)
        numpy.seterr(**previous)
        cord_list = self._create_drawing_iterator(x_values, y_list)

        return cord_list

    def calculate_aggregate_callpath_function(self, functions, length_x_axis):
        """
        This function calculates the x values, based on the range that were provided & then it uses
        ExtraP_Function_Generic to calculate and aggregate the corresponding y value for all the model functions
        """
        number_of_x_points, x_list, x_values = self._calculate_evaluation_points(length_x_axis)

        y_list = numpy.zeros(number_of_x_points)

        previous = numpy.seterr(invalid='ignore', divide='ignore')
        for function in functions:
            y_list += function.evaluate(x_list).reshape(-1)
        numpy.seterr(**previous)

        cord_list = self._create_drawing_iterator(x_values, y_list)

        return cord_list

    def _create_drawing_iterator(self, x_values, y_list):
        y_list[y_list == math.inf] = numpy.max(y_list[y_list != math.inf])
        y_list[y_list == -math.inf] = numpy.min(y_list[y_list != -math.inf])

        cord_list_before_filtering = zip(x_values, y_list)
        cord_groups = itertools.groupby(cord_list_before_filtering, key=lambda k: not math.isnan(k[1]) and 0 <= k[1])
        cord_list = (v for k, v in cord_groups if k)
        return cord_list

    def _calculate_evaluation_points(self, length_x_axis, x_min=None, x_max=None):
        number_of_x_points = int(length_x_axis / 2)
        if x_min != None and x_max != None:
            x_values = numpy.linspace(x_min, x_max, number_of_x_points)
        else:
            x_values = numpy.linspace(0, self.max_x, number_of_x_points)
        x_list = numpy.ndarray((len(self.main_widget.getExperiment().parameters), number_of_x_points))
        param = self.main_widget.data_display.getAxisParameter(0).id
        parameter_value_list = self.main_widget.data_display.getValues()
        for i, val in parameter_value_list.items():
            x_list[i] = numpy.repeat(val, number_of_x_points)
        x_list[param] = x_values
        return number_of_x_points, x_list, x_values

    @staticmethod
    def get_axis_mark_list(max_val, number_of_points):
        """ This function takes as parameter as number of points to be marked on an axis,
              the maximum value to be marked on axis and returns a list of points to be marked on axis.
        """

        axis_points_to_mark = list()
        axis_range = float((float(max_val - 0)) / number_of_points)
        value = 0
        for _ in range(1, number_of_points + 1):
            value = value + axis_range
            if value < 1:
                digits_after_point = int(math.log10(1 / value)) + 2
                value_to_append = float(
                    "{0:.{1}f}".format(value, digits_after_point))
            else:
                value_to_append = float("{0:.2f}".format(value))
            axis_points_to_mark.append(float(value_to_append))
        return axis_points_to_mark

    def calculate_absolute_position(self, cord_list, identifier):
        """ This function calculates the absolute position of the points on the
            graph widget wrt the coordinate system.
        """

        absolute_position_list = list()
        if identifier == "x":
            for cord in cord_list:
                cur_pixel = self.logicalXtoPixel(cord)
                absolute_position_list.append(cur_pixel)
        elif identifier == "y":
            for cord in cord_list:
                cur_pixel = self.logicalYtoPixel(cord)
                absolute_position_list.append(cur_pixel)
        return absolute_position_list

    # function to format the numbers to be marked on the graph
    @staticmethod
    def format_numbers_to_be_displayed(value_list):
        """ This function formats and beautify the number to be shown on the graph.
        """
        new_mark_list = list()
        for value in value_list:
            if value >= 10:
                precision = 1
            else:
                precision = 2
            value_str = formatNumber(str(value), precision)
            new_mark_list.append(value_str)
        return new_mark_list

    @staticmethod
    def reduce_length(value):
        """ This function formats and beautify the number to be shown on the graph.
        """
        splitted_value = value.split('e')
        first_part = float(splitted_value[0])
        first_part = round(first_part, 2)
        return '{:g}'.format(float(first_part)) + "e" + ''.join(splitted_value[1])

    def calculateDataPoints(self, model, ignore_limit=False):
        """ This function calculates datapoints to be marked on the graph
        """
        datapoints = model.measurements

        parameter_datapoint = self.main_widget.data_display.getAxisParameter(0).id
        datapoint_x_absolute_pos_list = list()
        datapoint_y_absolute_pos_list = list()
        datapoint_x_list = list()
        datapoint_y_list = list()

        if self.datapoints_type == "min":
            datapoint_list = self.getDataPoints(
                datapoints, parameter_datapoint, ignore_limit, lambda d: d.minimum)
        elif self.datapoints_type == "mean":
            datapoint_list = self.getDataPoints(
                datapoints, parameter_datapoint, ignore_limit, lambda d: d.mean)
        elif self.datapoints_type == "max":
            datapoint_list = self.getDataPoints(
                datapoints, parameter_datapoint, ignore_limit, lambda d: d.maximum)
        elif self.datapoints_type == "median":
            datapoint_list = self.getDataPoints(
                datapoints, parameter_datapoint, ignore_limit, lambda d: d.median)
        elif self.datapoints_type == "standardDeviation":
            datapoint_list = self.getDataPoints(
                datapoints, parameter_datapoint, ignore_limit, lambda d: d.std)
            # TODO think about drawing as bar around value
        else:
            datapoint_list = None

        if datapoint_list:
            datapoint_x_list, datapoint_y_list = zip(*datapoint_list)
            datapoint_x_absolute_pos_list = self.calculate_absolute_position(
                datapoint_x_list, "x")
            datapoint_y_absolute_pos_list = self.calculate_absolute_position(
                datapoint_y_list, "y")

        datapoint_on_graph_values = zip(datapoint_x_absolute_pos_list, datapoint_y_absolute_pos_list)
        datapoint_actual_values = zip(datapoint_x_list, datapoint_y_list)
        return list(zip(datapoint_on_graph_values, datapoint_actual_values))

    def getDataPoints(self, datapoints, parameter_datapoint, ignore_limit, key):
        """
        This function calculates datapoints with property selected by the
        key function to be marked on the graph
        """
        return [
            (dp.coordinate[parameter_datapoint], key(dp))
            for dp in datapoints
            if (dp.coordinate[parameter_datapoint] <= self.max_x or ignore_limit)
        ]

    def calculateMaxY(self, modelList):
        y_max = 0
        pv_list = self.main_widget.data_display.getValues()
        param = self.main_widget.data_display.getAxisParameter(0).id
        pv_list[param] = self.max_x

        # Check the maximum value of a displayed data point
        if self.show_datapoints:
            for model in modelList:
                y = max(model.predictions)
                y_max = max(y, y_max)

        previous = numpy.seterr(invalid='ignore', divide='ignore')

        if self.combine_all_callpath:
            y_agg = 0
            for model in modelList:
                function = model.hypothesis.function
                y_agg = y_agg + function.evaluate(pv_list)
            y_max = max(y_agg, y_max)

            pv_list[param] = 1
            y_agg = 0
            for model in modelList:
                function = model.hypothesis.function
                y = function.evaluate(pv_list)
                if math.isinf(y):
                    y = max(model.predictions)
                y_agg += y
            y_max = max(y_agg, y_max)

        # Check the value at the end of the displayed interval
        for model in modelList:
            function = model.hypothesis.function
            y = function.evaluate(pv_list)
            if math.isinf(y):
                y = max(model.predictions)
            y_max = max(y, y_max)

        # Check the value at the beginning of the displayed interval
        pv_list[param] = 1
        for model in modelList:
            function = model.hypothesis.function
            y = function.evaluate(pv_list)
            if math.isinf(y):
                y = max(model.predictions)
            y_max = max(y, y_max)

        numpy.seterr(**previous)
        # Ensure that the maximum value is never too small
        if y_max < 0.000001:
            y_max = 1

        return y_max

    def showOutlierPoints(self, paint, selected_model):
        if isinstance(selected_model, list):
            parameter_datapoint = self.main_widget.data_display.getAxisParameter(0).id
            for i in range(len(selected_model)):
                datapoints = selected_model[i].measurements
                for datapoint in datapoints:
                    x_value = datapoint.coordinate[parameter_datapoint]
                    if x_value <= self.max_x:
                        y_min_value = datapoint.minimum
                        y_max_value = datapoint.maximum
                        y_mean_value = datapoint.mean
                        y_median_value = datapoint.median
                        self.plotOutliers(paint, x_value, y_min_value,
                                          y_mean_value, y_median_value, y_max_value)

        else:
            datapoints = selected_model.measurements
            parameter_datapoint = self.main_widget.data_display.getAxisParameter(0).id
            for datapoint in datapoints:
                x_value = datapoint.coordinate[parameter_datapoint]
                if x_value <= self.max_x:
                    y_min_value = datapoint.minimum
                    y_max_value = datapoint.maximum
                    y_mean_value = datapoint.mean
                    y_median_value = datapoint.median
                    self.plotOutliers(paint, x_value, y_min_value,
                                      y_mean_value, y_median_value, y_max_value)

    def plotOutliers(self, paint, x_value, y_min_value, y_mean_value, y_median_value, y_max_value):
        # create cordinates and merge them to list
        x_list = [x_value, x_value, x_value, x_value]
        y_list = [y_min_value, y_median_value, y_mean_value, y_max_value]
        # outlier_list = list(zip(x_list, y_list))
        # print("In plotOutliers", outlier_list)

        outlier_x_absolute_pos_list = self.calculate_absolute_position(
            x_list, "x")
        outlier_y_absolute_pos_list = self.calculate_absolute_position(
            y_list, "y")
        max_y = max(outlier_y_absolute_pos_list)
        min_y = min(outlier_y_absolute_pos_list)
        x = max(outlier_x_absolute_pos_list)

        pen = QPen(self.DATA_RANGE_COLOR)
        pen.setWidth(2)
        paint.setPen(pen)
        paint.drawLine(x, min_y, x, max_y)

        pen = QPen(self.DATA_POINT_COLOR)
        pen.setWidth(2)
        paint.setPen(pen)
        outlier_on_graph_list = list(
            zip(outlier_x_absolute_pos_list, outlier_y_absolute_pos_list))
        for (x_cordinate, y_cordinate) in outlier_on_graph_list:
            # paint.drawPoint(x_cordinate, y_cordinate)
            paint.drawLine(x_cordinate - 2, y_cordinate,
                           x_cordinate + 2, y_cordinate)

    @staticmethod
    def plotPointsOnGraph(paint, dataPoints):
        """ This function plots datapoints on the graph
        """
        for (x_cordinate, y_cordinate), (x_actual_val, y_actual_val) in dataPoints:
            paint.drawPoint(x_cordinate, y_cordinate)
            # displayString="(" + str(x_actual_val)+ "," + str(y_actual_val) + ")"
            # paint.drawText((x_cordinate-20), y_cordinate, displayString)
        # paint.drawText((x_cordinate/2), y_cordinate-10, selected_callpath.name)

    def mousePressEvent(self, event):
        if event.buttons() & ~Qt.LeftButton:
            return
        x = int(event.x())
        y = int(event.y())
        if (0 <= x - self.legend_x <= self.legend_width) and (0 <= y - self.legend_y <= self.legend_height) or \
                (0 <= x - self.left_margin <= self.graph_width) and (0 <= y - self.top_margin <= self.graph_height):
            self.clicked_x_pos = x
            self.clicked_y_pos = y
            # print ("clicked_x_pos, clicked_y_pos", self.clicked_x_pos, self.clicked_y_pos)

    def mouseMoveEvent(self, event):
        if (event.buttons() & ~Qt.LeftButton):
            return
        if self.clicked_x_pos is None or self.clicked_y_pos is None:
            return
        x_pos = int(event.x())
        y_pos = int(event.y())
        # print ("mouse release event x_pos", x_pos)

        # move legend id clicked on legend
        if (0 <= self.clicked_x_pos - self.legend_x <= self.legend_width) and (
                0 <= self.clicked_y_pos - self.legend_y <= self.legend_height):
            self.legend_x = self.legend_x + x_pos - self.clicked_x_pos
            self.legend_y = self.legend_y + y_pos - self.clicked_y_pos
            self.legend_x = max(0, self.legend_x)
            self.legend_y = max(0, self.legend_y)
            self.update()

        # scale graph
        else:
            relative_x = (self.clicked_x_pos - x_pos) * 2
            self.setMax(0, (self.graph_width + relative_x) / self.graph_width * self.max_x)
            self.update()

        self.clicked_x_pos = x_pos
        self.clicked_y_pos = y_pos

    def mouseReleaseEvent(self, event):
        self.clicked_x_pos = None
        self.clicked_y_pos = None

    @staticmethod
    def getNumAxis():
        return 1


class GraphWrapperWidget(AbstractPlotWidget):
    _menu_icon_path = importlib.resources.path("extrap.gui.resources", "menu.svg").__enter__()

    def setMax(self, axis, maxValue):
        raise NotImplementedError("The assignment should happen in __init__")

    def set_initial_value(self):
        raise NotImplementedError("The assignment should happen in __init__")

    @staticmethod
    def getNumAxis():
        return GraphWidget.getNumAxis()

    def __init__(self, main_widget: MainWidget, parent):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self._main_widget = main_widget
        self._graph_widget = GraphWidget(main_widget, self)
        grid.addWidget(self._graph_widget, 0, 0)

        self.setMax = self._graph_widget.setMax
        self.set_initial_value = self._graph_widget.set_initial_value

        self.context_menu_button = QToolButton(self)

        self.context_menu_button.setIcon(QIcon(str(GraphWrapperWidget._menu_icon_path)))
        self.context_menu_button.setText("Graph options")
        self.context_menu_button.setEnabled(False)
        self.context_menu_button.clicked.connect(self._context_menu_button_clicked)
        button_layout = QVBoxLayout()
        button_layout.setContentsMargins(15, 15, 15, 15)
        button_layout.addWidget(self.context_menu_button)

        grid.addLayout(button_layout, 0, 0, alignment=Qt.AlignLeft | Qt.AlignBottom)

    def _context_menu_button_clicked(self):
        if self._graph_widget:
            menu = self._graph_widget.create_context_menu()
            self.context_menu_button.setMenu(menu)
            self.context_menu_button.showMenu()
            self.context_menu_button.setMenu(None)
            # button_pos = self.context_menu_button.pos()
            # button_pos -= QPoint(0, self.context_menu_button.height())
            # graph_widget.showContextMenu(button_pos)

    def drawGraph(self):
        model_list, selected_call_nodes = self._main_widget.get_selected_models()
        if model_list:
            self.context_menu_button.setEnabled(True)
        else:
            self.context_menu_button.setEnabled(False)
        self._graph_widget.update()
