"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""

import math
import numpy

from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtCore import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport
from gui.Utils import formatFormula
from gui.Utils import formatNumber


#####################################################################


class GraphWidget(QWidget):
    """This class represents a Widget that is used to show the graph on
       the main window.
    """

    #####################################################################

    def __init__(self, main_widget, parent):
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

    def getMaxX(self):
        """
           This function returns the highest value of x that is being shown on x axis.
        """
        return self.max_x

    def getMaxY(self):
        """
           This function returns the highest value of Y that is being shown on x axis.
        """
        return self.max_y

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

        # colors
        self.BACKGROUND_COLOR = QColor("white")
        self.TEXT_COLOR = QColor("black")
        self.AXES_COLOR = QColor("black")
        self.AGGREGATE_MODEL_COLOR = QColor(self.main_widget.graph_color_list[0])
        self.DATA_POINT_COLOR = QColor(self.main_widget.graph_color_list[0]).darker(200)
        self.DATA_RANGE_COLOR = QColor(self.main_widget.graph_color_list[0]).darker(150)

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
        return self.aggregate_callpath and len(self.main_widget.getSelectedCallpath()) > 1

    @Slot(QPoint)
    def showContextMenu(self, point):
        """
          This function takes care of different options and their visibility in the context menu.
        """

        # selected_metric = self.main_widget.getSelectedMetric()
        selected_callpaths = self.main_widget.getSelectedCallpath()

        if not selected_callpaths:
            return

        menu = QMenu()
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

        menu.exec_(self.mapToGlobal(point))

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
          This function combine all callpathe shown on graph.
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
    def exportData(self):
        """
          This function allows to export the currently shown points and functions in a text format
        """

        selected_metric = self.main_widget.getSelectedMetric()
        selected_callpaths = self.main_widget.getSelectedCallpath()

        if not selected_callpaths:
            return

        # model_list = list()

        text = ''
        model_set = self.main_widget.getCurrentModel().models
        for selected_callpath in selected_callpaths:
            model = model_set[selected_callpath.path, selected_metric]
            if model is None:
                return
            model_function = model.hypothesis.function
            data_points = [p for (_, p) in self.calculateDataPoints(
                selected_metric, selected_callpath, True)]
            callpath_name = selected_callpath.name

            parameters = self.main_widget.experiment.parameters
            model_function_text = 'Model: ' + \
                                  formatFormula(
                                      model_function.to_string(parameters))

            data_points_text = '\n'.join(
                ('(' + str(x) + ', ' + str(y) + ')') for (x, y) in data_points)
            text += callpath_name + '\n' + data_points_text + \
                    '\n' + model_function_text + '\n\n'

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            "Exported data (text can be copied to the clipboard using the context menu):")
        msg.setInformativeText(text)
        msg.setWindowTitle("Export Data")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def drawGraph(self, paint):
        """
            This function is being called by paintEvent to draw the graph
        """

        # Get data
        model_set = self.main_widget.getCurrentModel()
        selected_metric = self.main_widget.getSelectedMetric()
        selected_callpaths = self.main_widget.getSelectedCallpath()
        if not selected_callpaths:
            return

        model_list = list()
        for selected_callpath in selected_callpaths:
            key = (selected_callpath.path, selected_metric)
            if key in model_set.models:
                model = model_set.models[key]
                model_list.append(model)

        # Calculate geometry constraints
        self.graph_width = self.frameGeometry().width() - self.left_margin - self.right_margin
        self.graph_height = self.frameGeometry().height() - self.top_margin - self.bottom_margin
        y = self.calculateMaxY(model_list) * 1.2
        self.max_y = y

        # Draw coordinate system
        self.drawAxis(paint, selected_metric)

        # Draw functions
        index_indicator = 0
        if not self.combine_all_callpath:
            for model in model_list:
                color = self.main_widget.getColorForCallPath(
                    selected_callpaths[index_indicator])
                self.drawModel(paint, model, color)
                index_indicator = index_indicator + 1
        else:
            color = self.main_widget.getColorForCallPath(selected_callpaths[0])
            self.drawAggregratedModel(paint, model_list)

        # Draw data points
        self.drawDataPoints(paint, selected_metric, selected_callpaths)

        # Draw legend
        self.drawLegend(paint)

    def drawDataPoints(self, paint, selectedMetric, selectedCallpaths):
        if self.show_datapoints is True:
            pen = QPen(self.DATA_POINT_COLOR)
            pen.setWidth(4)
            paint.setPen(pen)
            # data_points_list = list()
            for selected_callpath in selectedCallpaths:
                if self.datapoints_type == "outlier":
                    self.showOutlierPoints(
                        paint, selectedMetric, selected_callpath)

                else:
                    data_points = self.calculateDataPoints(
                        selectedMetric, selected_callpath)
                    self.plotPointsOnGraph(
                        paint, data_points)

    def drawLegend(self, paint):

        # drawing the graph legend
        px_between = 15
        callpath_color_dict = self.main_widget.get_callpath_color_map()
        dict_size = len(callpath_color_dict)
        font_size = int(self.main_widget.getFontSize())
        paint.setFont(QFont('Decorative', font_size))
        paint.setBrush(self.BACKGROUND_COLOR)
        pen = QPen(self.TEXT_COLOR)
        pen.setWidth(1)
        paint.setPen(pen)
        counter_increment = font_size + 3

        if self.combine_all_callpath is False:
            text_len = 0
            for callpath, color in callpath_color_dict.items():
                text_len = max(text_len, len(callpath.name))
            self.legend_width = 55 + text_len * (font_size - 1)
            self.legend_height = counter_increment * dict_size + px_between

            paint.drawRect(self.legend_x,
                           self.legend_y,
                           self.legend_width,
                           self.legend_height)
            counter = 0
            for callpath, color in callpath_color_dict.items():
                pen = QPen(QColor(color))
                pen.setWidth(2)
                paint.setPen(pen)
                paint.drawLine(self.legend_x + 5,
                               self.legend_y + px_between + counter,
                               self.legend_x + 35,
                               self.legend_y + px_between + counter)
                paint.setPen(self.TEXT_COLOR)
                paint.drawText(self.legend_x + 45,
                               self.legend_y + px_between + counter,
                               callpath.name)
                counter = counter + counter_increment

        else:
            text_len = 0
            callpath_list = list()

            for callpath, color in callpath_color_dict.items():
                callpath_list.append(callpath.name)
                text_len = max(text_len, text_len +
                               len(callpath.name))

            aggregated_callpath_name = str.join('+', callpath_list)
            self.legend_width = 55 + text_len * (font_size - 1)
            self.legend_height = counter_increment * 1 + px_between

            paint.drawRect(self.legend_x,
                           self.legend_y,
                           self.legend_width,
                           self.legend_height)
            pen = QPen(self.AGGREGATE_MODEL_COLOR)
            pen.setWidth(2)
            paint.setPen(pen)
            paint.drawLine(self.legend_x + 5,
                           self.legend_y + px_between,
                           self.legend_x + 35,
                           self.legend_y + px_between)
            paint.setPen(self.TEXT_COLOR)
            paint.drawText(self.legend_x + 45,
                           self.legend_y + px_between,
                           aggregated_callpath_name)

    def drawModel(self, paint, model, color):

        function = model.hypothesis.function

        cord_list = self.calculate_function(function, self.graph_width)

        pen = QPen(QColor(color))
        pen.setWidth(2)
        paint.setPen(pen)

        points = [
            QPointF(self.logicalXtoPixel(x), self.logicalYtoPixel(y)) for x, y in cord_list
        ]
        paint.drawPolyline(points)

    def drawAggregratedModel(self, paint, model_list):
        functions = list()
        for model in model_list:
            function = model.hypothesis.function
            functions.append(function)
        cord_list = self.calculate_aggregate_callpath_function(
            functions, self.graph_width)
        pen = QPen(self.AGGREGATE_MODEL_COLOR)
        pen.setWidth(2)
        paint.setPen(pen)
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

    def calculate_function(self, function, length_x_axis):
        """
         This function calculates the x values,
         based on the range that were provided &
         then it uses ExtraP_Function_Generic to calculate the correspoding y value
        """

        # m_x_lower_bound = 1
        number_of_x_points, x_list, x_values = self._calculate_evaluation_points(length_x_axis)

        y_list = function.evaluate(x_list).reshape(-1)

        cord_list = self._create_drawing_iterator(x_values, y_list)

        return cord_list

    def calculate_aggregate_callpath_function(self, functions, length_x_axis):
        """
        This function calculates the x values, based on the range that were provided & then it uses
        ExtraP_Function_Generic to calculate and aggregate the corresponding y value for all the model functions
        """
        number_of_x_points, x_list, x_values = self._calculate_evaluation_points(length_x_axis)

        y_list = numpy.zeros(number_of_x_points)

        previous = numpy.seterr(invalid='ignore')
        for function in functions:
            y_list += function.evaluate(x_list).reshape(-1)
        numpy.seterr(**previous)

        cord_list = self._create_drawing_iterator(x_values, y_list)

        return cord_list

    @staticmethod
    def _create_drawing_iterator(x_values, y_list):
        y_list[y_list == math.inf] = numpy.max(y_list[y_list != math.inf])
        y_list[y_list == -math.inf] = numpy.min(y_list[y_list != -math.inf])
        cord_list_before_filtering = zip(x_values, y_list)
        cord_list = ((x, y)
                     for x, y in cord_list_before_filtering
                     if not math.isnan(y) and y >= 0)
        return cord_list

    def _calculate_evaluation_points(self, length_x_axis):
        number_of_x_points = int(length_x_axis / 2)
        x_values = numpy.linspace(0, self.max_x, number_of_x_points)
        x_list = numpy.ndarray((len(self.main_widget.experiment.parameters), number_of_x_points))
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

    def calculateDataPoints(self, selectedMetric, selectedCallpath, ignore_limit=False):
        """ This function calculates datapoints to be marked on the graph
        """

        datapoints = self.main_widget.getCurrentModel().models[(selectedCallpath.path, selectedMetric)].measurements
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
        # TODO: fix this
        if self.show_datapoints:
            for model in modelList:
                y = max(model.predictions)
                y_max = max(y, y_max)

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
                y_agg = y_agg + function.evaluate(pv_list)
            y_max = max(y_agg, y_max)

        # Check the value at the end of the displayed interval
        for model in modelList:
            function = model.hypothesis.function
            y = function.evaluate(pv_list)
            y_max = max(y, y_max)

        # Check the value at the beginning of the displayed interval
        pv_list[param] = 1
        for model in modelList:
            function = model.hypothesis.function
            y = function.evaluate(pv_list)
            y_max = max(y, y_max)

        # Ensure that the maximum value is never too small
        if y_max < 0.000001:
            y_max = 1

        return y_max

    def showOutlierPoints(self, paint, selectedMetric, selectedCallpath):
        model_set = self.main_widget.getCurrentModel()
        datapoints = model_set.models[(selectedCallpath.path, selectedMetric)].measurements
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

    def mousePressEvent(self, QMouseEvent):

        self.clicked_x_pos = int(QMouseEvent.x())
        self.clicked_y_pos = int(QMouseEvent.y())
        # print ("clicked_x_pos, clicked_y_pos", self.clicked_x_pos, self.clicked_y_pos)

    def mouseReleaseEvent(self, releaseEvent):

        release_x_pos = int(releaseEvent.x())
        release_y_pos = int(releaseEvent.y())
        # print ("mouse release event release_x_pos", release_x_pos)

        # move legend id clicked on legend
        if (0 <= self.clicked_x_pos - self.legend_x <= self.legend_width) and (
                0 <= self.clicked_y_pos - self.legend_y <= self.legend_height):
            self.legend_x = self.legend_x + release_x_pos - self.clicked_x_pos
            self.legend_y = self.legend_y + release_y_pos - self.clicked_y_pos
            self.legend_x = max(0, self.legend_x)
            self.legend_y = max(0, self.legend_y)
            self.update()

        # scale graph
        else:
            release_x_pos = release_x_pos - self.left_margin
            clicked_x_pos = self.clicked_x_pos - self.left_margin

            if clicked_x_pos > 0 and release_x_pos > 0:
                self.setMax(0, clicked_x_pos / release_x_pos * self.getMaxX())
                self.update()

    @staticmethod
    def getNumAxis():
        return 1
