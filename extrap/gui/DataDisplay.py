# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from collections import defaultdict

from PySide2.QtCore import *  # @UnusedWildImport
from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport

from extrap.entities.parameter import Parameter
from extrap.gui.AdvancedPlotWidget import AdvancedPlotWidget
from extrap.gui.GraphWidget import GraphWidget
from extrap.gui.plots.AllFunctionsAsDifferentSurfacePlotWidget import AllFunctionsAsDifferentSurfacePlot
from extrap.gui.plots.AllFunctionsAsOneSurfacePlotWidget import AllFunctionsAsOneSurfacePlot
from extrap.gui.plots.DominatingFunctionsAsSingleScatterPlotWidget import DominatingFunctionsAsSingleScatterPlot
from extrap.gui.plots.HeatMapGraphWidget import HeatMapGraph
from extrap.gui.plots.InterpolatedContourDisplayWidget import InterpolatedContourDisplay
from extrap.gui.plots.IsolinesDisplayWidget import IsolinesDisplay
from extrap.gui.plots.MaxZAsSingleSurfacePlotWidget import MaxZAsSingleSurfacePlot
from extrap.gui.plots.MeasurementPointsPlotWidget import MeasurementPointsPlot
#####################################################################
from extrap.gui.plots.line_graph_plus import LineGraphPlus

MIN_PARAM_VALUE = 0.01
MAX_PARAM_VALUE = 2000000000


class AxisSelection(QWidget):
    """ This class is a helper class for the class DataDisplay.
        It represents one parameter in the data display which
        shown on one of the graph axis. It allows to set the maximum
        value for the axis in the graph.
    """
    #####################################################################

    max_values = [10, 10, 10]

    def __init__(self, manager, parent, index: int, parameters):
        super(AxisSelection, self).__init__(parent)
        self.manager = manager
        self.index: int = index
        self.initUI(parameters)
        self.updateDisplay()
        self.old_name = self.getParameter()

    # noinspection PyAttributeOutsideInit
    def initUI(self, parameters):
        self.grid = QGridLayout(self)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(3, 1)
        if self.index == 0:
            label1 = QLabel("X-axis")
            # label1.setMinimumHeight( 75 )
        elif self.index == 1:
            label1 = QLabel("Y-axis")
        elif self.index == 2:
            label1 = QLabel("Z-axis")
        else:
            label1 = QLabel("Axis " + str(self.index))
        label1.setMinimumWidth(75)
        self.combo_box = QComboBox(self)
        # self.combo_box.setMinimumWidth( 75 )
        self.combo_box.setMinimumHeight(20)
        for i in range(0, len(parameters)):
            self.combo_box.addItem(parameters[i].name)
        self.combo_box.setEnabled(self.index < len(parameters))
        self.combo_box.setCurrentIndex(self.index)

        self.combo_box.currentIndexChanged.connect(self.parameter_selected)

        label2 = QLabel("max.")
        label2.setMinimumWidth(40)
        label2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.max_edit = QDoubleSpinBox()
        self.max_edit.setMinimum(MIN_PARAM_VALUE)
        self.max_edit.setMinimumHeight(25)
        self.max_edit.setMaximum(MAX_PARAM_VALUE)
        if self.index <= 2:
            self.max_edit.setValue(AxisSelection.max_values[self.index])
        else:
            self.max_edit.setValue(10)
        # self.max_edit.setValue( 10 )
        self.max_edit.valueChanged.connect(self.max_changed)

        self.grid.addWidget(label1, 0, 0)
        self.grid.addWidget(self.combo_box, 0, 1)
        self.grid.addWidget(label2, 0, 2, Qt.AlignRight)
        self.grid.addWidget(self.max_edit, 0, 3)
        self.grid.setContentsMargins(QMargins(0, 0, 0, 0))

        self.setLayout(self.grid)
        self.setMaximumHeight(40)

        self.show()

    def updateDisplay(self):
        display = self.manager.display_widget.currentWidget()
        display.setMax(self.index, self.max_edit.value())

    def max_changed(self):
        """ This function should only be called from the connected event
            when the user has entered a new value.
            Otherwise use maxChanged() which does not update the graph drawing.
            This is to avoid multiple updates of the graph.
        """
        self.maxChanged()
        display = self.manager.display_widget.currentWidget()
        if isinstance(display, GraphWidget):
            display.update()
        else:
            display.drawGraph()
            display.update()

    def maxChanged(self):
        """ This function updates the max value without redrawing the graph.
            Use this function from external calls to avoid multiple redraws
            of the graph.
        """
        if self.max_edit.value() == 0:
            self.max_edit.setValue(self.max_edit.minimum())
            return
        display = self.manager.display_widget.currentWidget()

        if self.index <= 2:
            AxisSelection.max_values[self.index] = self.max_edit.value()

        display.setMax(self.index, self.max_edit.value())

    def clearAxisLayout(self):
        if self.grid is not None:
            while self.grid.count():
                item = self.grid.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

    def setMax(self, axis, maxValue):
        if axis <= 2:
            self.max_values[axis] = maxValue
            self.max_edit.setValue(maxValue)

    def getParameter(self):
        p = Parameter(self.combo_box.currentText())
        p.id = self.combo_box.currentIndex()
        return p

    def parameter_selected(self):
        new_name = self.getParameter()
        self.manager.parameterSelected(self.index,
                                       new_name,
                                       self.old_name)
        self.old_name = new_name

    def getValue(self):
        return self.max_edit.value()

    def switchParameter(self, newParam):
        i = self.combo_box.findText(newParam, Qt.MatchExactly)
        self.combo_box.setCurrentIndex(i)


#####################################################################
class ValueSelection(QWidget):
    ''' This class represents a Parameter in the data display that
        is not shown of one of the axis. It allows to select a value
        for this parameter.
        It is a helper class for the class DataDisplay.
    '''

    #####################################################################
    default_values = defaultdict(lambda: 10)

    def __init__(self, manager, parent, param_id, parameter_name):
        super(ValueSelection, self).__init__(parent)
        self.manager = manager
        self.parameter = param_id
        self.parameter_name = parameter_name
        self.initUI()
        self.show()

    # noinspection PyAttributeOutsideInit
    def initUI(self):
        self.grid = QGridLayout(self)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(3, 1)

        label0 = QLabel("Parameter:")
        label0.setMinimumWidth(75)
        self.parameter_label = QLabel(self.parameter_name)
        self.parameter_label.setMinimumWidth(100)
        label2 = QLabel("Value:")
        label2.setMinimumWidth(40)
        label2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.value_edit = QDoubleSpinBox()
        self.value_edit.setMinimum(MIN_PARAM_VALUE)
        self.value_edit.setMaximum(MAX_PARAM_VALUE)
        self.value_edit.setValue(float(self.default_values[self.parameter]))
        self.value_edit.setMinimumHeight(25)
        self.value_edit.valueChanged.connect(self._value_changed)

        self.grid.addWidget(label0, 0, 0)
        self.grid.addWidget(self.parameter_label, 0, 1)
        self.grid.addWidget(label2, 0, 2, Qt.AlignRight)
        self.grid.addWidget(self.value_edit, 0, 3)
        self.grid.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setLayout(self.grid)
        self.setMaximumHeight(30)

    def getValue(self):
        return self.value_edit.value()

    def setValue(self, value):
        self.value_edit.setValue(value)

    @Slot(float)
    def _value_changed(self, value):
        if value == 0:
            self.value_edit.setValue(self.value_edit.minimum())
            return
        self.default_values[self.parameter] = value
        display = self.manager.display_widget.currentWidget()
        if isinstance(display, GraphWidget):
            display.update()
        else:
            display.drawGraph()
            display.update()

    def setName(self, parameter):
        self.parameter = parameter.id
        self.parameter_name = parameter.name
        self.parameter_label.setText(self.parameter_name)

    def clearRowLayout(self):
        if self.grid is not None:
            while self.grid.count():
                item = self.grid.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())


#####################################################################
class DataDisplayManager(QWidget):
    """ This class manages the different data views and display
        options.
        To add a new display:
        1. It must be a class derived from QWidget.
        2. It must implement a function getNumAxis() that returns
           the number of free parameters of the display
        3. It must implement a function setMax( axis, value ) where
           axis is the index of the axis that is changed. 0 is the X-axis
           1 is the Y-axis, 2 is the Z-axis. And value is the new max value.
        4. Add it to self.display_widget via addTab()
        5. For evaluation of functions retrieve the value if fixed parameters
           with getValues()
    """

    #####################################################################

    def __init__(self, main_widget, parent):
        super(DataDisplayManager, self).__init__(parent)
        self.main_widget = main_widget
        self.axis_selections = list()
        self.value_selections = list()
        self._experiment = None
        self.parameters = []
        self.limits_widget = None
        self.initUI()

    # noinspection PyAttributeOutsideInit
    def initUI(self):
        grid = QGridLayout(self)

        self.display_widget = QTabWidget(self)
        self.display_widget.setMovable(True)
        self.display_widget.setTabsClosable(True)
        self.display_widget.tabCloseRequested.connect(self.closeTab)
        grid.addWidget(self.display_widget, 0, 0)
        # loading this tab as default view (Line graph)
        self.reloadTabs([0])

        self.display_widget.tabsClosable()
        self.display_widget.currentChanged.connect(self.experimentChange)
        self.show()

    def closeTab(self, currentIndex):
        self.display_widget.removeTab(currentIndex)

    def is_tab_already_opened(self, text):
        tabStatus = False
        tabCount = self.display_widget.count()
        for index in range(0, tabCount):
            if text == self.display_widget.tabText(index):
                tabStatus = True

        return tabStatus

    def reloadTabs(self, selectedCheckBoxesIndex):
        # 0: Line Graph,
        # 1: AllFunctionsAsOneSurfacePlotWidget,
        # 2 :AllFunctionsAsDifferentSurfacePlotWidget
        # 3: DominatingFunctionsAsSingleScatterPlotWidget,
        # 4: MaxZAsSingleSurfacePlotWidget
        # 5: HeatMapGraphWidget,
        # 6: IsolinesDisplayWidget
        # 7: InterpolatedContourDisplayWidget
        # 8: Measurement Points
        if 0 in selectedCheckBoxesIndex:
            labelText = "Line graph"
            tabStatus = self.is_tab_already_opened(labelText)
            if tabStatus is False:
                graph = GraphWidget(self.main_widget, self)
                self.display_widget.addTab(graph, labelText)

        graph_widgets = {
            1: ('Single surface plot', AllFunctionsAsOneSurfacePlot),
            2: ('Surface plots', AllFunctionsAsDifferentSurfacePlot),
            3: ('Domination scatter plot', DominatingFunctionsAsSingleScatterPlot),
            4: ('Max. Z surface plot', MaxZAsSingleSurfacePlot),
            5: ("Heat map", HeatMapGraph),
            6: ("Contour plot", IsolinesDisplay),
            7: ("Interpolated contour", InterpolatedContourDisplay),
            8: ("Measurement points", MeasurementPointsPlot),
            9: ("Line graph plus", LineGraphPlus)
        }

        for i in selectedCheckBoxesIndex:
            if i == 0:
                continue
            labelText, plot = graph_widgets[i]
            if not self.is_tab_already_opened(labelText):
                advance_plot_widget = AdvancedPlotWidget(
                    self.main_widget, self, plot)
                self.display_widget.addTab(
                    advance_plot_widget, labelText)

    def experimentChange(self):
        experiment = self.main_widget.getExperiment()
        if experiment is None:
            return
        if self._experiment is None or self._experiment != experiment:
            self._experiment = experiment
            max_coordinate = max(experiment.coordinates)
            for i, pos in enumerate(max_coordinate):
                if i < 3:
                    AxisSelection.max_values[i] = pos * 1.2
                ValueSelection.default_values[i] = pos

        self.parameters = experiment.parameters
        self.limits_widget.generateSelections(self.parameters)
        self.updateWidget()

    def setMaxValue(self, index, value):
        if index < len(self.axis_selections):
            self.axis_selections[index].max_edit.setValue(value)

    def getValues(self):
        pv_list = {}
        for i in self.value_selections:
            pv_list[i.parameter] = i.getValue()
        return pv_list

    def getAxisParameter(self, index):
        return self.axis_selections[index].getParameter()

    def parameterSelected(self, index, newName, oldName):
        old_value = self.axis_selections[index].getValue()
        for i in self.axis_selections:
            if i.index != index:
                if i.getParameter().name == newName:
                    self.setMaxValue(index, i.getValue())
                    self.setMaxValue(i.index, old_value)
                    i.switchParameter(oldName)
                    i.maxChanged()
                    self.axis_selections[index].maxChanged()
                    self.updateWidget()
                    return
        for i in self.value_selections:
            if i.parameter == newName.id:
                i.setName(oldName)
                self.setMaxValue(index, i.getValue())
                i.setValue(old_value)
                self.axis_selections[index].maxChanged()
                self.updateWidget()
                return

    def updateWidget(self):
        display = self.display_widget.currentWidget()
        if not display:
            return
        if isinstance(display, GraphWidget):
            display.update()
        else:
            display.drawGraph()


class GraphLimitsWidget(QWidget):
    def __init__(self, parent, data_display: DataDisplayManager):
        super().__init__(parent)
        self.data_display = data_display
        self.display_widget = data_display.display_widget
        self.axis_selections = data_display.axis_selections
        self.value_selections = data_display.value_selections
        self.data_display.limits_widget = self

        self.grid = QGridLayout(self)
        self.grid.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)

        self._placeholder = AxisSelection(data_display, self, 0, [])
        self._placeholder.setEnabled(False)
        self.grid.addWidget(self._placeholder)

        self.setLayout(self.grid)

    def generateSelections(self, parameters):

        if not self.display_widget.currentWidget():
            return

        if self._placeholder is not None:
            self._placeholder.hide()
            self._placeholder.deleteLater()
            self._placeholder = None

        num_axis = self.display_widget.currentWidget().getNumAxis()
        for axis in self.axis_selections:
            axis.clearAxisLayout()
        del self.axis_selections[:]

        for v in self.value_selections:
            v.clearRowLayout()
        del self.value_selections[:]

        for i in range(0, num_axis):
            axis_selection = AxisSelection(self.data_display, self, i, parameters)
            self.axis_selections.append(axis_selection)
            self.grid.addWidget(axis_selection, i, 0)
        num_param = len(parameters)
        for i in range(num_axis, num_param):
            value_selection = ValueSelection(self.data_display, self, i,
                                             parameters[i].name)
            self.value_selections.append(value_selection)
            self.grid.addWidget(value_selection, i, 0)
