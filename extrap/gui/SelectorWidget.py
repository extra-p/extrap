# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import math
from typing import Optional, Sequence

import numpy
from PySide2.QtCore import *  # @UnusedWildImport
from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport

from extrap.entities.calltree import Node
from extrap.entities.metric import Metric
from extrap.gui.ParameterValueSlider import ParameterValueSlider
from extrap.gui.TreeModel import TreeModel
from extrap.gui.TreeView import TreeView
from extrap.modelers.model_generator import ModelGenerator


class SelectorWidget(QWidget):
    def __init__(self, mainWidget, parent):
        super(SelectorWidget, self).__init__(parent)
        self.main_widget = mainWidget
        self.tree_model = TreeModel(self)
        self.parameter_sliders = list()
        self.initUI()
        self._sections_switched = False

    # noinspection PyAttributeOutsideInit
    def initUI(self):
        self.grid = QGridLayout(self)
        self.setLayout(self.grid)

        # Model selection
        model_label = QLabel("Model:", self)
        self.model_selector = QComboBox(self)
        self.model_selector.currentIndexChanged.connect(self.model_changed)

        # model_list = list()
        self.updateModelList()

        # Metric selection
        metric_label = QLabel("Metric:", self)
        self.metric_selector = QComboBox(self)
        self.metric_selector.currentIndexChanged.connect(
            self.metric_index_changed)

        # Callpath selection
        self.tree_view = TreeView(self)

        # Input variable values
        self.asymptoticCheckBox = QCheckBox('Show model', self)
        self.asymptoticCheckBox.toggle()
        self.asymptoticCheckBox.stateChanged.connect(
            self.changeAsymptoticBehavior)

        # Positioning
        self.grid.addWidget(model_label, 0, 0)
        self.grid.addWidget(self.model_selector, 0, 1)
        self.grid.addWidget(metric_label, 1, 0)
        self.grid.addWidget(self.metric_selector, 1, 1)
        self.grid.addWidget(self.tree_view, 2, 0, 1, 2)
        self.grid.addWidget(self.asymptoticCheckBox, 3, 1, Qt.AlignRight)
        self.grid.setColumnStretch(1, 1)

    def createParameterSliders(self):
        for param in self.parameter_sliders:
            param.clearRowLayout()
            self.grid.removeWidget(param)
        del self.parameter_sliders[:]
        experiment = self.main_widget.getExperiment()
        parameters = experiment.parameters
        for i, param in enumerate(parameters):
            new_widget = ParameterValueSlider(self, param, self)
            self.parameter_sliders.append(new_widget)
            self.grid.addWidget(new_widget, i + 4, 0, 1, 2)

    def fillCalltree(self):
        self.tree_model = TreeModel(self)
        self.tree_view.setModel(self.tree_model)
        self.tree_view.header().setDefaultSectionSize(65)
        # increase width of "Callpath" and "Value" columns
        self.tree_view.setColumnWidth(0, 150)
        self.tree_view.setColumnWidth(3, 150)
        if not self._sections_switched:
            self.tree_view.header().swapSections(0, 1)
            self._sections_switched = True
        self.tree_view.header().setMinimumSectionSize(23)
        self.tree_view.header().resizeSection(1, 23)
        self.tree_view.header().resizeSection(2, 23)
        selectionModel = self.tree_view.selectionModel()
        selectionModel.selectionChanged.connect(
            self.callpath_selection_changed)

    def callpath_selection_changed(self):
        callpath_list = self.getSelectedCallpath()
        # self.dict_callpath_color = {}
        self.main_widget.populateCallPathColorMap(callpath_list)
        self.main_widget.updateAllWidget()

    def fillMetricList(self):
        self.metric_selector.clear()
        experiment = self.main_widget.getExperiment()
        if experiment is None:
            return
        metrics = experiment.metrics
        for metric in metrics:
            name = metric.name if metric.name != '' else '<default>'
            self.metric_selector.addItem(name, metric)

    def changeAsymptoticBehavior(self):
        self.tree_model.valuesChanged()

    def getSelectedMetric(self) -> Metric:
        return self.metric_selector.currentData()

    def getSelectedCallpath(self) -> Sequence[Node]:
        indexes = self.tree_view.selectedIndexes()
        callpath_list = list()

        for index in indexes:
            # We only care for the first column, otherwise we would get the same callpath repeatedly for each column
            if index.column() != 0:
                continue
            callpath = self.tree_model.getValue(index)
            callpath_list.append(callpath)
        return callpath_list

    def getCurrentModel(self) -> Optional[ModelGenerator]:
        model = self.model_selector.currentData()
        return model

    def renameCurrentModel(self, newName):
        index = self.model_selector.currentIndex()
        self.getCurrentModel().name = newName
        self.model_selector.setItemText(index, newName)

    def getModelIndex(self):
        return self.model_selector.currentIndex()

    def selectLastModel(self):
        self.model_selector.setCurrentIndex(self.model_selector.count() - 1)

    def updateModelList(self):
        experiment = self.main_widget.getExperiment()
        if not experiment:
            return
        models_list = experiment.modelers
        self.model_selector.clear()
        for model in models_list:
            self.model_selector.addItem(model.name, model)
        # self.main_widget.data_display.updateWidget()
        # self.update()

    def model_changed(self):
        # index = self.model_selector.currentIndex()
        # text = str(self.model_selector.currentText())

        # Introduced " and text != "No models to load" " as a second guard since always when the text would be
        # "No models to load" the gui would crash.
        # if model != None and text != "No models to load":
        #     generator = model._modeler
        self.main_widget.selector_widget.tree_model.valuesChanged()

        self.main_widget.updateAllWidget()
        self.update()

    def model_rename(self):
        index = self.getModelIndex()
        if index < 0:
            return
        result = QInputDialog.getText(self,
                                      'Rename Current Model',
                                      'Enter new name', QLineEdit.EchoMode.Normal)
        new_name = result[0]
        if result[1] and new_name:
            self.renameCurrentModel(new_name)

    def model_delete(self):
        reply = QMessageBox.question(self,
                                     'Delete Current Model',
                                     "Are you sure to delete the model?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            index = self.getModelIndex()
            experiment = self.main_widget.getExperiment()
            if index < 0:
                return

            self.model_selector.removeItem(index)
            del experiment.modelers[index]

    @staticmethod
    def get_all_models(experiment):
        if experiment is None:
            return None
        models = experiment.modelers
        if len(models) == 0:
            return None
        return models

    def metric_index_changed(self):
        self.main_widget.metricIndexChanged()
        self.tree_model.on_metric_changed()

    def getParameterValues(self):
        ''' This functions returns the parameter value list with the
            parameter values from the bottom of the calltree selection.
            This information is necessary for the evaluation of the model
            functions, e.g. to colot the severity boxes.
        '''
        value_list = []
        for param in self.parameter_sliders:
            value_list.append(param.getValue())
        return value_list

    def iterate_children(self, paramValueList, callpaths, metric):
        ''' This is a helper function for getMinMaxValue.
            It iterates the calltree recursively.
        '''
        value_list = list()
        for callpath in callpaths:
            model = self.getCurrentModel().models.get((callpath.path, metric))
            if model is None:
                continue

            formula = model.hypothesis.function
            value = formula.evaluate(paramValueList)
            if not math.isinf(value):
                value_list.append(value)
            children = callpath.childs
            value_list += self.iterate_children(paramValueList,
                                                children, metric)
        return value_list

    def getMinMaxValue(self):
        ''' This function calculated the minimum and the maximum values that
            appear in the call tree. This information is e.g. used to scale
            legends ot the color line at the bottom of the extrap window.
        '''
        value_list = list()
        experiment = self.main_widget.getExperiment()
        if experiment is None:
            value_list.append(1)
            return value_list
        selectedMetric = self.getSelectedMetric()
        if selectedMetric is None:
            value_list.append(1)
            return value_list
        param_value_list = self.getParameterValues()
        call_tree = experiment.call_tree
        nodes = call_tree.get_nodes()
        previous = numpy.seterr(divide='ignore', invalid='ignore')
        value_list.extend(self.iterate_children(param_value_list,
                                                nodes,
                                                selectedMetric))
        numpy.seterr(**previous)
        if len(value_list) == 0:
            value_list.append(1)
        return value_list
