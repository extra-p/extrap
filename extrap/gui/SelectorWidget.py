# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import math
from typing import Optional, Sequence, TYPE_CHECKING, Tuple

import numpy
from PySide6.QtWidgets import *  # @UnusedWildImport

from extrap.entities.calltree import Node
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.gui.TreeModel import TreeModel, TreeItemFilterProvider
from extrap.gui.TreeView import TreeView
from extrap.gui.components.ParameterValueSlider import ParameterValueSlider
from extrap.modelers.model_generator import ModelGenerator

if TYPE_CHECKING:
    from extrap.gui.MainWidget import MainWidget


class SelectorWidget(QWidget):
    def __init__(self, main_widget: MainWidget, parent):
        super(SelectorWidget, self).__init__(parent)
        self.main_widget = main_widget
        self.tree_model = TreeModel(self)
        self.parameter_sliders = list()
        self.initUI()
        self._sections_switched = False
        self.min_value = 0
        self.max_value = 0

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

        group = QFrame(self)
        group.setObjectName("TreeViewContainer")
        group.setStyleSheet("QFrame#TreeViewContainer{border:1px solid #939393}")
        group_layout = QVBoxLayout(group)
        group.setLayout(group_layout)
        group.setContentsMargins(0, 0, 0, 0)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(0)

        # group.setAutoFillBackground(True)
        # Callpath selection
        self.tree_view = TreeView(group)
        group_layout.addWidget(self.tree_view)
        # Input variable values

        self.toolbar = QToolBar(group)
        self.toolbar.layout().setContentsMargins(2, 3, 2, 2)
        self.toolbar.layout().setSpacing(5)
        group_layout.addWidget(self.toolbar)

        self.tree_display_select = QComboBox(self.toolbar)
        self.tree_display_select.addItem('All', TreeItemFilterProvider.DisplayType.INCLUDE)
        self.tree_display_select.addItem('Compact', TreeItemFilterProvider.DisplayType.COMPACT)
        self.tree_display_select.addItem('Flat', TreeItemFilterProvider.DisplayType.FLAT)

        def select_view_type(x):
            self.tree_model.item_filter.display_type = self.tree_display_select.currentData()

        self.tree_display_select.currentIndexChanged.connect(select_view_type)
        self.toolbar.addWidget(self.tree_display_select)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar.addWidget(spacer)

        self.asymptoticCheckBox = QCheckBox('Show model', self.toolbar)
        self.asymptoticCheckBox.toggle()
        self.asymptoticCheckBox.stateChanged.connect(
            self.changeAsymptoticBehavior)
        self.toolbar.addWidget(self.asymptoticCheckBox)

        self.show_parameters = QCheckBox('Show parameters', self.toolbar)
        # self.show_parameters.toggle()
        self.show_parameters.stateChanged.connect(
            self.changeAsymptoticBehavior)
        self.toolbar.addWidget(self.show_parameters)

        # Positioning
        self.grid.addWidget(model_label, 0, 0)
        self.grid.addWidget(self.model_selector, 0, 1)
        self.grid.addWidget(metric_label, 1, 0)
        self.grid.addWidget(self.metric_selector, 1, 1)
        self.grid.addWidget(group, 2, 0, 1, 2)
        # self.grid.addWidget(self.toolbar, 3, 0, 1, 2)

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
        call_tree_nodes = self.get_selected_call_tree_nodes()
        metric = self.getSelectedMetric()
        call_tree_nodes = [c for c in call_tree_nodes if
                           (c.path, metric) in self.getCurrentModel().models]
        self.main_widget.model_color_map.update(call_tree_nodes)
        self.main_widget.on_selection_changed()

    def fillMetricList(self):
        self.metric_selector.clear()
        experiment = self.main_widget.getExperiment()
        if experiment is None:
            return
        metrics = experiment.metrics
        for metric in metrics:
            name = metric.name if metric.name != '' else '<default>'
            self.metric_selector.addItem(name, metric)

    def on_experiment_changed(self):
        self.updateModelList()
        self.fillMetricList()
        self.createParameterSliders()
        self.fillCalltree()
        self.tree_model.valuesChanged()

    def changeAsymptoticBehavior(self):
        self.tree_model.valuesChanged()

    def getSelectedMetric(self) -> Metric:
        return self.metric_selector.currentData()

    def get_selected_call_tree_nodes(self) -> Sequence[Node]:
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

    def get_selected_models(self) -> Tuple[Optional[Sequence[Model]], Optional[Sequence[Node]]]:
        selected_metric = self.getSelectedMetric()
        selected_call_tree_nodes = self.get_selected_call_tree_nodes()
        model_set = self.getCurrentModel()
        if not selected_call_tree_nodes or model_set is None:
            return None, None
        model_set_models = model_set.models
        if not model_set_models:
            return None, None
        model_list = list()
        for node in selected_call_tree_nodes:
            key = (node.path, selected_metric)
            if key in model_set_models:
                model = model_set_models[key]
                model_list.append(model)
        return model_list, selected_call_tree_nodes

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

        self.main_widget.on_selection_changed()
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
                                     "Are you sure to delete the current model?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            index = self.getModelIndex()
            experiment = self.main_widget.getExperiment()
            if index < 0:
                return

            self.model_selector.removeItem(index)
            del experiment.modelers[index]

    def delete_metric(self):
        reply = QMessageBox.question(self,
                                     'Delete Current Metric',
                                     "Are you sure to delete the current metric?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            index = self.metric_selector.currentIndex()
            metric = self.getSelectedMetric()
            if index < 0:
                return
            experiment = self.main_widget.getExperiment()
            self.metric_selector.removeItem(index)
            experiment.metrics.remove(metric)
            for callpath in experiment.callpaths:
                key = (callpath, metric)
                if key in experiment.measurements:
                    del experiment.measurements[key]
            for model_set in experiment.modelers:
                for callpath in experiment.callpaths:
                    key = (callpath, metric)
                    if key in model_set.models:
                        del model_set.models[key]

    @staticmethod
    def get_all_models(experiment):
        if experiment is None:
            return None
        models = experiment.modelers
        if len(models) == 0:
            return None
        return models

    def metric_index_changed(self):
        self.main_widget.on_selection_changed()
        self.tree_model.on_metric_changed()

    def getParameterValues(self):
        """ This functions returns the parameter value list with the
            parameter values from the bottom of the calltree selection.
            This information is necessary for the evaluation of the model
            functions, e.g. to color the severity boxes.
        """
        value_list = []
        for param in self.parameter_sliders:
            value_list.append(param.getValue())
        return value_list

    def iterate_children(self, models, param_value_list, callpaths, metric):
        """ This is a helper function for update_min_max_value.
            It iterates the calltree recursively.
        """
        value_list = list()
        for callpath in callpaths:
            model = models.get((callpath.path, metric))
            if model is not None:
                if isinstance(model, list):
                    pass
                    #TODO: need code here
                else:
                    formula = model.hypothesis.function
                    value = formula.evaluate(param_value_list)
                    if not math.isinf(value) and not math.isnan(value):
                        value_list.append(value)
            children = callpath.childs
            value_list += self.iterate_children(models, param_value_list, children, metric)
        return value_list

    def update_min_max_value(self):
        """ This function calculated the minimum and the maximum values that
            appear in the call tree. This information is e.g. used to scale
            legends ot the color line at the bottom of the extrap window.
        """
        min_max_value = (0, 0)
        experiment = self.main_widget.getExperiment()
        if experiment:
            selected_metric = self.getSelectedMetric()
            model_set = self.getCurrentModel()
            if selected_metric and model_set:
                param_value_list = self.getParameterValues()
                call_tree = experiment.call_tree
                nodes = call_tree.get_nodes()
                previous = numpy.seterr(divide='ignore', invalid='ignore')
                value_list = self.iterate_children(model_set.models, param_value_list, nodes, selected_metric)
                numpy.seterr(**previous)
                if len(value_list) > 0:
                    min_max_value = max(0.0, min(value_list)), max(0.0, max(value_list))
        self.min_value, self.max_value = min_max_value
        return min_max_value
