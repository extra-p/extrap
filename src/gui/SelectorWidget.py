"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


try:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
except ImportError:
    from PyQt5.QtGui import *  # @UnusedWildImport
    from PyQt5.QtCore import *  # @UnusedWildImport
    from PyQt5.QtWidgets import *  # @UnusedWildImport
from gui.TreeModel import TreeModel
from gui.TreeView import TreeView
from gui.ParameterValueSlider import ParameterValueSlider


class SelectorWidget(QWidget):
    def __init__(self, mainWidget, parent):
        super(SelectorWidget, self).__init__(parent)
        self.main_widget = mainWidget
        self.tree_model = TreeModel(self)
        self.parameter_sliders = list()
        self.initUI()

    def initUI(self):
        self.grid = QGridLayout(self)
        self.setLayout(self.grid)

        # Model selection
        model_label = QLabel("Model :", self)
        self.model_selector = QComboBox(self)
        self.model_selector.currentIndexChanged.connect(self.model_changed)

        #model_list = list()
        self.updateModelList()

        # Metric selection
        metric_label = QLabel("Metric :", self)
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
        self.grid.addWidget(self.model_selector, 0, 1, 1, 10)
        self.grid.addWidget(metric_label, 1, 0)
        self.grid.addWidget(self.metric_selector, 1, 1, 1, 10)
        self.grid.addWidget(self.tree_view, 2, 0, 1, 20)
        self.grid.addWidget(self.asymptoticCheckBox, 3, 15, 1, 5)

    def createParameterSliders(self):
        for param in self.parameter_sliders:
            param.clearRowLayout()
            self.grid.removeWidget(param)
        del self.parameter_sliders[:]
        experiment = self.main_widget.getExperiment()
        parameters = experiment.get_parameters()
        for i in range(len(parameters)):
            new_widget = ParameterValueSlider(self, parameters[i], self)
            self.parameter_sliders.append(new_widget)
            self.grid.addWidget(new_widget, i+4, 0, 1, 20)

    def fillCalltree(self):
        self.tree_model = TreeModel(self)
        self.tree_view.setModel(self.tree_model)
        # increase width of "Callpath" and "Value" columns
        self.tree_view.setColumnWidth(0, 150)
        self.tree_view.setColumnWidth(1, 25)
        self.tree_view.setColumnWidth(2, 25)
        self.tree_view.setColumnWidth(3, 150)
        self.tree_view.header().swapSections(0, 1)
        selectionModel = self.tree_view.selectionModel()
        selectionModel.selectionChanged.connect(
            self.callpath_selection_changed)

    def callpath_selection_changed(self):
        callpath_list = self.getSelectedCallpath()
        #self.dict_callpath_color = {}
        self.main_widget.populateCallPathColorMap(callpath_list)
        self.main_widget.updateAllWidget()

    def fillMetricList(self):
        self.metric_selector.clear()
        experiment = self.main_widget.getExperiment()
        if not experiment == None:
            metrics = experiment.get_metrics()
            for i in range(len(metrics)):
                self.metric_selector.addItem(metrics[i].get_name())

    def changeAsymptoticBehavior(self):
        self.tree_model.valuesChanged()

    def getSelectedMetric(self):
        experiment = self.main_widget.getExperiment()
        if not experiment == None:
            index = self.metric_selector.currentIndex()
            if index >= 0:
                return experiment.get_metrics()[index]
        return None

    def getSelectedCallpath(self):
        indexes = self.tree_view.selectedIndexes()
        callpath_list = list()

        for index in indexes:
            # We only care for the first column, otherwise we would get the same callpath repeatedly for each column
            if index.column() != 0:
                continue
            callpath = self.tree_model.getValue(index)
            callpath_list.append(callpath)
        return callpath_list

    def getCurrentModel(self):
        experiment = self.main_widget.getExperiment()
        if experiment == None:
            return None
        models = experiment.get_models()
        if len(models) == 0:
            return None
        index = self.getModelIndex()
        if index < 0 or index >= len(models):
            return None
        return models[index]

    def renameCurrentModel(self, newName):
        experiment = self.main_widget.getExperiment()
        metric = experiment.getMetrics()[0]
        callpath = experiment.getRootCallpaths()[0]
        index = self.getModelIndex()
        model = experiment.getModels(metric, callpath)[index]
        generator = model.getGenerator()
        generator.setUserName(newName)
        self.model_selector.setItemText(index, newName)

    def getModelIndex(self):
        return self.model_selector.currentIndex()

    def selectLastModel(self):
        self.model_selector.setCurrentIndex(self.model_selector.count() - 1)

    def updateModelList(self):
        models_list = self.get_model_list()
        self.model_selector.clear()
        for model, model_info in models_list:
            self.model_selector.addItem(model, model_info)
        # self.main_widget.data_display.updateWidget()
        # self.update()

    def get_model_list(self):
        model_list = list()
        model_additional_info_list = list()
        #all_models = list()

        experiment = self.main_widget.getExperiment()
        if experiment == None:
            model_list.append('No models to load')
            model_additional_info_list.append("")
            model_info_list = zip(model_list, model_additional_info_list)
            return model_info_list

        metric = experiment.get_metrics()[0]
        call_tree = experiment.get_call_tree()
        nodes = call_tree.get_nodes()
        callpath = nodes[0]
        all_models = self.get_all_models(experiment)
        if all_models != None:
            for model in all_models:
                #TODO: this name should be pulled from the ui field
                modeler_name = "New Model"
                model_list.append(modeler_name)
                #model_list.append(model.getGenerator().getUserName())
                model_additional_info_list.append(model)

        if len(model_list) == 0:
            model_list.append('No models to load')
            model_additional_info_list.append("")

        model_info_list = zip(model_list, model_additional_info_list)
        return model_info_list

    def model_changed(self):
        index = self.model_selector.currentIndex()
        text = str(self.model_selector.currentText())
        model = self.model_selector.itemData(index)

        #TODO: fix this
        # Introduced " and text != "No models to load" " as a second guard since always when the text would be "No models to load" the gui would crash.
        if model != None and text != "No models to load":
            generator = model.getGenerator()
        self.main_widget.updateAllWidget()
        self.update()

    def model_rename(self):
        index = self.getModelIndex()
        if index < 0:
            return
        result = QInputDialog.getText(self,
                                      self.tr('Rename current model'),
                                      self.tr('Enter new name'))
        new_name = result[0]
        if result[1] and new_name:
            self.renameCurrentModel(new_name)

    def model_delete(self):
        reply = QMessageBox.question(self,
                                     self.tr('Quit'),
                                     self.tr(
                                         "Are you sure to delete the model?"),
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            index = self.getModelIndex()
            experiment = self.main_widget.getExperiment()
            if index < 0:
                return

            self.model_selector.removeItem(index)
            experiment.deleteModel(index)

    def get_all_models(self, experiment):
        if experiment == None:
            return None
        models = experiment.get_models()
        if len(models) == 0:
            return None
        return models

    def metric_index_changed(self):
        self.main_widget.metricIndexChanged()
        
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
        for i in range(0, len(callpaths)):
            model = self.getCurrentModel()
            if model == None:
                return value_list
            
            
            formula = model.getModelFunction()
            value = formula.evaluate(paramValueList)
            value_list.append(value)
            children = callpaths[i].getChildren()
            value_list.extend(self.iterate_children(paramValueList,
                                                    children, metric))
        return value_list
    def getMinMaxValue(self):
        ''' This function calculated the minimum and the maximum values that
            appear in the call tree. This information is e.g. used to scale
            legends ot the color line at the bottom of the main window.
        '''
        value_list = list()
        experiment = self.main_widget.getExperiment()
        if experiment == None:
            value_list.append(1)
            return value_list
        selectedMetric = self.getSelectedMetric()
        if selectedMetric == None:
            value_list.append(1)
            return value_list
        param_value_list = self.getParameterValues()
        call_tree = experiment.get_call_tree()
        nodes = call_tree.get_nodes()

        value_list.extend(self.iterate_children(param_value_list,
                                                nodes,
                                                selectedMetric))

        if len(value_list) == 0:
            value_list.append(1)
        return value_list
