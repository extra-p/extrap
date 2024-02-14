# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import numpy as np
from extrap.entities.callpath import Callpath
from extrap.entities.metric import Metric
import sys
from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import *  # @UnusedWildImport
from PySide6.QtGui import QDoubleValidator, QIntValidator
import matplotlib

from extrap.gpr.measurement_point_advisor import MeasurementPointAdvisor


class MeasurementWizardWidget(QWidget):
    
    def __init__(self, mainWidget, parent):
        super(MeasurementWizardWidget, self).__init__(parent)
        self.main_widget = mainWidget
        self.main_layout = QHBoxLayout(self)
        self.init_empty()
        self.calculate_cost_manual = False
        self.experiment = None
        self.mean_noise_level = 0.0
        self.initialized = False
        self.processes_parameter_id = 0
        self.no_model_parameters = 0
        self.budget = 0.0
        self.current_cost = 0.0
        self.manual_pms_selection = False
        self.parameter_value_series = []


    def init_empty(self):
        self.setLayout(self.main_layout)
        self.empty_label = QLabel(self)
        self.empty_label.setText("Please load a performance experiment.")
        self.main_layout.addWidget(self.empty_label)

    
    def init_multi_parameter(self):
        self.initialized = True

        self.no_model_parameters = len(self.experiment.parameters)
    
        self.main_layout.removeWidget(self.empty_label)
        if self.empty_label is not None:
            self.empty_label.deleteLater()
            self.empty_label = None

        self.layout = QGridLayout(self)
        self.layout.setRowStretch(99, 1)
        self.layout.setColumnStretch(0, 0)
        self.layout.setColumnStretch(1, 1)

        self.main_layout.addLayout(self.layout)

        self.checkbox = QCheckBox("No. of processes (MPI ranks) is one of the model parameters", self)
        self.checkbox.setChecked(True)
        self.checkbox.toggled.connect(self.clickCheckbox)
        self.layout.addWidget(self.checkbox, 0, 0)

        self.parameter_selector_label = QLabel(self)
        self.parameter_selector_label.setText("Model Paramater resembling the number of processes (MPI ranks):")
        self.layout.addWidget(self.parameter_selector_label, 1, 0)

        self.parameter_selector = QComboBox(self)
        for parameter in self.experiment.parameters:
            self.parameter_selector.addItem(str(parameter))
        self.layout.addWidget(self.parameter_selector, 1, 1)
        for i in range(len(self.experiment.parameters)):
            if self.parameter_selector.currentText() == str(self.experiment.parameters[i]):
                self.processes_parameter_id = i
                break
        self.parameter_selector.currentIndexChanged.connect(self.parameterChanged)

        self.process_label = QLabel(self)
        self.process_label.setText("No. of processes (MPI ranks):")
        self.layout.addWidget(self.process_label, 2, 0)
        self.process_label.setVisible(False)

        self.processes = QLineEdit(self)
        self.processes.setValidator(QIntValidator())
        self.layout.addWidget(self.processes, 2, 1)
        self.processes.setVisible(False)
        self.processes.setText("2")
        self.processes.textChanged.connect(self.processesChanged)

        metric_label = QLabel(self)
        metric_label.setText("Metric used for cost calculation:")
        self.layout.addWidget(metric_label, 3, 0)

        self.metric_selector = QComboBox(self)
        for metric in self.experiment.metrics:
            self.metric_selector.addItem(str(metric))
        self.layout.addWidget(self.metric_selector, 3, 1)
        self.metric_selector.currentIndexChanged.connect(self.metricChanged)

        budget_label = QLabel(self)
        budget_label.setText("Modeling Budget [core hours]:")
        self.layout.addWidget(budget_label, 4, 0)

        #TODO: enter budget does not work correctly, conversion when using . is very strange!
        self.modeling_budget = QLineEdit(self)
        self.modeling_budget.setValidator(QDoubleValidator())
        self.layout.addWidget(self.modeling_budget, 4, 1)
        self.modeling_budget.textChanged.connect(self.budgetChanged)

        current_cost_label = QLabel(self)
        current_cost_label.setText("Current Measurement Cost [core hours]:")
        self.layout.addWidget(current_cost_label, 5, 0)

        self.current_cost_edit = QLineEdit(self)
        self.current_cost_edit.setEnabled(False)
        self.current_cost_edit.setText("0.0")
        self.layout.addWidget(self.current_cost_edit, 5, 1)

        used_budget_label = QLabel(self)
        used_budget_label.setText("Used Modeling Budget [%]:")
        self.layout.addWidget(used_budget_label, 6, 0)

        self.used_budget_edit = QLineEdit(self)
        self.used_budget_edit.setEnabled(False)
        self.used_budget_edit.setText("0.0")
        self.layout.addWidget(self.used_budget_edit, 6, 1)

        noise_level_label = QLabel(self)
        noise_level_label.setText("Measurement Noise Level [%]:")
        self.layout.addWidget(noise_level_label, 7, 0)

        self.noise_level_edit = QLineEdit(self)
        self.noise_level_edit.setEnabled(False)
        noise_level_str = "{:.2f}".format(self.mean_noise_level)
        self.noise_level_edit.setText(noise_level_str)
        self.layout.addWidget(self.noise_level_edit, 7, 1)

        self.parameter_value_checkbox = QCheckBox("Enter parameter-value series manually", self)
        self.parameter_value_checkbox.setChecked(False)
        self.parameter_value_checkbox.toggled.connect(self.clickParameterValueCheckbox)
        self.layout.addWidget(self.parameter_value_checkbox, 8, 0)

        self.param_value_layout = QGridLayout(self)
        self.param_value_layout.setRowStretch(99, 1)
        self.param_value_layout.setColumnStretch(0, 0)
        self.param_value_layout.setColumnStretch(1, 1)
        self.layout.addLayout(self.param_value_layout, 9, 0)

        self.advise_button = QPushButton(self)
        self.layout.addWidget(self.advise_button, 10, 0)
        self.advise_button.setText("Suggest Additional Measurement Points")
        self.advise_button.clicked.connect(self.advice_button_clicked)

        self.measurement_point_suggestions_label = QPlainTextEdit(self)
        self.measurement_point_suggestions_label.setPlainText("")
        self.measurement_point_suggestions_label.setEnabled(True)
        self.layout.addWidget(self.measurement_point_suggestions_label, 11, 0)

        self.calculate_current_measurement_cost()
        self.calculate_noise_level()


    def init_single_parameter(self):
        self.initialized = True
        self.no_model_parameters = 1
        self.main_layout.removeWidget(self.empty_label)
        if self.empty_label is not None:
            self.empty_label.deleteLater()
            self.empty_label = None

        self.layout = QGridLayout(self)
        self.layout.setRowStretch(99, 1)
        self.layout.setColumnStretch(0, 0)
        self.layout.setColumnStretch(1, 1)

        self.main_layout.addLayout(self.layout)

        self.checkbox = QCheckBox("Model parameters is the no. of processes (MPI ranks)", self)
        self.checkbox.setChecked(True)
        self.checkbox.toggled.connect(self.clickCheckbox)
        self.layout.addWidget(self.checkbox, 0, 0)
        self.processes_parameter_id = 0

        self.process_label = QLabel(self)
        self.process_label.setText("No. of processes (MPI ranks):")
        self.layout.addWidget(self.process_label, 1, 0)
        self.process_label.setVisible(False)

        self.processes = QLineEdit(self)
        self.processes.setValidator(QIntValidator())
        self.layout.addWidget(self.processes, 1, 1)
        self.processes.setVisible(False)
        self.processes.setText("2")
        self.processes.textChanged.connect(self.processesChanged)

        metric_label = QLabel(self)
        metric_label.setText("Metric used for cost calculation:")
        self.layout.addWidget(metric_label, 2, 0)

        self.metric_selector = QComboBox(self)
        for metric in self.experiment.metrics:
            self.metric_selector.addItem(str(metric))
        self.layout.addWidget(self.metric_selector, 2, 1)
        self.metric_selector.currentIndexChanged.connect(self.metricChanged)

        budget_label = QLabel(self)
        budget_label.setText("Modeling Budget [core hours]:")
        self.layout.addWidget(budget_label, 3, 0)

        self.modeling_budget = QLineEdit(self)
        self.modeling_budget.setValidator(QDoubleValidator())
        self.layout.addWidget(self.modeling_budget, 3, 1)
        self.modeling_budget.textChanged.connect(self.budgetChanged)

        current_cost_label = QLabel(self)
        current_cost_label.setText("Current Measurement Cost [core hours]:")
        self.layout.addWidget(current_cost_label, 4, 0)

        self.current_cost_edit = QLineEdit(self)
        self.current_cost_edit.setEnabled(False)
        self.current_cost_edit.setText("0.0")
        self.layout.addWidget(self.current_cost_edit, 4, 1)

        used_budget_label = QLabel(self)
        used_budget_label.setText("Used Modeling Budget [%]:")
        self.layout.addWidget(used_budget_label, 5, 0)

        self.used_budget_edit = QLineEdit(self)
        self.used_budget_edit.setEnabled(False)
        self.used_budget_edit.setText("0.0")
        self.layout.addWidget(self.used_budget_edit, 5, 1)

        noise_level_label = QLabel(self)
        noise_level_label.setText("Measurement Noise Level [%]:")
        self.layout.addWidget(noise_level_label, 6, 0)

        self.noise_level_edit = QLineEdit(self)
        self.noise_level_edit.setEnabled(False)
        noise_level_str = "{:.2f}".format(self.mean_noise_level)
        self.noise_level_edit.setText(noise_level_str)
        self.layout.addWidget(self.noise_level_edit, 6, 1)

        self.parameter_value_checkbox = QCheckBox("Enter parameter-value series manually", self)
        self.parameter_value_checkbox.setChecked(False)
        self.parameter_value_checkbox.toggled.connect(self.clickParameterValueCheckbox)
        self.layout.addWidget(self.parameter_value_checkbox, 7, 0)

        self.param_value_layout = QGridLayout(self)
        self.param_value_layout.setRowStretch(99, 1)
        self.param_value_layout.setColumnStretch(0, 0)
        self.param_value_layout.setColumnStretch(1, 1)
        self.layout.addLayout(self.param_value_layout, 8, 0)

        self.advise_button= QPushButton(self)
        self.advise_button.setText("Suggest Additional Measurement Points")
        self.layout.addWidget(self.advise_button, 9, 0)
        self.advise_button.clicked.connect(self.advice_button_clicked)

        self.measurement_point_suggestions_label = QPlainTextEdit(self)
        self.measurement_point_suggestions_label.setPlainText("")
        self.measurement_point_suggestions_label.setEnabled(True)
        self.layout.addWidget(self.measurement_point_suggestions_label, 10, 0)

        self.calculate_current_measurement_cost()
        self.calculate_noise_level()


    def calculate_noise_level(self):
        # do an noise analysis on the existing measurement points
        mm = self.experiment.measurements
        metric_string = self.metric_selector.currentText()
        metrics = self.experiment.metrics
        runtime_metric = None
        for metric in metrics:
            if metric_string == str(metric):
                runtime_metric = metric
        selected_callpath = self.main_widget.get_selected_call_tree_nodes()

        # if there is one callpath selected in the tree
        if len(selected_callpath) == 1:
            try:
                nns = []
                try:
                    for meas in mm[(selected_callpath[0].path, runtime_metric)]:
                        pps = []
                        for val in meas.values:
                            if np.mean(meas.values) == 0.0:
                                pp = 0
                            else:
                                pp = abs((val / (np.mean(meas.values) / 100)) - 100)
                            pps.append(pp)
                        nns.append(np.mean(pps))
                except KeyError:
                    nns = [0.0]
                self.mean_noise_level = np.mean(nns)
            except TypeError:
                self.mean_noise_level = 0.0

        # if there is more than one callpath selected in the tree
        elif len(selected_callpath) > 1:
            measurements = self.experiment.measurements
            try:
                callpath_noise_levels = []
                for callpath in selected_callpath:
                    nns = []
                    try:
                        for meas in measurements[(callpath.path, runtime_metric)]:
                            pps = []
                            for val in meas.values:
                                if np.mean(meas.values) == 0.0:
                                    pp = 0
                                else:
                                    pp = abs((val / (np.mean(meas.values) / 100)) - 100)
                                pps.append(pp)
                            nns.append(np.mean(pps))
                    except KeyError:
                        nns = [0.0]
                    callpath_noise_levels.append(np.mean(nns))
                self.mean_noise_level = np.mean(callpath_noise_levels)
            except TypeError:
                self.mean_noise_level = 0.0
            
        # if there is no callpath selected in the tree
        elif len(selected_callpath) == 0:
            measurements = self.experiment.measurements
            try:
                callpath_noise_levels = []
                for callpath in self.experiment.callpaths:
                    nns = []
                    try:
                        for meas in measurements[(callpath, runtime_metric)]:
                            pps = []
                            for val in meas.values:
                                if np.mean(meas.values) == 0.0:
                                    pp = 0
                                else:
                                    pp = abs((val / (np.mean(meas.values) / 100)) - 100)
                                pps.append(pp)
                            nns.append(np.mean(pps))
                    except KeyError:
                        nns = [0.0]
                    callpath_noise_levels.append(np.mean(nns))
                self.mean_noise_level = np.mean(callpath_noise_levels)
            except TypeError:
                self.mean_noise_level = 0.0
        noise_level_str = "{:.2f}".format(self.mean_noise_level)
        self.noise_level_edit.setText(noise_level_str)


    def calculate_used_budget(self):
        budget_text = self.modeling_budget.text()
        budget_text = budget_text.replace(",",".")
        try:
            self.budget = float(budget_text)
        except ValueError:
            self.budget = self.current_cost
        used_budget_percent = self.current_cost / (self.budget / 100)
        used_budget_percent_str = "{:.2f}".format(used_budget_percent)
        self.used_budget_edit.setText(str(used_budget_percent_str))


    def calculate_current_measurement_cost(self):
        current_cost_str = "0.0"
        metric_string = self.metric_selector.currentText()
        metrics = self.experiment.metrics
        runtime_metric = None
        for metric in metrics:
            if metric_string == str(metric):
                runtime_metric = metric
        selected_callpath = self.main_widget.get_selected_call_tree_nodes()
        
        # if the model parameter is not the number of mpi ranks or processes
        if self.calculate_cost_manual:
            try:
                number_processes = int(self.processes.text())
            except ValueError:
                number_processes = 1.0

            # if there is only one callpath selected
            if len(selected_callpath) == 1:
                measurements = self.experiment.measurements
                core_hours_total = 0
                try:
                    for measurement in measurements[(selected_callpath[0].path, runtime_metric)]:
                        core_hours_per_point = 0
                        try:
                            for value in measurement.values:
                                core_hours_per_point += value * number_processes
                        except TypeError:
                            core_hours_per_point += measurement.mean * number_processes
                        core_hours_total += core_hours_per_point
                except KeyError:
                    pass

                self.current_cost = core_hours_total
                current_cost_str = "{:.2f}".format(self.current_cost)

            # if there is no callpath selected
            elif len(selected_callpath) == 0:
                measurements = self.experiment.measurements
                core_hours_total = 0
                for callpath in self.experiment.callpaths:
                    core_hours_callpath = 0
                    try:
                        for measurement in measurements[(callpath, runtime_metric)]:
                            core_hours_per_point = 0
                            try:
                                for value in measurement.values:
                                    core_hours_per_point += value * number_processes
                            except TypeError:
                                core_hours_per_point += measurement.mean * number_processes
                            core_hours_callpath += core_hours_per_point
                    except KeyError:
                        pass
                    core_hours_total += core_hours_callpath
    
                self.current_cost = core_hours_total
                current_cost_str = "{:.2f}".format(self.current_cost)
            
            # if there are several callpaths selected
            elif len(selected_callpath) > 1:
                measurements = self.experiment.measurements
                core_hours_total = 0
                for callpath in selected_callpath:
                    core_hours_callpath = 0
                    try:
                        for measurement in measurements[(callpath.path, runtime_metric)]:
                            core_hours_per_point = 0
                            try:
                                for value in measurement.values:
                                    core_hours_per_point += value * number_processes
                            except TypeError:
                                core_hours_per_point += measurement.mean * number_processes
                            core_hours_callpath += core_hours_per_point
                    except KeyError:
                        pass
                    core_hours_total += core_hours_callpath
    
                self.current_cost = core_hours_total
                current_cost_str = "{:.2f}".format(self.current_cost)

        # if the model parameter is the number of processes or mpi ranks
        else:

            # if there is only one callpath selected
            if len(selected_callpath) == 1:
                measurements = self.experiment.measurements
                core_hours_total = 0
                try:
                    for measurement in measurements[(selected_callpath[0].path, runtime_metric)]:
                        core_hours_per_point = 0
                        parameter_values = measurement.coordinate.as_tuple()
                        try:
                            for value in measurement.values:
                                if self.no_model_parameters == 1:
                                    core_hours_per_point += value * parameter_values[0]
                                else:
                                    core_hours_per_point += value * parameter_values[self.processes_parameter_id]
                        except TypeError:
                            if self.no_model_parameters == 1:
                                core_hours_per_point += measurement.mean * parameter_values[0]
                            else:
                                core_hours_per_point += measurement.mean * parameter_values[self.processes_parameter_id]
                        core_hours_total += core_hours_per_point
                except KeyError:
                    pass
                self.current_cost = core_hours_total
                current_cost_str = "{:.2f}".format(self.current_cost)

            # if there is no callpath selected
            elif len(selected_callpath) == 0:
                measurements = self.experiment.measurements
                core_hours_total = 0
                for callpath in self.experiment.callpaths:
                    core_hours_callpath = 0
                    try:
                        for measurement in measurements[(callpath, runtime_metric)]:
                            core_hours_per_point = 0
                            parameter_values = measurement.coordinate.as_tuple()
                            try:
                                for value in measurement.values:
                                    if self.no_model_parameters == 1:
                                        core_hours_per_point += value * parameter_values[0]
                                    else:
                                        core_hours_per_point += value * parameter_values[self.processes_parameter_id]
                            except TypeError:
                                if self.no_model_parameters == 1:
                                    core_hours_per_point += measurement.mean * parameter_values[0]
                                else:
                                    core_hours_per_point += measurement.mean * parameter_values[self.processes_parameter_id]
                            core_hours_callpath += core_hours_per_point
                    except KeyError:
                        pass
                    core_hours_total += core_hours_callpath
    
                self.current_cost = core_hours_total
                current_cost_str = "{:.2f}".format(self.current_cost)
            
            # if there are several callpaths selected
            elif len(selected_callpath) > 1:         
                measurements = self.experiment.measurements
                core_hours_total = 0
                for callpath in selected_callpath:
                    core_hours_callpath = 0
                    try:
                        for measurement in measurements[(callpath.path, runtime_metric)]:
                            core_hours_per_point = 0
                            parameter_values = measurement.coordinate.as_tuple()
                            try:
                                for value in measurement.values:
                                    if self.no_model_parameters == 1:
                                        core_hours_per_point += value * parameter_values[0]
                                    else:
                                        core_hours_per_point += value * parameter_values[self.processes_parameter_id]
                            except TypeError:
                                if self.no_model_parameters == 1:
                                    core_hours_per_point += measurement.mean * parameter_values[0]
                                else:
                                    core_hours_per_point += measurement.mean * parameter_values[self.processes_parameter_id]
                            core_hours_callpath += core_hours_per_point
                    except KeyError:
                        pass
                    core_hours_total += core_hours_callpath
    
                self.current_cost = core_hours_total
                current_cost_str = "{:.2f}".format(self.current_cost)

        if self.budget == 0.0:
            self.budget = self.current_cost
            self.modeling_budget.setText(str(current_cost_str))

        self.current_cost_edit.setText(str(current_cost_str))


    def advice_button_clicked(self):

        # get the modeling budget from the GUI
        if self.budget <= self.current_cost:
            self.measurement_point_suggestions_label.setPlainText("Not enough budget available to suggest further measurement points!")
        
        else:
            # get the performance metric that should be used for the analysis
            metric_string = self.metric_selector.currentText()
            metrics = self.experiment.metrics
            runtime_metric = None
            for metric in metrics:
                if metric_string == str(metric):
                    runtime_metric = metric
                
            # get the selected callpath(s) in the tree
            selected_callpath = self.main_widget.get_selected_call_tree_nodes()

            # get parameter value series if manual selection
            if self.manual_pms_selection:
                for i in range(len(self.experiment.parameters)):
                    self.parameter_value_series.append(self.line_edits[i].text())

            if self.calculate_cost_manual:
                try:
                    number_processes = int(self.processes.text())
                except ValueError:
                    number_processes = 1
            else:
                number_processes = 0
            
            # initialize a new measurement point advisor object
            mpa = MeasurementPointAdvisor(budget=self.budget, 
                                        processes=self.processes_parameter_id, 
                                        callpaths=selected_callpath,
                                        metric=runtime_metric,
                                        experiment=self.experiment,
                                        current_cost=self.current_cost,
                                        manual_pms_selection=self.manual_pms_selection,
                                        manual_parameter_value_series=self.parameter_value_series,
                                        calculate_cost_manual=self.calculate_cost_manual,
                                        number_processes=number_processes)
            
            point_suggestions = mpa.suggest_points()

            if len(point_suggestions) == 0:
                text = "There was an error during the point suggestion process."

            else:
                #TODO: how to deal with repetitions:
                # only take a look at repetitions when in GPR mode and self.values is available in measurements!
                #print("DEBUG point_suggestions:",point_suggestions)

                """point_suggestions = []
                # (point_parameter_values, repetition no.)
                point_suggestions.append(([32.0, 10.0], 1))
                point_suggestions.append(([128.0, 50.0], 3))"""

                text = ""
                for i in range(len(point_suggestions)):
                    temp = ""
                    temp += str(i+1) + "."
                    temp += " P("
                    parameter_values = point_suggestions[i].as_tuple()
                    for j in range(len(self.experiment.parameters)):
                        if j != 0:
                            temp += ","
                        temp += str(self.experiment.parameters[j])
                        temp += "="
                        parameter_value = parameter_values[j]
                        temp += str(parameter_value)
                    temp += "), repetition no.=1"
                    #temp += str(point_suggestions[i][1])
                    temp += "\n"
                    text += temp

            self.measurement_point_suggestions_label.setPlainText(text)


    def budgetChanged(self):
        self.calculate_used_budget()


    def parameterChanged(self):
        for i in range(len(self.experiment.parameters)):
            if str(self.experiment.parameters[i]) == self.parameter_selector.currentText():
                self.processes_parameter_id = i
                break
        self.calculate_current_measurement_cost()
        self.calculate_used_budget()


    def metricChanged(self):
        self.calculate_current_measurement_cost()
        self.calculate_used_budget()


    def processesChanged(self):
        self.calculate_current_measurement_cost()
        self.calculate_used_budget()


    def clickCheckbox(self):
        cbutton = self.sender()
        if cbutton.isChecked():
            self.process_label.setVisible(False)
            self.calculate_cost_manual = False
            for i in range(len(self.experiment.parameters)):
                if str(self.experiment.parameters[i]) == self.parameter_selector.currentText():
                    self.processes_parameter_id = i
                    break
            self.processes.setVisible(False)
            self.parameter_selector_label.setVisible(True)
            self.parameter_selector.setVisible(True)
            self.calculate_current_measurement_cost()
            self.calculate_used_budget()
        else:
            self.process_label.setVisible(True)
            self.calculate_cost_manual = True
            self.processes.setVisible(True)
            self.processes_parameter_id = -1
            self.parameter_selector.setVisible(False)
            self.parameter_selector_label.setVisible(False)
            self.calculate_current_measurement_cost()
            self.calculate_used_budget()
        

    def clickParameterValueCheckbox(self):
        cbutton = self.sender()
        if cbutton.isChecked():
            self.manual_pms_selection = True
            self.line_edits = []
            self.labels = []
            for i in range(len(self.experiment.parameters)):
                line_edit = QLineEdit(self)
                self.param_value_layout.addWidget(line_edit, i, 1)
                self.line_edits.append(line_edit)
                label = QLabel(self)
                label.setText("Value series for parameter "+str(str(self.experiment.parameters[i]))+":")
                self.param_value_layout.addWidget(label, i, 0)
        else:
            self.manual_pms_selection = False
            for i in range(len(self.line_edits)):
                self.line_edits[i].setVisible(False)
                self.labels.setVisible(False)
  

    def reset(self, model_parameters=1):
    
        # if there is only one model parameter
        if model_parameters == 1:
            if self.initialized == False:
                self.init_single_parameter()
            self.metric_selector.clear()
            for metric in self.experiment.metrics:
                self.metric_selector.addItem(str(metric))
            self.repaint()
            self.update()
        
        # if there are several model parameters
        else:
            if self.initialized == False:
                self.init_multi_parameter()
            self.metric_selector.clear()
            for metric in self.experiment.metrics:
                self.metric_selector.addItem(str(metric))
            self.repaint()
            self.update()
        
    
    def experimentChanged(self):
        
        self.experiment = self.main_widget.getExperiment()
        
        if self.experiment is None:
            self.reset()
            return
        
        self.reset(len(self.experiment.parameters))
        

    def callpath_selection_changed(self):    
        self.calculate_current_measurement_cost()
        self.calculate_used_budget()
        self.calculate_noise_level()
