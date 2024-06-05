# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import *  # @UnusedWildImport

from extrap.entities.metric import Metric
from extrap.mpa.gpr_selection_strategy import analyze_noise
from extrap.mpa.measurement_point_advisor import MeasurementPointAdvisor

if TYPE_CHECKING:
    from extrap.gui.MainWidget import MainWidget


class MeasurementWizardWidget(QWidget):

    def __init__(self, mainWidget, parent):
        super(MeasurementWizardWidget, self).__init__(parent)
        self.main_widget: MainWidget = mainWidget
        self.main_layout = QHBoxLayout(self)
        self.calculate_cost_manual = False
        self.experiment = None
        self.mean_noise_level = 0.0
        self.processes_parameter_id = 0
        self.no_model_parameters = 0
        self.budget = 0.0
        self.current_cost = 0.0
        self.manual_pms_selection = False
        self.parameter_value_series = []
        self._layout = None
        self.empty_label = None
        self.init_empty()

    def init_empty(self):
        self.no_model_parameters = 0
        self.setLayout(self.main_layout)
        if not self.empty_label:
            self.empty_label = QLabel(self)
            self.empty_label.setWordWrap(True)
            self.main_layout.addWidget(self.empty_label)
        if msg is None:
            msg = 'Please load a performance experiment to get suggestions for further measurement points.'
        self.empty_label.setText(msg)

    def init_ui(self):
        self.no_model_parameters = len(self.experiment.parameters)
        if self.empty_label is not None:
            self.main_layout.removeWidget(self.empty_label)
            self.empty_label.deleteLater()
            self.empty_label = None

        self._layout = QGridLayout(self)
        self._layout.setRowStretch(99, 1)
        self._layout.setColumnStretch(0, 0)
        self._layout.setColumnStretch(1, 1)

        self.main_layout.addLayout(self._layout)

        if self.no_model_parameters == 1:
            label = "Model parameter is the no. of processes (MPI ranks)"
        else:
            label = "No. of processes (MPI ranks) is one of the model parameters"
        self.checkbox = QCheckBox(label, self)
        self.checkbox.setChecked(True)
        self.checkbox.toggled.connect(self.clickCheckbox)
        self._layout.addWidget(self.checkbox, 0, 0, 1, 2)

        if self.no_model_parameters == 1:
            self.processes_parameter_id = 0
            self.parameter_selector_label = None
            self.parameter_selector = None
        else:
            self.parameter_selector_label = QLabel(self)
            self.parameter_selector_label.setText("Model Parameter resembling the\n number of processes (MPI ranks):")
            self._layout.addWidget(self.parameter_selector_label, 1, 0)

            self.parameter_selector = QComboBox(self)
            for parameter in self.experiment.parameters:
                self.parameter_selector.addItem(str(parameter))
            self._layout.addWidget(self.parameter_selector, 1, 1)
            for i in range(len(self.experiment.parameters)):
                if self.parameter_selector.currentText() == str(self.experiment.parameters[i]):
                    self.processes_parameter_id = i
                    break
            self.parameter_selector.currentIndexChanged.connect(self.parameterChanged)

        self.process_label = QLabel(self)
        self.process_label.setText("No. of processes (MPI ranks):")
        self._layout.addWidget(self.process_label, 2, 0)
        self.process_label.setVisible(False)

        self.processes = QLineEdit(self)
        self.processes.setValidator(QIntValidator())
        self._layout.addWidget(self.processes, 2, 1)
        self.processes.setVisible(False)
        self.processes.setText("2")
        self.processes.textChanged.connect(self.processesChanged)

        metric_label = QLabel(self)
        metric_label.setText("Metric used for cost calculation:")
        self._layout.addWidget(metric_label, 3, 0)

        self.metric_selector = QComboBox(self)
        for metric in self.experiment.metrics:
            self.metric_selector.addItem(str(metric))
        self._layout.addWidget(self.metric_selector, 3, 1)
        self.metric_selector.currentIndexChanged.connect(self.metricChanged)

        budget_label = QLabel(self)
        budget_label.setText("Modeling Budget [core hours]:")
        self._layout.addWidget(budget_label, 4, 0)

        self.modeling_budget = QDoubleSpinBox(self)
        self.modeling_budget.setMaximum(math.inf)
        self.modeling_budget.setMinimum(0.0)
        self._layout.addWidget(self.modeling_budget, 4, 1)
        self.modeling_budget.textChanged.connect(self.budgetChanged)

        current_cost_label = QLabel(self)
        current_cost_label.setText("Current Measurement Cost [core hours]:")
        self._layout.addWidget(current_cost_label, 5, 0)

        self.current_cost_edit = QLineEdit(self)
        self.current_cost_edit.setReadOnly(True)
        self.current_cost_edit.setText("0.0")
        self._layout.addWidget(self.current_cost_edit, 5, 1)

        used_budget_label = QLabel(self)
        used_budget_label.setText("Used Modeling Budget [%]:")
        self._layout.addWidget(used_budget_label, 6, 0)

        self.used_budget_edit = QLineEdit(self)
        self.used_budget_edit.setReadOnly(True)
        self.used_budget_edit.setText("0.0")
        self._layout.addWidget(self.used_budget_edit, 6, 1)

        noise_level_label = QLabel(self)
        noise_level_label.setText("Measurement Noise Level [%]:")
        self._layout.addWidget(noise_level_label, 7, 0)

        self.noise_level_edit = QLineEdit(self)
        self.noise_level_edit.setReadOnly(True)
        noise_level_str = "{:.2f}".format(self.mean_noise_level)
        self.noise_level_edit.setText(noise_level_str)
        self._layout.addWidget(self.noise_level_edit, 7, 1)

        self.parameter_value_checkbox = QCheckBox("Enter parameter-value series manually", self)
        self.parameter_value_checkbox.setChecked(False)
        self.parameter_value_checkbox.toggled.connect(self.clickParameterValueCheckbox)
        self._layout.addWidget(self.parameter_value_checkbox, 8, 0, 1, 2)

        self.param_value_layout = QGridLayout(self)
        self.param_value_layout.setRowStretch(99, 1)
        self.param_value_layout.setColumnStretch(0, 0)
        self.param_value_layout.setColumnStretch(1, 1)
        self._layout.addLayout(self.param_value_layout, 9, 0, 1, 2)

        self.advise_button = QPushButton(self)
        self._layout.addWidget(self.advise_button, 10, 0, 1, 2)
        self.advise_button.setText("Suggest Additional Measurement Points")
        self.advise_button.clicked.connect(self.advice_button_clicked)

        suggestion_label = QLabel(self)
        suggestion_label.setText("Measurement point suggestions:")
        self._layout.addWidget(suggestion_label, 11, 0)

        self.measurement_point_suggestions_label = QPlainTextEdit(self)
        self.measurement_point_suggestions_label.setPlainText("")
        self.measurement_point_suggestions_label.setReadOnly(True)
        self._layout.addWidget(self.measurement_point_suggestions_label, 12, 0, 1, 2)

        self.calculate_current_measurement_cost()
        self.calculate_noise_level()

    def calculate_noise_level(self):
        if self.no_model_parameters == 0:
            return
        # do an noise analysis on the existing measurement points
        metric_string = self.metric_selector.currentText()
        runtime_metric = Metric(metric_string)

        selected_callpath = self._get_selected_callpaths()

        temp_noise = []
        for callpath in selected_callpath:
            temp_noise.append(analyze_noise(self.experiment, callpath, runtime_metric) * 100)
        self.mean_noise_level = np.mean(temp_noise)

        noise_level_str = "{:.2f}".format(self.mean_noise_level)
        self.noise_level_edit.setText(noise_level_str)

    def calculate_used_budget(self):
        if self.no_model_parameters == 0:
            return
        self.budget = self.modeling_budget.value()
        if self.budget == 0.0:
            self.budget = self.current_cost
        used_budget_percent = self.current_cost / (self.budget / 100)
        used_budget_percent_str = "{:.2f}".format(used_budget_percent)
        self.used_budget_edit.setText(str(used_budget_percent_str))

    def calculate_current_measurement_cost(self):
        if self.no_model_parameters == 0:
            return
        current_cost_str = "0.0"
        metric_string = self.metric_selector.currentText()
        metrics = self.experiment.metrics
        if Metric(metric_string) not in metrics:
            return
        else:
            runtime_metric = Metric(metric_string)
        selected_callpath = self._get_selected_callpaths()

        measurements = self.experiment.measurements
        core_time_total = 0

        mpa = MeasurementPointAdvisor(budget=self.budget, process_parameter_id=self.processes_parameter_id,
                                      callpaths=selected_callpath, metric=runtime_metric,
                                      experiment=self.experiment, current_cost=self.current_cost,
                                      manual_pms_selection=self.manual_pms_selection,
                                      manual_parameter_value_series=self.parameter_value_series,
                                      calculate_cost_manual=self.calculate_cost_manual,
                                      number_processes=self._get_number_processes(), model_generator=None)

        for callpath in selected_callpath:
            core_time_callpath = 0
            for measurement in measurements[(callpath, runtime_metric)]:
                core_time_per_point = mpa.calculate_cost(measurement.coordinate,
                                                         measurement.mean) * measurement.repetitions
                core_time_callpath += core_time_per_point
            core_time_total += core_time_callpath

        self.current_cost = core_time_total
        current_cost_str = "{:.2f}".format(self.current_cost)

        if self.budget == 0.0:
            self.budget = self.current_cost
            self.modeling_budget.setValue(self.current_cost)

        self.current_cost_edit.setText(current_cost_str)

    def advice_button_clicked(self):

        # get the modeling budget from the GUI
        if int(self.budget) <= int(self.current_cost):
            self.measurement_point_suggestions_label.setPlainText(
                "Not enough budget available to suggest further measurement points!")

        else:
            # get the performance metric that should be used for the analysis
            metric_string = self.metric_selector.currentText()
            metrics = self.experiment.metrics
            runtime_metric = None
            for metric in metrics:
                if metric_string == str(metric):
                    runtime_metric = metric

            # get the selected callpath(s) in the tree
            selected_callpath = self._get_selected_callpaths()

            # get the currently used model from the GUI
            model_gen = self.main_widget.get_current_model_gen()

            # get parameter value series if manual selection
            if self.manual_pms_selection:
                for i in range(len(self.experiment.parameters)):
                    self.parameter_value_series.append(self.line_edits[i].text())

            number_processes = self._get_number_processes()

            # initialize a new measurement point advisor object
            mpa = MeasurementPointAdvisor(budget=self.budget, process_parameter_id=self.processes_parameter_id,
                                          callpaths=selected_callpath, metric=runtime_metric,
                                          experiment=self.experiment, current_cost=self.current_cost,
                                          manual_pms_selection=self.manual_pms_selection,
                                          manual_parameter_value_series=self.parameter_value_series,
                                          calculate_cost_manual=self.calculate_cost_manual,
                                          number_processes=number_processes, model_generator=model_gen)

            point_suggestions, rep_numbers = mpa.suggest_points()

            if len(point_suggestions) == 0:
                text = ("No measurement points could be found that fit into the available budget. Please consider "
                        "increasing the modeling budget.")

            else:
                text = ""
                for i in range(len(point_suggestions)):
                    temp = ""
                    temp += str(i + 1) + "."
                    temp += " P("
                    parameter_values = point_suggestions[i].as_tuple()
                    for j in range(len(self.experiment.parameters)):
                        if j != 0:
                            temp += ","
                        temp += str(self.experiment.parameters[j])
                        temp += "="
                        parameter_value = parameter_values[j]
                        temp += str(parameter_value)
                    if rep_numbers is not None:
                        rep_number = rep_numbers[i]
                    else:
                        rep_number = 1
                    temp += "), repetition no.="
                    temp += str(rep_number)
                    temp += "\n"
                    text += temp

            self.measurement_point_suggestions_label.setPlainText(text)

    def _get_selected_callpaths(self):
        selected_callpath = self.main_widget.get_selected_call_tree_nodes()
        if len(selected_callpath) == 0:
            selected_callpath = self.experiment.callpaths
        else:
            selected_callpath = [n.path for n in selected_callpath]
        return selected_callpath

    def _get_number_processes(self):
        if self.calculate_cost_manual:
            try:
                number_processes = int(self.processes.text())
            except ValueError as e:
                number_processes = 1
                # raise RecoverableError("Number of processes must be a number.") from e
        else:
            number_processes = 0
        return number_processes

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
            self.processes.setVisible(False)
            if self.parameter_selector:
                for i in range(len(self.experiment.parameters)):
                    if str(self.experiment.parameters[i]) == self.parameter_selector.currentText():
                        self.processes_parameter_id = i
                        break
                self.parameter_selector_label.setVisible(True)
                self.parameter_selector.setVisible(True)
            else:
                self.processes_parameter_id = 0
            self.calculate_current_measurement_cost()
            self.calculate_used_budget()
        else:
            self.process_label.setVisible(True)
            self.calculate_cost_manual = True
            self.processes.setVisible(True)
            self.processes_parameter_id = -1
            if self.parameter_selector:
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
                label.setText("Value series for parameter " + str(str(self.experiment.parameters[i])) + ":")
                self.labels.append(label)
                self.param_value_layout.addWidget(label, i, 0)
        else:
            self.manual_pms_selection = False
            for i in range(len(self.line_edits)):
                self.line_edits[i].setVisible(False)
                self.labels[i].setVisible(False)

    def reset(self, model_parameters=0):
        if self._layout:
            while (child := self._layout.takeAt(0)) is not None:
                widget = child.widget()
                if hasattr(widget, "deleteLater"):
                    widget.deleteLater()
            self._layout.deleteLater()
            self._layout = None

        if model_parameters == 0:
            self.init_empty()
        elif not experiment_has_repetitions(self.experiment):
            self.init_empty('<p><b>Please load a performance experiment that contains '
                            'repetition information to get suggestions '
                            'for further measurement points.</b></p>'
                            '<p>Your current experiment does not contain repetition information. '
                            'It was likely created with an older version of Extra-P '
                            'which did not store the necessary data.</p>'
                            )
        # if there are several model parameters
        else:
            self.init_ui()

        if model_parameters != 0:
            self.metric_selector.clear()
            for metric in self.experiment.metrics:
                name = metric.name if metric.name != '' else '<default>'
                self.metric_selector.addItem(name)

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
