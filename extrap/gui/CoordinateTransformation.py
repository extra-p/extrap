# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import sympy
from PySide6.QtCore import QTimer
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QGridLayout, QLabel, QLineEdit
from sympy import Matrix
from sympy.parsing import sympy_parser

from extrap.entities.coordinate import Coordinate
from extrap.entities.parameter import Parameter
from extrap.gui.components.ProgressWindow import ProgressWindow
from extrap.util.unique_list import UniqueList

if TYPE_CHECKING:
    from extrap.gui.MainWidget import MainWidget


class CoordinateTransformationDialog(QDialog):
    def __init__(self, parent: MainWidget):
        super(CoordinateTransformationDialog, self).__init__(parent)
        self._main_widget = parent
        self.setWindowTitle('Transform Coordinates')
        self._layout = QGridLayout()
        self.setLayout(self._layout)
        self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                            | QDialogButtonBox.StandardButton.Cancel, self)
        self._button_box.rejected.connect(self.reject)
        self._button_box.accepted.connect(self.accept)
        self._button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._name_edits: list[QLineEdit] = []
        self._formula_edits: list[QLineEdit] = []

    def showEvent(self, event: QShowEvent) -> None:
        super(CoordinateTransformationDialog, self).showEvent(event)
        self._clear_layout()
        self._name_edits.clear()
        self._formula_edits.clear()
        self._experiment = self._main_widget.getExperiment()
        r_ctr = 0
        if self._experiment:
            available_params = QLabel('Available params: ' + ', '.join(p.name for p in self._experiment.parameters))
            self._layout.setColumnMinimumWidth(3, 200)
            self._layout.addWidget(available_params, 0, 0, 1, 4)
            self._layout.addWidget(QLabel('Original Name'), 1, 0)
            self._layout.addWidget(QLabel('Name'), 1, 1)
            self._layout.addWidget(QLabel('Formula'), 1, 3)
            r_ctr = 2
            for param in self._experiment.parameters:
                self._layout.addWidget(QLabel(param.name + ':'), r_ctr, 0)
                name_edit = QLineEdit(param.name)
                self._name_edits.append(name_edit)
                self._layout.addWidget(name_edit, r_ctr, 1)
                self._layout.addWidget(QLabel("="), r_ctr, 2)
                formula_edit = QLineEdit(param.name)
                self._formula_edits.append(formula_edit)
                self._layout.addWidget(formula_edit, r_ctr, 3)
                r_ctr += 1
            self._button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self._layout.addWidget(self._button_box, r_ctr, 0, 1, 4)

    def _clear_layout(self):
        for i in reversed(range(self._layout.count())):
            widget = self._layout.itemAt(i).widget()
            self._layout.removeWidget(widget)
            if widget != self._button_box:
                widget.setParent(None)

    def accept(self) -> None:
        old_param_names = [p.name for p in self._experiment.parameters]
        new_param_names = [name_edit.text().strip() for name_edit in self._name_edits]

        for i, (param_name, formula_edit) in enumerate(zip(new_param_names, self._formula_edits)):
            if not param_name:
                warnings.warn("Parameters cannot be empty.")
                return
            if old_param_names[i] != param_name \
                    and Parameter(param_name) in self._experiment.parameters:
                warnings.warn(f"Parameter {param_name} already exists, "
                              "you cannot have two parameters with the same name.")
                return
            if new_param_names.count(param_name) > 1:
                warnings.warn(f"Parameter {param_name} already exists, "
                              "you cannot have two parameters with the same name.")
                return

        changed = False
        formulae = []
        for i, (param_name, formula_edit) in enumerate(zip(new_param_names, self._formula_edits)):
            try:
                formula_input = formula_edit.text().strip()
                if formula_input != old_param_names[i]:
                    changed = True
                formula = sympy_parser.parse_expr(formula_input)
            except SyntaxError:
                warnings.warn(f"Syntax error in formula of parameter {old_param_names[i]}")
                return
            logging.debug(f"Coordinate Transformation: Formula: {formula}")
            formulae.append(formula)

        if changed:
            transformation = sympy.lambdify(old_param_names, Matrix(formulae))
        else:
            transformation = None

        super(CoordinateTransformationDialog, self).accept()

        def _process():
            for i, (param_name, formula_edit) in enumerate(zip(new_param_names, self._formula_edits)):
                if old_param_names[i] != param_name:
                    self._experiment.parameters[i] = Parameter(param_name)
            if transformation:
                self._experiment.coordinates = UniqueList(
                    Coordinate(transformation(*coord).reshape(-1)) for coord in self._experiment.coordinates)
                with ProgressWindow(self, 'Transforming') as pbar:
                    pbar.total += len(self._experiment.measurements)
                    for modeler in self._experiment.modelers:
                        pbar.total += len(modeler.models)
                    for m in self._experiment.measurements:
                        measurement_list = self._experiment.measurements[m]
                        for measurement in measurement_list:
                            pbar.update(0)
                            measurement.coordinate = Coordinate(transformation(*measurement.coordinate).reshape(-1))
                        pbar.update()
                    for modeler in self._experiment.modelers:
                        modeler.model_all(pbar, auto_append=False)
                        pbar.total -= len(modeler.models)

            self._main_widget.set_experiment(self._experiment)

        QTimer.singleShot(0, _process)
