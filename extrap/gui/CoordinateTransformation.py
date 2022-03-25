from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import sympy
from PySide2.QtCore import QTimer
from PySide2.QtGui import QShowEvent
from PySide2.QtWidgets import QDialog, QDialogButtonBox, QGridLayout, QLabel, QLineEdit
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
        self._button_box = QDialogButtonBox(QDialogButtonBox.Ok
                                            | QDialogButtonBox.Cancel, self)
        self._button_box.rejected.connect(self.reject)
        self._button_box.accepted.connect(self.accept)
        self._button_box.button(QDialogButtonBox.Ok).setEnabled(False)
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
            self._button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        self._layout.addWidget(self._button_box, r_ctr, 0, 1, 4)

    def _clear_layout(self):
        for i in reversed(range(self._layout.count())):
            widget = self._layout.itemAt(i).widget()
            self._layout.removeWidget(widget)
            if widget != self._button_box:
                widget.setParent(None)

    def accept(self) -> None:
        old_params = self._experiment.parameters.copy()
        formulae = []
        old_param_names = [p.name for p in old_params]
        for i, (name_edit, formula_edit) in enumerate(zip(self._name_edits, self._formula_edits)):
            if old_params[i].name != name_edit.text().strip():
                self._experiment.parameters[i] = Parameter(name_edit.text().strip())
            formula = sympy_parser.parse_expr(formula_edit.text().strip())
            logging.debug(f"Coordinate Transformation: Formula: {formula}")
            formulae.append(formula)

        transformation = sympy.lambdify(old_param_names, Matrix(formulae))

        super(CoordinateTransformationDialog, self).accept()

        def _process():
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
