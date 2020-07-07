"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""

from typing import Mapping
from PySide2.QtWidgets import QWidget, QFormLayout, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, \
    QComboBox, QVBoxLayout

from modelers.abstract_modeler import AbstractModeler, MultiParameterModeler
from modelers.modeler_options import ModelerOption, ModelerOptionsGroup
from modelers import single_parameter


class ModelerOptionsWidget(QWidget):
    def __init__(self, parent, modeler: AbstractModeler, keep_parent_enabled=False):
        super().__init__(parent)
        self._modeler = modeler
        self._single_parameter_modeler_widget = None
        self.initUI(keep_parent_enabled)

    def initUI(self, keep_parent_enabled):
        layout = QFormLayout()
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.setLayout(layout)
        self.parent().setEnabled(keep_parent_enabled)

        if isinstance(self._modeler, MultiParameterModeler) and self._modeler.single_parameter_modeler is not None:
            group = self._create_single_parameter_selection()
            layout.addRow(group)
            self.parent().setEnabled(True)

        if hasattr(self._modeler, 'OPTIONS'):
            self.parent().setEnabled(True)
            self._create_options(layout, self._modeler.OPTIONS.items())

    def _create_options(self, layout, options):
        for name, option in options:
            if isinstance(option, ModelerOption):
                layout.addRow(self._determine_name(option), self._determine_field(name, option))
            elif isinstance(option, ModelerOptionsGroup):
                group = QGroupBox(name)
                group.setToolTip(option.description)
                g_layout = QFormLayout()
                g_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
                self._create_options(g_layout, option.items())
                group.setLayout(g_layout)
                layout.addRow(group)

    def _create_single_parameter_selection(self):
        group = QGroupBox('Single Parameter Modeler')
        g_layout = QVBoxLayout()
        g_layout.setContentsMargins(0, 0, 0, 0)
        modeler_selection = QComboBox()

        def selection_changed():
            modeler = modeler_selection.currentData()
            self._modeler.single_parameter_modeler = modeler()
            options = ModelerOptionsWidget(self, self._modeler.single_parameter_modeler, keep_parent_enabled=True)
            g_layout.replaceWidget(self._single_parameter_modeler_widget, options)
            self._single_parameter_modeler_widget.deleteLater()
            self._single_parameter_modeler_widget = options

        for name, modeler in single_parameter.all_modelers.items():
            modeler_selection.addItem(name, modeler)
            if modeler == self._modeler.single_parameter_modeler:
                modeler_selection.setCurrentText(name)
        modeler_selection.currentIndexChanged.connect(selection_changed)

        self._single_parameter_modeler_widget = ModelerOptionsWidget(self, self._modeler.single_parameter_modeler)

        g_layout.addWidget(modeler_selection)
        g_layout.addWidget(self._single_parameter_modeler_widget)
        group.setLayout(g_layout)
        return group

    def check_and_apply(self, option, value):
        value = option.type(value)
        setattr(self._modeler, option.field, value)

    def _determine_name(self, option):
        if option.name is not None:
            return option.name
        return option.field.replace('_', ' ').title()

    def _determine_field(self, name, option):
        def slot(value):
            self.check_and_apply(option, value)

        if isinstance(option.range, Mapping):
            field = QComboBox()
            for name, item in option.range.items():
                field.addItem(name, item)
                if item == option.value:
                    field.setCurrentText(name)
            field.currentIndexChanged.connect(lambda: self.check_and_apply(option, field.currentData()))
        elif option.type is bool:
            field = QCheckBox()
            field.setChecked(option.value)
            field.stateChanged[int].connect(slot)
        elif option.type is float:
            field = QDoubleSpinBox()
            field.setValue(option.value)
            if isinstance(option.range, range):
                field.setRange(option.range.start, option.range.stop)
            field.valueChanged[float].connect(slot)
        elif option.type is int:
            field = QSpinBox()
            field.setValue(option.value)
            if isinstance(option.range, range):
                field.setRange(option.range.start, option.range.stop)
                field.setSingleStep(option.range.step)
            field.valueChanged[int].connect(slot)
        else:
            field = QLineEdit(str(option.value), self)
            field.textChanged[str].connect(slot)
        field.setToolTip(option.description)
        return field
