"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""

from typing import Mapping, Collection, cast

from PySide2.QtWidgets import QWidget, QFormLayout, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, \
    QComboBox, QVBoxLayout, QLabel, QPushButton

from extrap.modelers import single_parameter
from extrap.modelers.abstract_modeler import AbstractModeler, MultiParameterModeler
from extrap.modelers.modeler_options import ModelerOption, ModelerOptionsGroup, modeler_options


class ModelerOptionsWidget(QWidget):
    def __init__(self, parent, modeler: AbstractModeler, has_parent_options=False):
        super().__init__(parent)
        self._modeler = modeler
        self._single_parameter_modeler_widget = None
        layout = QFormLayout()
        self.initUI(layout, has_parent_options)

    def initUI(self, layout, has_parent_options):
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.setLayout(layout)
        self.parent().setEnabled(has_parent_options)

        if hasattr(self._modeler, 'OPTIONS'):
            self.parent().setEnabled(True)
            if not has_parent_options:
                reset_button = QPushButton('Reset options to defaults', self)
                reset_button.clicked.connect(self._reset_options)
                layout.addRow(reset_button)
            self._create_options(layout, getattr(self._modeler, 'OPTIONS').items())

        if isinstance(self._modeler, MultiParameterModeler) and self._modeler.single_parameter_modeler is not None:
            group = self._create_single_parameter_selection()
            layout.addRow(group)
            self.parent().setEnabled(True)

    def _reset_options(self):
        for option in modeler_options.iter(self._modeler):
            setattr(self._modeler, option.field, option.value)
        layout = self.layout()
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(cast(QWidget, None))
        if isinstance(self._modeler, MultiParameterModeler):
            self._modeler.reset_single_parameter_modeler()
        self.initUI(cast(QFormLayout, layout), False)
        self.update()

    def _create_options(self, layout, options):
        for name, option in options:
            if isinstance(option, ModelerOption):
                layout.addRow(self._determine_label(option), self._determine_field(option))
            elif isinstance(option, ModelerOptionsGroup):
                group = QGroupBox(name)
                group.setToolTip(option.description)
                g_layout = QFormLayout()
                g_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
                self._create_options(g_layout, option.items())
                group.setLayout(g_layout)
                layout.addRow(group)

    def _create_single_parameter_selection(self):
        self._modeler: MultiParameterModeler

        group = QGroupBox('Single Parameter Modeler')
        g_layout = QVBoxLayout()
        g_layout.setContentsMargins(0, 0, 0, 0)
        modeler_selection = QComboBox()

        def selection_changed():
            modeler = modeler_selection.currentData()
            self._modeler.single_parameter_modeler = modeler()
            options = ModelerOptionsWidget(self, self._modeler.single_parameter_modeler, has_parent_options=True)
            g_layout.replaceWidget(self._single_parameter_modeler_widget, options)
            self._single_parameter_modeler_widget.deleteLater()
            self._single_parameter_modeler_widget = options

        for name, modeler in single_parameter.all_modelers.items():
            modeler_selection.addItem(name, modeler)
            if isinstance(self._modeler.single_parameter_modeler, modeler):
                modeler_selection.setCurrentText(name)
        modeler_selection.currentIndexChanged.connect(selection_changed)

        self._single_parameter_modeler_widget = ModelerOptionsWidget(self, self._modeler.single_parameter_modeler,
                                                                     has_parent_options=True)

        g_layout.addWidget(modeler_selection)
        g_layout.addWidget(self._single_parameter_modeler_widget)
        g_layout.addStrut(1)
        group.setLayout(g_layout)
        return group

    def check_and_apply(self, option, value):
        value = option.type(value)
        setattr(self._modeler, option.field, value)

    @staticmethod
    def _determine_label(option):
        if option.name is not None:
            name = option.name
        else:
            name = option.field.replace('_', ' ').title()
        label = QLabel(name)
        label.setToolTip(option.description)
        return label

    def _determine_field(self, option):
        def slot(value):
            self.check_and_apply(option, value)

        if isinstance(option.range, Mapping):
            field = QComboBox()
            for name, item in option.range.items():
                field.addItem(name, item)
                if item == option.value:
                    field.setCurrentText(name)
            field.currentIndexChanged.connect(lambda: self.check_and_apply(option, field.currentData()))
        elif option.type is str and isinstance(option.range, Collection):
            field = QComboBox()
            for name in option.range:
                field.addItem(str(name))
                if name == option.value:
                    field.setCurrentText(name)
            field.currentIndexChanged.connect(lambda: self.check_and_apply(option, field.currentText()))
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
