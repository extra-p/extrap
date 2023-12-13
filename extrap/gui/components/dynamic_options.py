# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Mapping, Collection, cast, Optional

from PySide6.QtWidgets import QWidget, QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox, \
    QComboBox, QLabel, QPushButton

from extrap.gui.components.switch_widget import SwitchWidget
from extrap.util.dynamic_options import DynamicOptions, DynamicOptionsGroup, DynamicOption


class DynamicOptionsWidget(QWidget):
    def __init__(self, parent, object_with_options: Optional[DynamicOptions], has_parent_options=False,
                 has_reset_button=False):
        super().__init__(parent)
        self.object_with_options = object_with_options
        self._has_reset_button = has_reset_button
        self._layout = QFormLayout()
        self._has_parent_options = has_parent_options
        self.init_ui()

    def init_ui(self):
        self._layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.setLayout(self._layout)
        if self.object_with_options and self.object_with_options.OPTIONS:
            if not self._has_parent_options and self._has_reset_button:
                reset_button = QPushButton('Reset options to defaults', self)
                reset_button.clicked.connect(self.reset_options)
                self._layout.addRow(reset_button)
            self._create_options(self._layout, self.object_with_options.OPTIONS.items())

    def reset_options(self):
        for option in self.object_with_options.options_iter():
            setattr(self.object_with_options, option.field, option.value)
        for i in reversed(range(self._layout.count())):
            self._layout.itemAt(i).widget().setParent(cast(QWidget, None))
        self.init_ui()
        self.update()

    def update_object_with_options(self, object_with_options):
        self.object_with_options = object_with_options
        for i in reversed(range(self._layout.count())):
            self._layout.itemAt(i).widget().setParent(cast(QWidget, None))
        self.init_ui()
        self.update()

    def _create_options(self, layout, options):
        for name, option in options:
            if isinstance(option, DynamicOption):
                if option.explanation_above:
                    layout.addWidget(QLabel(option.explanation_above))
                layout.addRow(self._determine_label(option), self._determine_field(option))
            elif isinstance(option, DynamicOptionsGroup):
                group = QGroupBox(name)
                group.setToolTip(option.description)
                g_layout = QFormLayout()
                g_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
                self._create_options(g_layout, option.items())
                group.setLayout(g_layout)
                layout.addRow(group)
            else:
                raise TypeError(f"Unsupported type {type(option)} for {name} in options.")

    def _check_and_apply(self, option, value):
        value = option.type(value)
        setattr(self.object_with_options, option.field, value)

    @staticmethod
    def _determine_label(option):
        if option.name is not None:
            name = option.name
        else:
            name = option.field.replace('_', ' ')
            name = name[0].capitalize() + name[1:]
        label = QLabel(name)
        label.setToolTip(option.description)
        return label

    def _determine_field(self, option):
        def slot(value):
            self._check_and_apply(option, value)

        if isinstance(option.range, Mapping):
            field = QComboBox()
            for name, item in option.range.items():
                field.addItem(name, item)
                if item == option.value:
                    field.setCurrentText(name)
            field.currentIndexChanged.connect(lambda: self._check_and_apply(option, field.currentData()))
        elif option.type is str and isinstance(option.range, Collection):
            field = QComboBox()
            for name in option.range:
                field.addItem(str(name))
                if name == option.value:
                    field.setCurrentText(name)
            field.currentIndexChanged.connect(lambda: self._check_and_apply(option, field.currentText()))
        elif option.type is bool:
            field = SwitchWidget()
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
