# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, \
    QComboBox, QVBoxLayout

from extrap.gui.components.dynamic_options import DynamicOptionsWidget
from extrap.modelers import single_parameter
from extrap.modelers.abstract_modeler import AbstractModeler, MultiParameterModeler
from extrap.util.dynamic_options import DynamicOptions


class ModelerOptionsWidget(DynamicOptionsWidget):
    object_with_options: Union[AbstractModeler, DynamicOptions]

    def __init__(self, parent, modeler: AbstractModeler, has_parent_options=False):
        self._single_parameter_modeler_widget = None
        super().__init__(parent, modeler, has_parent_options, has_reset_button=True)

    def init_ui(self):
        super().init_ui()
        self.parent().setEnabled(self._has_parent_options)

        if self.object_with_options and self.object_with_options.OPTIONS:
            self.parent().setEnabled(True)
        if isinstance(self.object_with_options,
                      MultiParameterModeler) and self.object_with_options.single_parameter_modeler is not None:
            group = self._create_single_parameter_selection()
            self._layout.addRow(group)
            self.parent().setEnabled(True)

    def _create_single_parameter_selection(self):
        self.object_with_options: MultiParameterModeler

        group = QGroupBox('Single-parameter modeler')
        g_layout = QVBoxLayout()
        g_layout.setContentsMargins(0, 0, 0, 0)
        modeler_selection = QComboBox()

        def selection_changed():
            modeler = modeler_selection.currentData()
            self.object_with_options.single_parameter_modeler = modeler()
            options = ModelerOptionsWidget(self, self.object_with_options.single_parameter_modeler,
                                           has_parent_options=True)
            g_layout.replaceWidget(self._single_parameter_modeler_widget, options)
            self._single_parameter_modeler_widget.deleteLater()
            self._single_parameter_modeler_widget = options

        for i, (name, modeler) in enumerate(single_parameter.all_modelers.items()):
            modeler_selection.addItem(name, modeler)
            modeler_selection.setItemData(i, modeler.DESCRIPTION, Qt.ItemDataRole.ToolTipRole)
            if isinstance(self.object_with_options.single_parameter_modeler, modeler):
                modeler_selection.setCurrentText(name)
        modeler_selection.currentIndexChanged.connect(selection_changed)

        self._single_parameter_modeler_widget = ModelerOptionsWidget(self,
                                                                     self.object_with_options.single_parameter_modeler,
                                                                     has_parent_options=True)

        g_layout.addWidget(modeler_selection)
        g_layout.addWidget(self._single_parameter_modeler_widget)
        g_layout.addStrut(1)
        group.setLayout(g_layout)
        return group
