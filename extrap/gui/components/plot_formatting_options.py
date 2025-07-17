# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import dataclasses
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDialog, QFormLayout, QFontComboBox, QSpinBox, QDialogButtonBox, QLayout, QComboBox

from extrap.gui.components.model_color_map import ModelColorMap


@dataclasses.dataclass
class PlotFormattingOptions:
    font_family: str = 'Arial'
    font_size: int = 10
    legend_font_size: int = 6
    surface_opacity: float = 1.0


class PlotFormattingDialog(QDialog):

    def __init__(self, options: PlotFormattingOptions, parent=None, f=..., model_color_map=None):
        super().__init__(parent, f)

        self._options = options
        self._model_color_map = model_color_map

        # init UI
        self.setWindowTitle("Plot Formatting Options")
        layout = QFormLayout()
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self._font_family_selector = QFontComboBox()
        self._font_family_selector.setCurrentFont(QFont(self._options.font_family))
        layout.addRow("Font family", self._font_family_selector)
        self._font_size_selector = QSpinBox()
        self._font_size_selector.setMinimum(4)
        self._font_size_selector.setMaximum(20)
        self._font_size_selector.setValue(self._options.font_size)
        layout.addRow("Font size", self._font_size_selector)
        self._legend_font_size_selector = QSpinBox()
        self._legend_font_size_selector.setMinimum(4)
        self._legend_font_size_selector.setMaximum(20)
        self._legend_font_size_selector.setValue(self._options.legend_font_size)
        layout.addRow("Legend font size", self._legend_font_size_selector)
        self._colormap_selector = QComboBox()
        self._colormap_selector.addItems(ModelColorMap.colormaps)
        self._colormap_selector.setCurrentText(self._model_color_map.name)
        self._colormap_selector.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        layout.addRow("Colormap", self._colormap_selector)
        self._opacity_selector = QSpinBox()
        self._opacity_selector.setMinimum(0)
        self._opacity_selector.setMaximum(100)
        self._opacity_selector.setValue(int(self._options.surface_opacity * 100))
        layout.addRow("Surface opacity", self._opacity_selector)

        _dialog_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        _dialog_buttons.accepted.connect(self.accept)
        _dialog_buttons.rejected.connect(self.reject)
        layout.addWidget(_dialog_buttons)

        self.setLayout(layout)

    def accept(self) -> None:
        self._options.font_family = self._font_family_selector.currentFont().family()
        self._options.font_size = self._font_size_selector.value()
        self._options.legend_font_size = self._legend_font_size_selector.value()
        self._model_color_map.set_colormap(self._colormap_selector.currentText())
        self._options.surface_opacity = self._opacity_selector.value() / 100

        super().accept()
