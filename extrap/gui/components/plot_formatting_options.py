import dataclasses

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDialog, QFormLayout, QFontComboBox, QSpinBox, QDialogButtonBox


@dataclasses.dataclass
class PlotFormattingOptions:
    font_family: str = 'Arial'
    font_size: int = 10
    legend_font_size: int = 6


class PlotFormattingDialog(QDialog):

    def __init__(self, options: PlotFormattingOptions, parent=None, f=Qt.WindowFlags()):
        super().__init__(parent, f)

        self._options = options

        # init UI
        self.setWindowTitle("Plot Formatting Options")
        layout = QFormLayout()
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

        _dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        _dialog_buttons.accepted.connect(self.accept)
        _dialog_buttons.rejected.connect(self.reject)
        layout.addWidget(_dialog_buttons)

        self.setLayout(layout)

    def accept(self) -> None:
        self._options.font_family = self._font_family_selector.currentFont().family()
        self._options.font_size = self._font_size_selector.value()
        self._options.legend_font_size = self._legend_font_size_selector.value()

        super().accept()
