# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Sequence, Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QDialog, QWidget, QFormLayout, QLabel, QDialogButtonBox, QComboBox, QMessageBox, QLayout, \
    QStyle, QGridLayout, QSizePolicy, QSpacerItem

from extrap.entities.experiment import Experiment
from extrap.gui.components.ProgressWindow import ProgressWidget

_TITLE = "Strong-Scaling Experiment Detected"

_MESSAGE = "<h3>Modeling strong-scaling is not supported, <br>do you want to convert to a " \
           "weak-scaling experiment?</h3>" \
           "<p>Extra-P detected that these measurements might describe a strong-scaling experiment, but modeling " \
           "is only supported for weak-scaling experiments. If you continue without conversion, the models " \
           "might not be correct.</p>" \
           "<p>After the conversion Extra-P will model your metrics globally, e.g., the run time will be the sum " \
           "of the run times of all ranks/threads.</p>"


class StrongScalingConversionDialog(QDialog):

    def __init__(self, experiment: Experiment, check_result: Sequence[int], parent: QWidget = None,
                 flags: Qt.WindowType = Qt.WindowType.Dialog):
        super().__init__(parent, flags)
        self.experiment = experiment
        self._check_result = check_result
        self._parameter_selection_cb: Optional[QComboBox] = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle(_TITLE)

        main_layout = QGridLayout(self)
        main_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        main_layout.setColumnStretch(0, 0)
        main_layout.setColumnStretch(1, 0)
        main_layout.setColumnStretch(2, 1)
        main_layout.setColumnMinimumWidth(2, 400)
        self.setLayout(main_layout)

        icon = QLabel()
        icon.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(
            self.style().pixelMetric(QStyle.PixelMetric.PM_MessageBoxIconSize, widget=self)))
        main_layout.addWidget(icon, 0, 0, 1, 1, Qt.AlignTop)
        main_layout.addItem(QSpacerItem(7, 1, QSizePolicy.Fixed, QSizePolicy.Fixed), 0, 1)

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        main_layout.addLayout(layout, 0, 2)

        message = QLabel(_MESSAGE)
        message.setWordWrap(True)
        layout.addRow(message)

        if len(self.experiment.parameters) >= 1:
            self._parameter_selection_cb = QComboBox(self)
            self._parameter_selection_cb.addItems([p.name for p in self.experiment.parameters])
            self._parameter_selection_cb.setCurrentIndex(self._check_result.index(max(self._check_result)))

            layout.addRow("<b>Strong-scaling parameter</b>", self._parameter_selection_cb)

            hint = QLabel(
                "The scaling parameter represents the computational resources (e.g., ranks, threads, processes). "
                "If the automatic detection identified the wrong parameter, you should correct it.")
            hint.setWordWrap(True)
            layout.addRow(hint)

        self.progress_widget = ProgressWidget(self)
        self.progress_widget.setContentsMargins(0, 10, 0, 0)
        main_layout.addWidget(self.progress_widget, 1, 0, 1, 3)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No)
        self.buttons.button(QDialogButtonBox.StandardButton.Yes).setText("Convert")
        self.buttons.button(QDialogButtonBox.StandardButton.No).setText("Don't convert")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        main_layout.addWidget(self.buttons, 2, 0, 1, 3)

    @Slot()
    def accept(self) -> None:
        with self.progress_widget.get_progress_bar() as pbar:

            if self._parameter_selection_cb is None:
                selected_dimension = 0
            else:
                self._parameter_selection_cb.setEnabled(False)
                selected_dimension = self._parameter_selection_cb.currentIndex()

            for measurements in pbar(self.experiment.measurements.values(), len(self.experiment.measurements)):
                for m in measurements:
                    m *= m.coordinate[selected_dimension]
        super().accept()

    @staticmethod
    def pose_question_for_readers_with_scaling_conversion(parent):
        msg_box = QMessageBox(QMessageBox.Icon.Warning, _TITLE, _MESSAGE,
                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, parent)
        msg_box.button(QMessageBox.StandardButton.Yes).setText("Convert")
        msg_box.button(QMessageBox.StandardButton.No).setText("Don't convert")
        msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)

        return msg_box.exec()
