# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from PySide6.QtCore import *  # @UnusedWildImport
from PySide6.QtWidgets import *  # @UnusedWildImport

from extrap.fileio.file_reader import FileReader
from extrap.gui.components.ProgressWindow import ProgressWidget
from extrap.gui.components.dynamic_options import DynamicOptionsWidget
from extrap.util.dynamic_options import DynamicOptions


class ImportOptionsDialog(QDialog):

    def __init__(self, parent, reader: FileReader | DynamicOptions, path):
        super(ImportOptionsDialog, self).__init__(parent)

        self.experiment = None
        self.scaling_choice = None
        self.reader = reader
        self.path = path

        self.init_UI()

    def init_UI(self):
        self.setWindowTitle("Import Options")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QFormLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        self.options_widget = DynamicOptionsWidget(self, self.reader)
        self.options_widget.layout().setContentsMargins(0, 0, 0, 0)
        layout.addRow(self.options_widget)

        self.progress_widget = ProgressWidget(self)
        self.progress_widget.setContentsMargins(0, 10, 0, 0)
        layout.addRow(self.progress_widget)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                          QDialogButtonBox.StandardButton.Cancel |
                                          QDialogButtonBox.StandardButton.RestoreDefaults)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self.options_widget.reset_options)
        layout.addRow(self.buttonBox)

        # self.change_param_num()
        self.setLayout(layout)

    @Slot()
    def reject(self):
        self.progress_widget.cancel()
        super().reject()

    @Slot()
    def accept(self):
        with self.progress_widget.get_progress_bar() as pbar:
            self._on_show_progressbar()

            # read the experiment from the files
            try:
                self.experiment = self.reader.read_experiment(self.path, pbar)
            except Exception as err:
                self.close()
                raise err

            if not self.experiment:
                gui_action = self.reader.GUI_ACTION.replace('&', '')
                gui_action = gui_action[0].lower() + gui_action[1:]
                QMessageBox.critical(self,
                                     "Error",
                                     f"Could not {gui_action}, files may be corrupt!",
                                     QMessageBox.StandardButton.Ok,
                                     QMessageBox.StandardButton.Ok)
                self.close()
                return
            super().accept()

    def _on_show_progressbar(self):
        self.options_widget.setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.StandardButton.RestoreDefaults).setEnabled(False)
