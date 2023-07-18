# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
from __future__ import annotations

from functools import partial
from threading import Event

from PySide6.QtCore import *  # @UnusedWildImport
from PySide6.QtWidgets import *  # @UnusedWildImport

from extrap.fileio import io_helper
from extrap.fileio.file_reader import FileReader
from extrap.gui.components.dynamic_options import DynamicOptionsWidget
from extrap.util.dynamic_options import DynamicOptions
from extrap.util.exceptions import CancelProcessError
from extrap.util.progress_bar import ProgressBar


class ImportSettingsDialog(QDialog):

    def __init__(self, parent, reader: FileReader | DynamicOptions, path):
        super(ImportSettingsDialog, self).__init__(parent)

        self.experiment = None
        self.scaling_choice = None
        self.reader = reader
        self.path = path
        self._cancel_event = Event()

        self.init_UI()

    def init_UI(self):
        self.setWindowTitle("Import Settings")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        main_layout = QFormLayout(self)
        layout = QFormLayout(self)
        self.controls_layout = layout

        self.options_widget = DynamicOptionsWidget(self, self.reader)
        self.options_widget.layout().setContentsMargins(0, 0, 0, 0)
        layout.addRow(self.options_widget)

        self.progress_indicator = QProgressBar(self)
        self.progress_indicator.hide()
        layout.addRow(self.progress_indicator)

        main_layout.addRow(layout)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                          QDialogButtonBox.StandardButton.Cancel |
                                          QDialogButtonBox.StandardButton.RestoreDefaults)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self.options_widget.reset_options)
        main_layout.addRow(self.buttonBox)

        # self.change_param_num()
        self.setLayout(main_layout)

    @Slot()
    def reject(self):
        self._cancel_event.set()
        super().reject()

    @Slot()
    def accept(self):

        with ProgressBar(total=0, gui=True) as pbar:
            self._show_progressbar()
            pbar.display = partial(self._display_progress, pbar)
            pbar.sp = None

            # read the experiment from the files
            try:
                self.experiment = self.reader.read_experiment(self.path, pbar)
                check_res = io_helper.check_for_strong_scaling(self.experiment)
                print(check_res)
                if max(check_res) > 0:
                    warnings.warn(
                        "Extra-P detected that these measurements might have been taken from a strong-scaling experiment. Extra-P only supports weak-scaling experiments, do you want to convert your experiment to a weak scaling experiment?")
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

    def _show_progressbar(self):
        self.controls_layout.setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.StandardButton.RestoreDefaults).setEnabled(False)
        self.progress_indicator.show()

    def _display_progress(self, pbar: ProgressBar, msg=None, pos=None):
        if self._cancel_event.is_set():
            raise CancelProcessError()
        self.progress_indicator.setMaximum(pbar.total)
        self.progress_indicator.setValue(pbar.n)
        QApplication.processEvents()
