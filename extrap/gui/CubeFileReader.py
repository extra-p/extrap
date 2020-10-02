# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from functools import partial
from threading import Event

from PySide2.QtCore import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport

from extrap.fileio.cube_file_reader2 import read_cube_file
from extrap.util.exceptions import CancelProcessError
from extrap.util.progress_bar import ProgressBar


class ParameterWidget(QWidget):

    def __init__(self, parent):
        super(ParameterWidget, self).__init__(parent)
        self.name = "Parameter"
        self.values = "1"

    def init_UI(self):
        layout = QFormLayout(self)
        self.name_edit = QLineEdit(self)
        self.name_edit.setText(self.name)
        layout.addRow("Parameter name:", self.name_edit)

        self.values_edit = QLineEdit(self)
        self.values_edit.setText(self.values)
        layout.addRow("Values:", self.values_edit)

        self.setLayout(layout)

    def onNewValues(self):
        self.name_edit.setText(self.name)
        self.values_edit.setText(self.values)


class CubeFileReader(QDialog):

    def __init__(self, parent, dirName):
        super(CubeFileReader, self).__init__(parent)

        self.valid = False
        self.dir_name = dirName
        self.num_params = 1
        self.max_params = 3
        self.prefix = ""
        self.postfix = ""
        self.filename = "profile.cubex"
        self.repetitions = 1
        self.parameters = list()

        self._cancel_event = Event()

        # for _ in range(0, self.max_params):
        #     self.parameters.append(ParameterWidget(self))

        self.init_UI()

    def init_UI(self):
        self.setWindowTitle("Import settings")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        main_layout = QFormLayout(self)
        layout = QFormLayout(self)
        self.controls_layout = layout

        # self.num_params_choice = QSpinBox(self)
        # self.num_params_choice.setMinimum(1)
        # self.num_params_choice.setMaximum(self.max_params)
        # self.num_params_choice.setValue(self.num_params)
        # self.num_params_choice.valueChanged.connect(self.change_param_num)
        #
        # layout.addRow("Number of Parameters:", self.num_params_choice)
        #
        # self.prefix_edit = QLineEdit(self)
        # self.prefix_edit.setText(self.prefix)
        #
        # layout.addRow("Prefix:", self.prefix_edit)
        #
        # self.postfix_edit = QLineEdit(self)
        # self.postfix_edit.setText(self.postfix)
        #
        # layout.addRow("Postfix:", self.postfix_edit)
        #
        # self.filename_edit = QLineEdit(self)
        # self.filename_edit.setText(self.filename)
        #
        # layout.addRow("File name:", self.filename_edit)
        #
        # self.parameter_tabs = QTabWidget(self)
        # self.parameter_tabs.setMovable(False)
        # self.parameter_tabs.setTabsClosable(False)
        # for param in self.parameters:
        #     param.init_UI()
        #
        # layout.addRow(self.parameter_tabs)
        #
        # self.spin_box = QSpinBox(self)
        # self.spin_box.setMinimum(1)
        # spin_box_max_val = 1073741824
        # self.spin_box.setMaximum(spin_box_max_val)
        # self.spin_box.setValue(self.repetitions)
        #
        # layout.addRow("Repetitions:", self.spin_box)
        #
        self.scaling_choice = QComboBox(self)
        self.scaling_choice.addItem("weak")
        self.scaling_choice.addItem("strong")

        layout.addRow("Scaling type:", self.scaling_choice)

        self.progress_indicator = QProgressBar(self)
        self.progress_indicator.hide()
        layout.addRow(self.progress_indicator)

        # If the user presses the enter key on any element it activates the
        # first button somehow. Thus, create a fake button, that does nothing
        # To avoid that any entry in the value list activates that OK button.
        # fake_button = QPushButton(self)
        # fake_button.setText("OK")
        # fake_button.move(-1120, -1220)
        # fake_button.hide()

        # self.ok_button = QPushButton(self)
        # self.ok_button.setText("OK")
        # self.ok_button.clicked.connect(self.ok_button.hide)
        # self.ok_button.pressed.connect(self.ok_pressed)
        #
        # layout.addRow(self.ok_button)

        # cancel_button = QPushButton(self)
        # cancel_button.setText("Cancel")
        # cancel_button.pressed.connect(self.close)
        main_layout.addRow(layout)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        main_layout.addRow(self.buttonBox)

        # self.change_param_num()
        self.setLayout(main_layout)

    # def change_param_num(self):
    #     self.num_params = self.num_params_choice.value()
    #     self.parameter_tabs.clear()
    #     for param in self.parameters:
    #         param.hide()
    #     for index in range(0, self.num_params):
    #         self.parameter_tabs.addTab(self.parameters[index],
    #                                    "Parameter " + str(index + 1))
    #         self.parameters[index].show()
    #
    #     self.autoFillOptions()
    #     self.prefix_edit.setText(self.prefix)
    #     self.postfix_edit.setText(self.postfix)
    #     self.filename_edit.setText(self.filename)
    #     self.spin_box.setValue(self.repetitions)
    #     for index in range(0, self.num_params):
    #         self.parameters[index].onNewValues()
    #
    #     self.update()
    #
    # def autoFillOptions(self):
    #
    #     auto_fill_rule = r"^([^.]+)"
    #     for i in range(0, self.num_params):
    #         auto_fill_rule = auto_fill_rule + r"([_\.][^\d]+)(\d+)"
    #     auto_fill_rule = auto_fill_rule + r"[_\.]r(\d+)$"
    #     auto_fill_regex = re.compile(auto_fill_rule)
    #
    #     # meaningful defaults
    #     self.filename = "profile.cubex"
    #     self.postfix = ""
    #     self.prefix = ""
    #     self.repetitions = 1
    #
    #     # get list of existing directories matching the pattern
    #     available_dirs = os.listdir(self.dir_name)
    #     dir_matches = list()
    #     for d in available_dirs:
    #         m = auto_fill_regex.match(d)
    #         if m is not None:
    #             dir_matches.append(m)
    #
    #     # print("matched directoty list with given pattern: ", dir_matches)
    #
    #     if len(dir_matches) == 0:
    #         return
    #
    #     # get prefix from first match
    #     self.prefix = dir_matches[0].group(1)
    #     matching_prefix = [d for d in dir_matches if d.group(1) == self.prefix]
    #
    #     for i in range(0, self.num_params):
    #         # get parameter name from first match
    #         self.parameters[i].name = dir_matches[0].group(2 + i * 2)
    #
    #         # extract all values for parameter p
    #         available_p_values = sorted(
    #             set(int(m.group(3 + i * 2)) for m in matching_prefix))
    #         self.parameters[i].values = ','.join(
    #             str(v) for v in available_p_values)
    #
    #     # get maximum repetition count
    #     max_repetitions = max(int(m.group(2 + self.num_params * 2))
    #                           for m in matching_prefix)
    #     self.repetitions = max_repetitions
    @Slot()
    def reject(self):
        self._cancel_event.set()
        super().reject()

    @Slot()
    def accept(self):

        # self.prefix = self.prefix_edit.text()
        # self.postfix = self.postfix_edit.text()
        # self.filename = self.filename_edit.text()
        # self.repetitions = self.spin_box.value()
        self.scaling_type = self.scaling_choice.currentText()

        with ProgressBar(total=0, gui=True) as pbar:
            self._show_progressbar()
            pbar.display = partial(self._display_progress, pbar)
            pbar.sp = None

            # read the cube files
            try:
                self.experiment = read_cube_file(self.dir_name, self.scaling_type, pbar)
            except Exception as err:
                self.close()
                raise err

            if not self.experiment:
                QMessageBox.critical(self,
                                     "Error",
                                     "Could not read Cube Files, may be corrupt!",
                                     QMessageBox.Ok,
                                     QMessageBox.Ok)
                self.close()
                return
            self.valid = True

            super().accept()

    def _show_progressbar(self):
        self.controls_layout.setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.progress_indicator.show()

    def _display_progress(self, pbar: ProgressBar, msg=None, pos=None):
        if self._cancel_event.is_set():
            raise CancelProcessError()
        self.progress_indicator.setMaximum(pbar.total)
        self.progress_indicator.setValue(pbar.n)
        QApplication.processEvents()
