"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


import os
import re

from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtCore import *   # @UnusedWildImport
from PySide2.QtWidgets import *   # @UnusedWildImport
from fileio.cube_file_reader import read_cube_file


class ParameterWidget(QWidget):

    def __init__(self, parent):
        super(ParameterWidget, self).__init__(parent)
        self.name = "Parameter"
        self.values = "1"

    def init_UI(self):
        self.name_edit = QLineEdit(self)
        self.name_edit.setText(self.name)
        self.name_edit.setFixedWidth(190)
        self.name_edit.move(120, 10)

        self.values_edit = QLineEdit(self)
        self.values_edit.setText(self.values)
        self.values_edit.setFixedWidth(190)
        self.values_edit.move(120, 40)

        name_label = QLabel(self)
        name_label.setText("Parameter name:")
        name_label.move(10, 15)

        value_label = QLabel(self)
        value_label.setText("Values:")
        value_label.move(10, 45)

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

        for _ in range(0, self.max_params):
            self.parameters.append(ParameterWidget(self))

        self.init_UI()

    def init_UI(self):
        self.setWindowTitle("Import settings")
        self.setFixedWidth(320)

        self.num_params_choice = QSpinBox(self)
        self.num_params_choice.setMinimum(1)
        self.num_params_choice.setMaximum(self.max_params)
        self.num_params_choice.setValue(self.num_params)
        self.num_params_choice.move(160, 10)
        self.num_params_choice.valueChanged.connect(self.change_param_num)

        self.prefix_edit = QLineEdit(self)
        self.prefix_edit.setText(self.prefix)
        self.prefix_edit.setFixedWidth(190)
        self.prefix_edit.move(120, 40)

        self.postfix_edit = QLineEdit(self)
        self.postfix_edit.setText(self.postfix)
        self.postfix_edit.setFixedWidth(190)
        self.postfix_edit.move(120, 70)

        self.filename_edit = QLineEdit(self)
        self.filename_edit.setText(self.filename)
        self.filename_edit.setFixedWidth(190)
        self.filename_edit.move(120, 100)

        self.parameter_tabs = QTabWidget(self)
        self.parameter_tabs.setMovable(False)
        self.parameter_tabs.setTabsClosable(False)
        self.parameter_tabs.move(0, 130)
        self.parameter_tabs.resize(320, 90)
        for param in self.parameters:
            param.init_UI()

        self.spin_box = QSpinBox(self)
        self.spin_box.setMinimum(1)
        spin_box_max_val = 1073741824
        self.spin_box.setMaximum(spin_box_max_val)
        self.spin_box.setValue(self.repetitions)
        self.spin_box.move(120, 220)

        self.scaling_choice = QComboBox(self)
        self.scaling_choice.move(120, 250)
        self.scaling_choice.setFixedWidth(110)
        self.scaling_choice.addItem("weak")
        self.scaling_choice.addItem("strong")

        self.progress_indicator = QProgressBar(self)
        self.progress_indicator.move(165, 280)
        self.progress_indicator.setFixedWidth(110)

        num_params_label = QLabel(self)
        num_params_label.setText("Number of Parameters:")
        num_params_label.move(10, 15)

        prefix_label = QLabel(self)
        prefix_label.setText("Prefix:")
        prefix_label.move(10, 45)

        prefix_label = QLabel(self)
        prefix_label.setText("Postfix:")
        prefix_label.move(10, 75)

        prefix_label = QLabel(self)
        prefix_label.setText("File name:")
        prefix_label.move(10, 105)

        prefix_label = QLabel(self)
        prefix_label.setText("Repetitions:")
        prefix_label.move(10, 225)

        prefix_label = QLabel(self)
        prefix_label.setText("Scaling type:")
        prefix_label.move(10, 255)

        # If the user presses the enter key on any element it activates the
        # first button somehow. Thus, create a fake button, that does nothing
        # To avoid that any entry in the value list activates that OK button.
        fake_button = QPushButton(self)
        fake_button.setText("OK")
        fake_button.move(-1120, -1220)
        fake_button.hide()

        self.ok_button = QPushButton(self)
        self.ok_button.setText("OK")
        self.ok_button.move(165, 280)
        self.ok_button.setFixedWidth(110)
        self.ok_button.clicked.connect(self.ok_button.hide)
        self.ok_button.pressed.connect(self.ok_pressed)

        cancel_button = QPushButton(self)
        cancel_button.setText("Cancel")
        cancel_button.move(40, 280)
        cancel_button.setFixedWidth(100)
        cancel_button.pressed.connect(self.close)

        self.change_param_num()

    def change_param_num(self):
        self.num_params = self.num_params_choice.value()
        self.parameter_tabs.clear()
        for param in self.parameters:
            param.hide()
        for index in range(0, self.num_params):
            self.parameter_tabs.addTab(self.parameters[index],
                                       "Parameter " + str(index + 1))
            self.parameters[index].show()

        self.autoFillOptions()
        self.prefix_edit.setText(self.prefix)
        self.postfix_edit.setText(self.postfix)
        self.filename_edit.setText(self.filename)
        self.spin_box.setValue(self.repetitions)
        for index in range(0, self.num_params):
            self.parameters[index].onNewValues()

        self.update()

    def autoFillOptions(self):

        auto_fill_rule = r"^([^.]+)"
        for i in range(0, self.num_params):
            auto_fill_rule = auto_fill_rule + r"([_\.][^\d]+)(\d+)"
        auto_fill_rule = auto_fill_rule + r"[_\.]r(\d+)$"
        auto_fill_regex = re.compile(auto_fill_rule)

        # meaningful defaults
        self.filename = "profile.cubex"
        self.postfix = ""
        self.prefix = ""
        self.repetitions = 1

        # get list of existing directories matching the pattern
        available_dirs = os.listdir(self.dir_name)
        dir_matches = list()
        for d in available_dirs:
            m = auto_fill_regex.match(d)
            if m is not None:
                dir_matches.append(m)

        #print("matched directoty list with given pattern: ", dir_matches)

        if (len(dir_matches) == 0):
            return

        # get prefix from first match
        self.prefix = dir_matches[0].group(1)
        matching_prefix = [d for d in dir_matches if d.group(1) == self.prefix]

        for i in range(0, self.num_params):
            # get parameter name from first match
            self.parameters[i].name = dir_matches[0].group(2+i*2)

            # extract all values for parameter p
            available_p_values = sorted(
                set(int(m.group(3+i*2)) for m in matching_prefix))
            self.parameters[i].values = ','.join(
                str(v) for v in available_p_values)

        # get maximum repetition count
        max_repetitions = max(int(m.group(2+self.num_params*2))
                              for m in matching_prefix)
        self.repetitions = max_repetitions

    def ok_pressed(self):
        self.prefix = self.prefix_edit.text()
        self.postfix = self.postfix_edit.text()
        self.filename = self.filename_edit.text()
        self.repetitions = self.spin_box.value()
        self.scaling_type = self.scaling_choice.currentIndex()

        # read the cube files
        self.experiment = read_cube_file(self.dir_name, self.scaling_type)

        # TODO: create callback method to update the progress bar

        if not self.experiment:
            QMessageBox.warning(self,
                                "Error",
                                "Could not read Cube Files, may be corrupt!",
                                QMessageBox.Ok,
                                QMessageBox.Ok)
            self.close()
            return
        self.valid = True

        self.close()

    def increment_progress(self):
        currentValue = self.progress_indicator.value()
        value = currentValue+1
        self.progress_indicator.setValue(value)
