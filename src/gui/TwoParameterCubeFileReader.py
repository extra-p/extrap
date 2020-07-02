"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""

# TODO: is this class even needed???


import os
import re

from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport
from PySide2.QtCore import *  # @UnusedWildImport

# TODO: fix this


def convert_string_to_numberlist(str):
    words = str.split(",")
    numbers = EXTRAP.StdVectorInt()
    for word in words:
        numbers.push_back(int(float(word)))
    return numbers

# TODO: fix this


def remove_digits(str):

    return "".join([i for i in str if not i.isdigit()])


class TwoParameterCubeFileReader(QDialog):

    def __init__(self, parent, dirName):
        super(TwoParameterCubeFileReader, self).__init__(parent)

        # regular expression as per the two parameter cube file format
        self.autoFillRegex = re.compile(
            r"^([^.]+)[_\.]([^\d]+)(\d+)[_\.]([^\d]+)(\d+)[_\.]r(\d+)$")

        self.valid = False
        self.dir_name = dirName
        self.autoFillOptions()
        self.init_UI()

    def init_UI(self):
        self.setWindowTitle("Import settings")
        self.setFixedWidth(320)

        self.prefix_edit = QLineEdit(self)
        self.prefix_edit.setText(self.prefix)
        self.prefix_edit.setFixedWidth(190)
        self.prefix_edit.move(120, 10)

        self.postfix_edit = QLineEdit(self)
        self.postfix_edit.setText(self.postfix)
        self.postfix_edit.setFixedWidth(190)
        self.postfix_edit.move(120, 40)

        self.filename_edit = QLineEdit(self)
        self.filename_edit.setText(self.filename)
        self.filename_edit.setFixedWidth(190)
        self.filename_edit.move(120, 70)

        self.parameter_first_edit = QLineEdit(self)
        self.parameter_first_edit.setText(self.first_parameter)
        self.parameter_first_edit.setFixedWidth(190)
        self.parameter_first_edit.move(120, 100)

        self.values_first_edit = QLineEdit(self)
        self.values_first_edit.setText(self.first_parameter_values)
        self.values_first_edit.setFixedWidth(190)
        self.values_first_edit.move(120, 130)

        self.parameter_second_edit = QLineEdit(self)
        self.parameter_second_edit.setText(self.second_parameter)
        self.parameter_second_edit.setFixedWidth(190)
        self.parameter_second_edit.move(120, 160)

        self.values_second_edit = QLineEdit(self)
        self.values_second_edit.setText(self.second_parameter_values)
        self.values_second_edit.setFixedWidth(190)
        self.values_second_edit.move(120, 190)

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

        prefix_label = QLabel(self)
        prefix_label.setText("Prefix:")
        prefix_label.move(10, 15)

        prefix_label = QLabel(self)
        prefix_label.setText("Postfix:")
        prefix_label.move(10, 45)

        prefix_label = QLabel(self)
        prefix_label.setText("File name:")
        prefix_label.move(10, 75)

        prefix_label = QLabel(self)
        prefix_label.setText("Parameter:")
        prefix_label.move(10, 105)

        prefix_label = QLabel(self)
        prefix_label.setText("Values:")
        prefix_label.move(10, 135)

        prefix_label = QLabel(self)
        prefix_label.setText("Parameter:")
        prefix_label.move(10, 165)

        prefix_label = QLabel(self)
        prefix_label.setText("Values:")
        prefix_label.move(10, 195)

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

    def autoFillOptions2(self):

        self.autoFillRegex = re.compile(
            r"^([^.]+)[_\.]([^\d]+)(\d+)[_\.]([^\d]+)(\d+)[_\.]r(\d+)$")

        # these are fixed
        self.filename = "profile.cubex"
        self.first_parameter = "p"
        self.second_parameter = "size"
        self.postfix = ""

        # meaningful defaults
        self.prefix = "unknown"
        self.first_parameter_values = "32,64,128"
        self.second_parameter_values = "5, 10, 15"
        self.repetitions = 1

        # get list of existing directories matching the pattern
        available_dirs = os.listdir(self.dir_name)
        dir_matches = list()
        for d in available_dirs:
            m = self.autoFillRegex.match(d)
            if m is not None:
                dir_matches.append(m)

        if (len(dir_matches) == 0):
            return

        # get prefix from first match
        self.prefix = dir_matches[0].group(1)

        matching_prefix = [d for d in dir_matches if d.group(1) == self.prefix]

        # get first parameter name from first match
        self.first_parameter = dir_matches[0].group(2)

        # extract all values for first parameter
        available_p1_values = sorted(
            set(int(m.group(3)) for m in matching_prefix))
        self.first_parameter_values = ','.join(
            str(v) for v in available_p1_values)

        # get second parameter name from first match
        self.second_parameter = dir_matches[0].group(4)

        # extract all values for second paramter
        available_p2_values = sorted(
            set(int(m.group(5)) for m in matching_prefix))
        self.second_parameter_values = ','.join(
            str(v) for v in available_p2_values)

        # get maximum repetition count
        max_repetitions = max(int(m.group(6)) for m in matching_prefix)
        self.repetitions = max_repetitions

    def ok_pressed(self):
        self.prefix = self.prefix_edit.text()
        self.postfix = self.postfix_edit.text()
        self.filename = self.filename_edit.text()

        # We have added an additional "-" in the parameter name in order to match current
        # Cube File Reader implemented in C++ else it produces a wrong file name
        # considering the file name format for single and multi parameter cube files are different
        self.first_parameter = "_" + self.parameter_first_edit.text()
        self.first_parameter_values = self.values_first_edit.text()
        self.second_parameter = "_" + self.parameter_second_edit.text()
        self.second_parameter_values = self.values_second_edit.text()
        self.repetitions = self.spin_box.value()
        self.scaling_type = self.scaling_choice.currentIndex()
        reader = EXTRAP.CubeFileReader()

        reader.prepareCubeFileReader(self.scaling_type + 1,
                                     self.dir_name,
                                     self.prefix,
                                     self.postfix,
                                     self.filename,
                                     self.repetitions)

        reader.addParameter(EXTRAP.Parameter(self.first_parameter),
                            self.first_parameter,
                            convert_string_to_numberlist(self.first_parameter_values))

        reader.addParameter(EXTRAP.Parameter(self.second_parameter),
                            self.second_parameter,
                            convert_string_to_numberlist(self.second_parameter_values))

        reader.set_pymethod(self.increment_progress)
        file_list = reader.getFileNames(2)

        for f in file_list:
            if not os.path.exists(f):
                QMessageBox.warning(self,
                                    "Error",
                                    "Can not open " + f,
                                    QMessageBox.Ok,
                                    QMessageBox.Ok)
                return
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self.ok_button.hide()
        self.progress_indicator.setMaximum(len(file_list))
        QApplication.instance().processEvents()

        self.experiment = reader.readCubeFiles(2)

        QApplication.restoreOverrideCursor()
        QApplication.instance().processEvents()
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
