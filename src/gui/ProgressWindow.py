"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""
from PySide2.QtCore import Qt, QCoreApplication
from PySide2.QtWidgets import QProgressDialog, QLabel

from util.progress_bar import ProgressBar


class ProgressWindow(ProgressBar):
    def __init__(self, parent, title, **kwargs):
        super().__init__(total=0, desc=title, **kwargs, gui=True)
        self.sp = None
        self.dialog = QProgressDialog(parent)
        self.dialog.setModal(True)
        self.dialog.setCancelButton(None)
        self.dialog.setMinimumDuration(500)
        self.dialog.setWindowTitle(self.desc)
        self.dialog.setAutoClose(False)
        self.dialog.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.dialog.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        label = QLabel()
        label.setTextFormat(Qt.PlainText)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.dialog.setLabel(label)
        # self.dialog.show()

    def close(self):
        super().close()
        self.dialog.close()

    def clear(self, nolock=False):
        super().clear(nolock)

    def display(self, msg=None, pos=None):
        format_dict = self.format_dict
        elapsed_str = self.format_interval(format_dict['elapsed'])

        rate = format_dict['rate']
        remaining = (self.total - self.n) / rate if rate and self.total else 0
        remaining_str = self.format_interval(remaining) if rate else '??:??'

        time_str = f'Time remaining:\t{remaining_str}\nTime elapsed:\t{elapsed_str}'

        self.dialog.setMaximum(self.total)
        if self.postfix:
            self.dialog.setLabelText(time_str + '\n\n' + self.postfix)
        else:
            self.dialog.setLabelText(time_str)
        self.dialog.setValue(self.n)
        QCoreApplication.processEvents()
