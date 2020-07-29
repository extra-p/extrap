"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""
from PySide2.QtCore import QCoreApplication
from PySide2.QtWidgets import QProgressDialog
import threading


class ProgressWindow(QProgressDialog):
    def __init__(self, parent, title, label):
        super().__init__(parent)
        self.setRange(0, 1000)
        self.setCancelButton(None)
        self.setWindowTitle(title)
        self.setLabelText(label)
        self.show()

    def progress_event(self, progress):
        if progress is None:
            self.close()
        else:
            self.setValue(int(progress * 1000))
        QCoreApplication.processEvents()

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.close()
