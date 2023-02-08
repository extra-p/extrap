# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging

import PySide6
from PySide6.QtGui import QPaintEvent
from PySide6.QtWidgets import QWidget, QGridLayout, QTextEdit


class LogWidget(QWidget):
    def __init__(self, parent):
        super(LogWidget, self).__init__(parent)
        self.log_box = QTextEdit(self)
        self.initUI()
        handler = logging.StreamHandler(self)
        handler.terminator = ' '
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logging.getLogger().addHandler(handler)

    def initUI(self):
        layout = QGridLayout(self)
        # layout.setContentsMargins(10, 5, 10, 5)
        #layout.setMargin(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self.log_box)
        self.log_box.setAcceptRichText(True)
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_box.setStyleSheet("QTextEdit{"
                                   "background-color:white;"
                                   "}")

    def showEvent(self, event: PySide6.QtGui.QShowEvent):
        self._reset_scrollbars()
        super().showEvent(event)

    def write(self, text):
        if text.startswith('WARNING:'):
            text = '<span style="color:#DF6F0F">' + text + '</span>'
        elif text.startswith('ERROR:'):
            text = '<span style="color:#C50030">' + text + '</span>'
        self.log_box.append(text)
        self._reset_scrollbars()

    def _reset_scrollbars(self):
        scroll_bar = self.log_box.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        scroll_bar = self.log_box.horizontalScrollBar()
        scroll_bar.setValue(scroll_bar.minimum())
