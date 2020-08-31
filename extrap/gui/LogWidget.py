import logging

import PySide2
from PySide2.QtGui import QPaintEvent
from PySide2.QtWidgets import QWidget, QGridLayout, QTextEdit


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
        layout.setMargin(0)
        self.setLayout(layout)
        layout.addWidget(self.log_box)
        self.log_box.setAcceptRichText(True)
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_box.setStyleSheet("QTextEdit{"
                                   "background-color:white;"
                                   "}")

    def showEvent(self, event: PySide2.QtGui.QShowEvent):
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
