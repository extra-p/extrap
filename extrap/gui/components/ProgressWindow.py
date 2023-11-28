# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
from threading import Event

from PySide6.QtCore import Qt, QCoreApplication, Slot
from PySide6.QtWidgets import QProgressDialog, QLabel, QWidget, QVBoxLayout, QProgressBar, QApplication

from extrap.util.exceptions import CancelProcessError
from extrap.util.progress_bar import ProgressBar


def format_progress_time_for_gui(pbar: ProgressBar):
    format_dict = pbar.format_dict
    elapsed_str = pbar.format_interval(format_dict['elapsed'])
    rate = format_dict['rate']
    remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
    remaining_str = pbar.format_interval(remaining) if rate else '??:??'
    time_str = f'Time remaining:\t{remaining_str}\nTime elapsed:\t{elapsed_str}'
    return time_str


class ProgressWindow(ProgressBar):
    def __init__(self, parent, title, **kwargs):
        super().__init__(total=0, desc=title, **kwargs, gui=True)
        self.sp = None
        self.dialog = QProgressDialog(parent)
        self.dialog.setModal(True)
        self.dialog.setMinimumDuration(500)
        self.dialog.setWindowTitle(self.desc)
        self.dialog.setAutoClose(False)
        self.dialog.setWindowModality(Qt.WindowModal)
        self.dialog.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.dialog.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        label = QLabel()
        label.setTextFormat(Qt.PlainText)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.dialog.setLabel(label)
        self._cancel_event = Event()
        self._internal_cancel_event = Event()
        self.dialog.canceled.connect(self.user_cancel)
        # self.dialog.show()

    def close(self):
        try:
            super().close()
        except AttributeError:
            pass
        try:
            if not self._cancel_event.is_set():
                self.dialog.cancel()
                self.dialog.reject()
        except RuntimeError:
            pass

    @Slot()
    def user_cancel(self):
        self._cancel_event.set()

    def clear(self, nolock=False):
        super().clear(nolock)

    def update(self, n=1):
        if self._cancel_event.is_set():
            raise CancelProcessError()
        super(ProgressWindow, self).update(n)

    def display(self, msg=None, pos=None):
        time_str = format_progress_time_for_gui(self)

        if not self._cancel_event.is_set():
            self.dialog.setMaximum(self.total)
            if self.postfix:
                self.dialog.setLabelText(time_str + '\n\n' + self.postfix)
            else:
                self.dialog.setLabelText(time_str)
            self.dialog.setValue(self.n)
        QCoreApplication.processEvents()


class ProgressWidget(QWidget):
    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super().__init__(parent, flags)
        self._cancel_event = Event()
        self.setMaximumHeight(0)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.progress_label = QLabel('Time remaining:\t??:??\nTime elapsed:\t00:00')
        layout.addWidget(self.progress_label)

        self.progress_indicator = QProgressBar(self)
        layout.addWidget(self.progress_indicator)

        self.progress_bar_class = self._make_progress_bar_class()

    def _make_progress_bar_class(self):
        widget = self

        class _ProgressBar(ProgressBar):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.sp = None

            def display(self: ProgressBar, msg=None, pos=None):
                if widget._cancel_event.is_set():
                    raise CancelProcessError()
                time_str = format_progress_time_for_gui(self)

                if self.postfix:
                    widget.progress_label.setText(time_str + '\n' + self.postfix)
                else:
                    widget.progress_label.setText(time_str)
                widget.progress_indicator.setMaximum(self.total)
                widget.progress_indicator.setValue(self.n)
                QApplication.processEvents()

            def __enter__(self):
                widget.setMaximumHeight(10000)
                QApplication.processEvents()
                return super().__enter__()

        return _ProgressBar

    def get_progress_bar(self) -> ProgressBar:
        return self.progress_bar_class(total=0, gui=True)

    def cancel(self):
        self._cancel_event.set()
