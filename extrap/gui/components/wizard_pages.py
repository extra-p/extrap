from functools import partial

import PySide2
from PySide2.QtCore import QEvent, QTimer
from PySide2.QtWidgets import QWizardPage, QVBoxLayout, QScrollArea, QProgressBar, QLabel, QApplication

from extrap.util.exceptions import CancelProcessError
from extrap.util.progress_bar import ProgressBar


class ScrollAreaPage(QWizardPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setLayout(QVBoxLayout(self))
        scroll_widget = QScrollArea(self)
        scroll_widget.setWidgetResizable(True)
        scroll_widget.setStyleSheet("QScrollArea{border:none}")
        scroll_widget.setViewportMargins(0, 0, 0, 0)
        scroll_widget.setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(scroll_widget)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.scroll_layout = QVBoxLayout(self)
        scroll_widget.setLayout(self.scroll_layout)


class ProgressPage(QWizardPage):
    once = True

    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Loading')
        self.setLayout(QVBoxLayout(self))
        self.progress_indicator = QProgressBar(self)
        self.layout().addWidget(self.progress_indicator)
        self.label = QLabel(self)
        self.layout().addWidget(self.label)
        self._is_complete = False
        self.next_id = None

    def nextId(self) -> int:
        if self.next_id is not None:
            return self.next_id
        return super().nextId()

    def initializePage(self) -> None:
        self.progress_bar = ProgressBar(total=0, gui=True)
        self.progress_bar.display = partial(self._display_progress, self.progress_bar)
        self.progress_bar.sp = None
        self.once = False

    def event(self, event: PySide2.QtCore.QEvent) -> bool:
        value = super(ProgressPage, self).event(event)
        if not self.once and event.type() == QEvent.Paint:
            self.once = True
            QTimer.singleShot(0, self.once_after_shown)
        return value

    def once_after_shown(self):
        target_progress = self.progress_bar.create_target(1)
        try:
            self.do_process(self.progress_bar)
            self.progress_bar.reach_target(target_progress)
            self.progress_bar.update(self.progress_bar.total - self.progress_bar.n)
            self.progress_bar.refresh()
            self._is_complete = True
            self.completeChanged.emit()
        except CancelProcessError as e:
            raise e
        finally:
            self.progress_bar.close()

    def do_process(self, pbar):
        pass

    def isComplete(self) -> bool:
        return self._is_complete

    def _display_progress(self, pbar: ProgressBar, msg=None, pos=None):
        if self.wizard().is_cancelled.is_set():
            raise CancelProcessError()

        elapsed_str = pbar.format_interval(pbar.format_dict['elapsed'])

        rate = pbar.format_dict['rate']
        remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
        remaining_str = pbar.format_interval(remaining) if rate else '??:??'

        time_str = f'Time remaining:\t{remaining_str}\nTime elapsed:\t{elapsed_str}'

        self.progress_indicator.setMaximum(pbar.total)
        if pbar.postfix:
            self.label.setText(self.postfix + '\n\n' + time_str)
        else:
            self.label.setText(time_str)
        self.progress_indicator.setValue(pbar.n)
        QApplication.processEvents()
