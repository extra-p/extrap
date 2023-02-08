# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import sys

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QToolButton, QSizePolicy, QWidget, QGroupBox, QVBoxLayout, QApplication


class ExpanderWidget(QGroupBox):
    def __init__(self, parent, title, content=None):
        super(ExpanderWidget, self).__init__(parent)
        self._title = title
        self._toggle = QToolButton(self)
        self._content = QWidget() if content is None else content
        self.initUI()

    def setTitle(self, title: str):
        self._title = title

    def title(self) -> str:
        return self._title

    def setContent(self, widget: QWidget):
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        widget.setMinimumHeight(0)
        self.layout().replaceWidget(self._content, widget, Qt.FindChildOption.FindDirectChildrenOnly)
        self._content.deleteLater()
        self._content = widget
        self._toggle_closed(self._toggle.isChecked())

    def content(self) -> QWidget:
        return self._content

    def setEnabled(self, arg: bool):
        if not arg:
            self.toggle(True)
        super().setEnabled(arg)

    def initUI(self):
        self.setStyleSheet('ExpanderWidget{ '
                           '    padding:2px -1px -1px 0px; margin-left:-1px; '
                           f'{" margin-top: -19px; padding-top:19px; " if sys.platform.startswith("darwin") else ""}'
                           '}'
                           'ExpanderWidget::title{ '
                           '    color: transparent; '
                           #    '    height: 0px; '
                           '}'
                           'ExpanderWidget>#expanderToggle{ '
                           '    background: transparent;'
                           '    border: 1px solid transparent;'
                           '    border-bottom: 1px solid #B0B0B0;'
                           '    border-radius:0;'
                           f'   font-size: {QApplication.font().pointSizeF()}pt; '
                           '}'
                           'ExpanderWidget>#expanderToggle:hover,'
                           'ExpanderWidget>#expanderToggle:checked:hover{ '
                           '    background: qlineargradient( x1:0 y1:-0.1, x2:0 y2:1, '
                           '        stop:0 white, stop:1 palette(button));'
                           '    border: 1px solid #B0B0B0; border-radius:2px;'
                           '}'
                           'ExpanderWidget>#expanderToggle:checked,'
                           'ExpanderWidget>#expanderToggle:checked:pressed,'
                           'ExpanderWidget>#expanderToggle:pressed{'
                           '    background: palette(background);'
                           '    border: 1px solid #B0B0B0; border-radius:2px;'
                           '}'
                           'ExpanderWidget:disabled{'
                           '    border: 1px solid transparent; border-radius:2px;'
                           '}'
                           'ExpanderWidget>#expanderToggle:focus{'
                           '    border: 1px solid palette(highlight); border-radius:2px;'
                           '}'
                           )

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setObjectName('expanderToggle')
        self._toggle.setText(self._title)
        self._toggle.setAutoRaise(True)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle.setMinimumHeight(self._toggle.height())
        self._toggle.setArrowType(Qt.ArrowType.DownArrow)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.clicked.connect(self._toggle_closed)

        self._content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._content.setMinimumHeight(0)
        self._content.setMaximumHeight(0)
        self._content

        layout.addWidget(self._toggle)
        layout.addWidget(self._content)

    @Slot(bool)
    def _toggle_closed(self, closed: bool):
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if closed else Qt.ArrowType.UpArrow)
        self._content.setMaximumHeight(0 if closed else 16777215)

    def toggle(self, closed: bool):
        self._toggle.setChecked(closed)
