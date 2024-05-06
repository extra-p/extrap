# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from PySide6.QtCore import QPoint, QRect, QPropertyAnimation, QEasingCurve, Slot, Property, QSize
from PySide6.QtGui import QPaintEvent, QPainter, QMouseEvent
from PySide6.QtWidgets import QCheckBox, QStyle, QStyleOptionProgressBar, QStyleOptionButton


class SwitchWidget(QCheckBox):

    def __init__(self, parent=None):
        self._drag_x = None
        self._switch_position_value = 0
        super().__init__(parent)
        self.animation = QPropertyAnimation(self, b"_switch_position", self)
        self.animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.animation.setDuration(100)

        self.stateChanged.connect(self._on_state_change)

    def sizeHint(self):
        return QSize(40, 22)

    def hitButton(self, pos: QPoint):
        return self._my_contents_rect().contains(pos)

    def paintEvent(self, arg__1: QPaintEvent) -> None:
        cont_rect = self._my_contents_rect()

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        background_options = QStyleOptionProgressBar()
        background_options.textVisible = False
        background_options.rect = cont_rect
        background_options.minimum = 0
        background_options.maximum = 1000
        background_options.progress = int(self._switch_position_value * background_options.maximum)

        self.style().drawControl(QStyle.ControlElement.CE_ProgressBar, background_options, p, self)
        handle_width = cont_rect.width() / 2
        xPos = cont_rect.x() + cont_rect.width() / 2 * self._switch_position_value + 2

        handle_options = QStyleOptionButton()
        self.initStyleOption(handle_options)
        handle_options.rect = QRect(int(xPos), cont_rect.y() + 2, handle_width - 2, cont_rect.height() - 4)
        handle_options.state = handle_options.state & ~QStyle.StateFlag.State_On
        self.style().drawControl(QStyle.ControlElement.CE_PushButtonBevel, handle_options, p, self)

        p.end()

    def _my_contents_rect(self):
        cont_rect = self.contentsRect()
        cont_rect.setWidth(min(cont_rect.width(), int(cont_rect.height() * 2.5)))
        return cont_rect

    @Slot(int)
    def _on_state_change(self, value):
        self.animation.stop()
        if value:
            self.animation.setEndValue(1)
        else:
            self.animation.setEndValue(0)
        self.animation.start()

    @Property(float)
    def _switch_position(self):
        return self._switch_position_value

    @_switch_position.setter
    def _switch_position(self, pos):
        self._switch_position_value = pos
        self.update()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        super().mousePressEvent(e)
        self._drag_x = e.x()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        super().mouseMoveEvent(e)
        if self._drag_x is None:
            return
        movement = (e.x() - self._drag_x)
        min_movement = self._my_contents_rect().width() / 4
        if movement > min_movement and not self.isChecked():
            self.setChecked(True)
        elif movement < -min_movement and self.isChecked():
            self.setChecked(False)
        # self._switch_position = min(max(0, movement), 1)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        min_movement = self._my_contents_rect().width() / 4
        movement = abs(e.x() - self._drag_x)
        if movement < min_movement:
            super().mouseReleaseEvent(e)
        else:
            self.setDown(False)
        self._drag_x = None
