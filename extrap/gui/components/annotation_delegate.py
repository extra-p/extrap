# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from PySide6.QtCore import QXmlStreamReader, QModelIndex, QMargins, QRect, QSize
from PySide6.QtGui import Qt
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem

from extrap.entities.annotations import AnnotationIconSVG


class AnnotationDelegate(QStyledItemDelegate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderer_cache = {}
        self.margins = QMargins(1, 1, 1, 1)
        self.spacing = 2

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex):
        annotation_icons: list[AnnotationIconSVG] = index.data()

        if not annotation_icons:
            return super().paint(painter, option, index)

        super().paint(painter, option, QModelIndex())  # Draw Background

        # highlightBrush: QBrush = option.palette.highlight()
        # if option.state & QStyle.State_Selected:
        #     painter.fillRect(option.rect, highlightBrush)

        rect_with_margin = option.rect.marginsRemoved(self.margins)
        x_pos = rect_with_margin.x()
        for icon in annotation_icons:
            if not icon:
                continue
            renderer = self.renderer_cache.get(icon)
            if renderer is None:
                renderer = QSvgRenderer(QXmlStreamReader(icon))
                if not renderer.isValid():
                    raise RuntimeError(f"Could not load SVG: {icon}")
                renderer.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
                self.renderer_cache[icon] = renderer

            rect: QRect = option.rect.marginsRemoved(self.margins)
            rect.setX(x_pos)
            svg_size = renderer.defaultSize()
            width = svg_size.width() * (rect_with_margin.height() / svg_size.height())
            rect.setWidth(width)
            x_pos = rect.right() + self.spacing
            if rect.right() <= rect_with_margin.right() + width - self.spacing:
                rect.setRight(min(rect.right(), rect_with_margin.right()))
                renderer.render(painter, rect)

        # if option.state & QStyle.State_HasFocus:
        #     focus_options = QStyleOptionFocusRect()
        #     focus_options.rect = option.rect
        #     focus_options.state = option.state | QStyle.State_KeyboardFocusChange | QStyle.State_Item
        #     focus_options.backgroundColor = highlightBrush.color()
        #     QApplication.style().drawPrimitive(QStyle.PE_FrameFocusRect, focus_options, painter)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        annotation_icons: list[AnnotationIconSVG] = index.data()

        if not annotation_icons:
            return super().sizeHint(option, index)

        width = self.margins.left()
        for icon in annotation_icons:
            if not icon:
                continue
            renderer = self.renderer_cache.get(icon)
            if renderer is None:
                renderer = QSvgRenderer(QXmlStreamReader(icon))
                if not renderer.isValid():
                    raise RuntimeError(f"Could not load SVG: {icon}")
                renderer.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
                self.renderer_cache[icon] = renderer

            svg_size = renderer.defaultSize()
            width += svg_size.width() * (option.decorationSize.height() / svg_size.height()) + self.spacing
        width -= self.spacing

        return QSize(width, option.decorationSize.height())
