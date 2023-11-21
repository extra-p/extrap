# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from PySide6.QtCore import QModelIndex, QSize, QRectF
from PySide6.QtGui import QTextDocument, QTextOption, QPalette, QAbstractTextDocumentLayout, \
    QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QApplication, QStyle


class RichTextDelegate(QStyledItemDelegate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderer = QTextDocument()
        self.margin = 0
        self.font_scale = 0
        self.ellipsis = "..."

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex):
        self.initStyleOption(option, index)
        if not option.text:
            return super().paint(painter, option, index)

        if self.font_scale != 0:
            font = QFont(option.font)
            font.setPointSizeF(option.font.pointSizeF() * self.font_scale)
            option.font = font

        textOption = QTextOption(option.displayAlignment)
        if QStyleOptionViewItem.WrapText in option.features:
            textOption.setWrapMode(QTextOption.WordWrap)
        else:
            textOption.setWrapMode(QTextOption.NoWrap)

        textOption.setTextDirection(option.direction)

        renderer = self.renderer
        painter.save()
        renderer.setTextWidth(option.rect.width())
        renderer.setHtml(option.text)
        renderer.setDefaultFont(option.font)
        renderer.setDefaultTextOption(textOption)
        renderer.setDocumentMargin(self.margin)
        renderer.adjustSize()

        option.text = ''
        painter.setRenderHint(QPainter.TextAntialiasing)

        style = QApplication.style()
        if option.widget:
            style = option.widget.style()

        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, option, painter)

        textRect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, option)

        painter.translate(textRect.topLeft())
        context = QAbstractTextDocumentLayout.PaintContext()
        if QStyle.State_Selected in option.state:
            context.palette.setColor(QPalette.Text, option.palette.color(QPalette.HighlightedText))
        else:
            context.palette.setColor(QPalette.Text, option.palette.color(QPalette.Text))

        if renderer.size().width() > option.rect.width():
            # clip and draw ellipsis
            ellipsis = self.ellipsis
            metric = QFontMetrics(option.font)
            ellipsis_width = metric.horizontalAdvance(ellipsis)
            ellipsis_margin = 1

            painter.save()
            if QStyle.State_Selected in option.state:
                painter.setPen(option.palette.color(QPalette.HighlightedText))
            else:
                painter.setPen(option.palette.color(QPalette.Text))

            painter.drawText(QRectF(textRect.width() - ellipsis_width, 0, ellipsis_width, textRect.height()), ellipsis)
            painter.restore()

            context.clip = QRectF(0, 0, textRect.width() - ellipsis_width - ellipsis_margin, textRect.height())
            painter.setClipRect(context.clip)

        renderer.documentLayout().draw(painter, context)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        self.initStyleOption(option, index)
        if not option.text:
            return super().sizeHint(option, index)

        if self.font_scale != 0:
            font = QFont(option.font)
            font.setPointSizeF(option.font.pointSizeF() * self.font_scale)
            option.font = font

        renderer = self.renderer
        renderer.setHtml(option.text)
        renderer.setTextWidth(option.rect.width())
        renderer.setDocumentMargin(self.margin)
        renderer.setDefaultFont(option.font)
        return QSize(renderer.idealWidth(), renderer.size().height())
