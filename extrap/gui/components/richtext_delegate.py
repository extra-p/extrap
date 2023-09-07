# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from PySide6.QtCore import QModelIndex, QSize
from PySide6.QtGui import QTextDocument, QTextOption, QTextCursor, QFontMetrics, QPalette, QAbstractTextDocumentLayout, \
    QFont
from PySide6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QApplication, QStyle


class RichTextDelegate(QStyledItemDelegate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderer = QTextDocument()
        self.margin = 0
        self.font_scale = 0

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
        renderer.setTextWidth(option.rect.width())
        renderer.adjustSize()

        if renderer.size().width() > option.rect.width():
            # add ellipsis
            cursor = QTextCursor(renderer)
            cursor.movePosition(QTextCursor.End)

            ellipsis = "..."

            metric = QFontMetrics(option.font)
            ellipsis_width = metric.horizontalAdvance(ellipsis)

            # endif
            while renderer.size().width() > option.rect.width() - ellipsis_width:
                cursor.deletePreviousChar()
                renderer.adjustSize()
            cursor.insertText(ellipsis)

        option.text = ''

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

        renderer.documentLayout().draw(painter, context)
        # renderer.drawContents(painter)
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
