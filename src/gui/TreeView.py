"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


try:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
except ImportError:
    from PySide2.QtGui import *  # @UnusedWildImport
    from PySide2.QtCore import *  # @UnusedWildImport
    from PySide2.QtWidgets import *  # @UnusedWildImport


class TreeView(QTreeView):

    def __init__(self, parent):

        super(TreeView, self).__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def contextMenuEvent(self, event):
        menu = QMenu()
        if self.model() is not None:
            if self.selectedIndexes():
                selectedCallpath = self.model().getValue(
                    self.selectedIndexes()[0])
                selectedModel = self.model().getSelectedModel(selectedCallpath.path)
                expandAction = menu.addAction("Expand All")
                expandAction.triggered.connect(self.expandAll)
                # showCommentsAction = menu.addAction("Show Comments")
                # showCommentsAction.setEnabled(
                #     selectedModel is not None and len(selectedModel.getComments()) > 0)
                # showCommentsAction.triggered.connect(
                #     lambda: self.showComments(selectedModel))
                showDataPointsAction = menu.addAction("Show Data Points")
                showDataPointsAction.triggered.connect(
                    lambda: self.showDataPoints(selectedModel))
                menu.exec_(self.mapToGlobal(event.pos()))

    def showComments(self, model):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            "Model has the following comments attached (text can be copied to the clipboard using the context menu):")
        allComments = '\n'.join(("– " + c.getMessage())
                                for c in model.getComments())
        msg.setInformativeText(allComments)
        msg.setWindowTitle("Model Comments")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    # print the data points used in compute cost
    def showDataPoints(self, model):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        msg.setFont(QFont('Courier'))
        horizontalSpacer = QSpacerItem(800, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout = msg.layout()
        layout.addItem(horizontalSpacer, layout.rowCount(), 0, 1, layout.columnCount())

        msg_txt = "Callpath: " + model.callpath.name+"\n\n"
        row_format = "{:20} {:>15} {:>15} {:>15}\n"
        msg_txt += row_format.format("Coordinate", "Predicted", "Actual Mean", "Actual Median")
        row_format = "{:20} {:>15g} {:>15g} {:>15g}\n"
        for pred, m in zip(model.predictions, model.measurements):
            msg_txt += row_format.format(str(m.coordinate), pred, m.mean, m.median)
            # print(str(ps[i])+","+str(actual_points[i]))
        print(msg_txt)
        print("")
        msg.setText(msg_txt)
        msg.setWindowTitle("Data Points")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
