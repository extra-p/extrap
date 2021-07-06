# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from PySide2.QtCore import *  # @UnusedWildImport
from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport

from extrap.gui.TreeModel import TreeModel


class TreeView(QTreeView):

    def __init__(self, parent):
        super(TreeView, self).__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAnimated(True)

    def contextMenuEvent(self, event):
        menu = QMenu()
        model: TreeModel = self.model()
        if model is not None:
            if self.selectedIndexes():
                selectedCallpath = model.getValue(
                    self.selectedIndexes()[0])
                if selectedCallpath is None:
                    return
                selectedModel = model.getSelectedModel(selectedCallpath.path)
                expandAction = menu.addAction("Expand all")
                expandAction.triggered.connect(self.expandAll)
                showCommentsAction = menu.addAction("Show comments")
                showCommentsAction.setEnabled(
                    selectedModel is not None and bool(selectedModel.comments))
                showCommentsAction.triggered.connect(
                    lambda: self.showComments(selectedModel))
                showDataPointsAction = menu.addAction("Show data points")
                showDataPointsAction.setDisabled(selectedModel is None)
                showDataPointsAction.triggered.connect(
                    lambda: self.showDataPoints(selectedModel))
                copyModel = menu.addAction("Copy model")
                copyModel.setDisabled(selectedModel is None)
                copyModel.triggered.connect(
                    lambda: self.copy_model_to_clipboard(selectedModel)
                )
                menu.exec_(self.mapToGlobal(event.pos()))

    def copy_model_to_clipboard(self, selectedModel):
        parameters = self.model().main_widget.getExperiment().parameters
        function_string = selectedModel.hypothesis.function.to_string(*parameters)
        QGuiApplication.clipboard().setText(function_string)

    @staticmethod
    def showComments(model):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            "Model has the following comments attached (text can be copied to the clipboard using the context menu):")
        allComments = '\n'.join(("– " + c)
                                for c in model.comments)
        msg.setInformativeText(allComments)
        msg.setWindowTitle("Model Comments")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    # print the data points used in compute cost
    @staticmethod
    def showDataPoints(model):
        msgBox = QDialog()
        msgBox.setWindowTitle("Data Points")
        msgBox.setFixedSize(600, 400)
        layout = QGridLayout()
        msg = QTextEdit()
        msg.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        msg.setFont(QFont('Courier'))
        layout.addWidget(msg)
        btn = QPushButton('OK', msgBox)
        btn.setDefault(True)
        btn.clicked.connect(msgBox.accept)
        layout.addWidget(btn)
        msgBox.setLayout(layout)

        msg_txt = "Callpath: " + model.callpath.name + "\n\n"
        row_format = "{:20} {:>15} {:>15} {:>15}\n"
        msg_txt += row_format.format("Coordinate", "Predicted", "Actual Mean", "Actual Median")
        row_format = "{:20} {:>15g} {:>15g} {:>15g}\n"
        for pred, m in zip(model.predictions, model.measurements):
            msg_txt += row_format.format(str(m.coordinate), pred, m.mean, m.median)
            # print(str(ps[i])+","+str(actual_points[i]))

        print(msg_txt)
        print("")
        msg.setText(msg_txt)
        msgBox.exec_()
