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
    from PyQt5.QtGui import *  # @UnusedWildImport
    from PyQt5.QtCore import *  # @UnusedWildImport
    from PyQt5.QtWidgets import *  # @UnusedWildImport


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
                selectedModel = self.model().getSelectedModel(selectedCallpath)
                expandAction = menu.addAction("Expand All")
                expandAction.triggered.connect(self.expandAll)
                showCommentsAction = menu.addAction("Show Comments")
                showCommentsAction.setEnabled(
                    selectedModel is not None and len(selectedModel.getComments()) > 0)
                showCommentsAction.triggered.connect(
                    lambda: self.showComments(selectedModel))
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
        #msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    # print the data points used in compute cost
    def showDataPoints(self, model):
        selectedCallpath = self.model().getValue(self.selectedIndexes()[0])
        callpath = selectedCallpath.getFullName()
        print("Callpath: "+callpath)
        print("Coordinate\tPredicted\tActual")
        predicted_points = model.getPredictedPoints()
        actual_points = model.getActualPoints()
        ps = model.getPs()
        #sizes = model.getSizes()
        for i in range(len(predicted_points)):
            # print("("+str(ps[i])+","+str(sizes[i])+")\t"+str(predicted_points[i])+"\t"+str(actual_points[i]))
            print(str(ps[i])+","+str(actual_points[i]))
        print("")
