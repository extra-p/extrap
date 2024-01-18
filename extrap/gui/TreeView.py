# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import math

from PySide6.QtCore import *  # @UnusedWildImport
from PySide6.QtGui import *  # @UnusedWildImport
from PySide6.QtWidgets import *  # @UnusedWildImport

from extrap.gui.TreeModel import TreeModel
from extrap.gui.components.annotation_delegate import AnnotationDelegate


# TODO Expand largest

class TreeView(QTreeView):

    def __init__(self, parent):
        super(TreeView, self).__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setItemDelegateForColumn(2, AnnotationDelegate())
        self.setAnimated(True)
        self.setAcceptDrops(True)

    def collapseRecursively(self, index):
        if not index.isValid():
            return

        self.collapse(index)

        child_count = index.model().rowCount(parent=index)
        for i in range(0, child_count):
            try:
                child = index.child(i, 0)
                self.collapseRecursively(child)
            except AttributeError:
                pass

    def expand_largest(self, model, index):
        root = index.internalPointer()
        root_parent_index = index.parent()

        # find path to node with largest value and expand along path
        arr = self.max_path(model, root, root_parent_index)
        for a in arr:
            self.expand(model.index(a[1].row(), 0, a[2]))

    def max_path(self, model, node, parent_index):
        """ find path with maximum value in tree """
        value = self.get_value_from_node(model, node, parent_index)
        if value is None:
            value = -math.inf
        curr_tuple = (value, node, parent_index)  # tuple with current node

        if not node.child_items:
            return [curr_tuple]

        # call max_path on children, then select list with largest max value, then insert current node in front of list
        node_index = model.index(node.row(), 0, parent_index)  # get index of current node
        children = [self.max_path(model, c, node_index) for c in node.child_items]  # call max_path on children

        children_max = [max(v for v, _, _ in c) for c in
                        children]  # find the maximum value in each c, where c is a list of tuples

        ret_list = children[children_max.index(max(children_max))]
        ret_list.insert(0, curr_tuple)

        return ret_list

    def get_value_from_node(self, tree_model, node, parent_index):
        index = tree_model.index(node.row(), 0, parent_index)
        callpath = tree_model.getValue(index).path
        node_model, _ = tree_model.getSelectedModel(callpath)
        if node_model:
            return tree_model.get_comparison_value(node_model)
        else:
            return None

    def contextMenuEvent(self, event):
        menu = QMenu()

        model: TreeModel = self.model()
        if model is not None:
            if self.selectedIndexes():
                selectedCallpath = model.getValue(
                    self.selectedIndexes()[0])
                if selectedCallpath is None:
                    return
                selectedModel, experiment = model.getSelectedModel(selectedCallpath.path)

                expandAction = menu.addAction("Expand all")
                expandAction.triggered.connect(self.expandAll)

                menu.addMenu(self._create_expand_collapse_menu(model))
                menu.addSeparator()  # --------------------------------------------------
                # showCommentsAction = menu.addAction("Show Comments")
                # showCommentsAction.setEnabled(
                #     selectedModel is not None and len(selectedModel.getComments()) > 0)
                # showCommentsAction.triggered.connect(
                #     lambda: self.showComments(selectedModel))
                showDataPointsAction = menu.addAction("Show data points")
                showDataPointsAction.setDisabled(selectedModel is None)
                showDataPointsAction.triggered.connect(
                    lambda: self.showDataPoints(selectedModel))
                copyModel = menu.addAction("Copy model")
                copyModel.setDisabled(selectedModel is None)
                copyModel.triggered.connect(
                    lambda: self.copy_model_to_clipboard(selectedModel, experiment)
                )
                menu.exec(self.mapToGlobal(event.pos()))

    def _create_expand_collapse_menu(self, model):
        expand_collapse_submenu = QMenu("Expand / Collapse")
        expandAction2 = expand_collapse_submenu.addAction("Expand all")
        expandAction2.triggered.connect(self.expandAll)
        expandSubtree = expand_collapse_submenu.addAction("Expand subtree")
        expandSubtree.triggered.connect(
            lambda: self.expandRecursively(self.selectedIndexes()[0]))
        expandLargest = expand_collapse_submenu.addAction("Expand largest")
        expandLargest.triggered.connect(lambda: self.expand_largest(model, self.selectedIndexes()[0]))
        expand_collapse_submenu.addSeparator()  # --------------------------------------------------
        collapseAction = expand_collapse_submenu.addAction("Collapse all")
        collapseAction.triggered.connect(self.collapseAll)
        collapseSubtree = expand_collapse_submenu.addAction("Collapse subtree")
        collapseSubtree.triggered.connect(
            lambda: self.collapseRecursively(self.selectedIndexes()[0]))
        return expand_collapse_submenu

    def copy_model_to_clipboard(self, selectedModel, experiment):
        parameters = self.model().main_widget.getExperiment().parameters
        if isinstance(selectedModel, list):
            param_value = selectedModel[0].changing_point.coordinate._values[0]
            function_string = selectedModel[0].hypothesis.function.to_string(*parameters) + " for " + str(experiment.parameters[0]) + "<=" + str(param_value)
            function_string = function_string + "\n" + selectedModel[1].hypothesis.function.to_string(*parameters) + " for " + str(
                            experiment.parameters[0]) + ">=" + str(param_value)
            QGuiApplication.clipboard().setText(function_string)
        else:
            function_string = selectedModel.hypothesis.function.to_string(*parameters)
            QGuiApplication.clipboard().setText(function_string)

    @staticmethod
    def showComments(model):
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
        msg.exec()

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

        if isinstance(model, list):
            msg_txt = "Callpath: " + model[0].callpath.name + "\n\n"
        else:
            msg_txt = "Callpath: " + model.callpath.name + "\n\n"

        if isinstance(model, list):
            for i in range(len(model)):
                if model[i].predictions is not None and model[i].measurements is not None:
                    msg_txt += "Model "+str(i+1)+":\n\n"
                    row_format = "{:20} {:>15} {:>15} {:>15}\n"
                    msg_txt += row_format.format("Coordinate", "Predicted", "Actual Mean", "Actual Median")
                    row_format = "{:20} {:>15g} {:>15g} {:>15g}\n"
                    for pred, m in zip(model[i].predictions, model[i].measurements):
                        msg_txt += row_format.format(str(m.coordinate), pred, m.mean, m.median)
                        # print(str(ps[i])+","+str(actual_points[i]))
                    msg_txt += "\n"
                else:
                    msg_txt += "No data available."
        else:
            if model.predictions is not None and model.measurements is not None:
                row_format = "{:20} {:>15} {:>15} {:>15}\n"
                msg_txt += row_format.format("Coordinate", "Predicted", "Actual Mean", "Actual Median")
                row_format = "{:20} {:>15g} {:>15g} {:>15g}\n"
                for pred, m in zip(model.predictions, model.measurements):
                    msg_txt += row_format.format(str(m.coordinate), pred, m.mean, m.median)
                    # print(str(ps[i])+","+str(actual_points[i]))
            else:
                msg_txt += "No data available."

        print(msg_txt)
        print("")
        msg.setText(msg_txt)
        msgBox.exec()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        self._handle_drop_event(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        self._handle_drop_event(event)

    def _handle_drop_event(self, event):
        event.setDropAction(Qt.DropAction.CopyAction)
        mimeData = event.mimeData()
        if mimeData.hasUrls() and len(mimeData.urls()) == 1 and mimeData.urls()[0].toLocalFile().endswith('.extra-p'):
            event.accept()
            return mimeData
        else:
            event.ignore()
            return None

    def dropEvent(self, event: QDropEvent) -> None:
        mimeData = self._handle_drop_event(event)
        if mimeData:
            file_name = mimeData.urls()[0].toLocalFile()  # Make sure to pass only python types into lambda
            QTimer.singleShot(0, lambda: self.parent().parent().main_widget.open_experiment(file_name))
