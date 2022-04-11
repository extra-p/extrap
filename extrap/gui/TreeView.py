# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide2.QtCore import *  # @UnusedWildImport
from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport

from extrap.comparison.experiment_comparison import COMPARISON_NODE_NAME, TAG_COMPARISON_NODE
from extrap.gui.TreeModel import TreeModel, TreeItem

if TYPE_CHECKING:
    from extrap.gui.MainWidget import MainWidget
    from extrap.gui.SelectorWidget import SelectorWidget


# TODO Expand largest

class TreeView(QTreeView):

    def __init__(self, parent, selector_widget: SelectorWidget):
        super(TreeView, self).__init__(parent)
        self._selector_widget = selector_widget
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAnimated(True)
        self.setAcceptDrops(True)

    def collapseRecursively(self, index):
        if not index.isValid():
            return

        self.collapse(index)

        child_count = index.model().rowCount(parent=index)
        for i in range(0, child_count):
            child = index.child(i, 0)
            self.collapseRecursively(child)

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
        node_model = tree_model.getSelectedModel(callpath)
        if node_model:
            return tree_model.get_comparison_value(node_model)
        else:
            return None

    def expand_recursively_without_comparison(self, index=None, context=None):
        if not index or not context:
            index = self.selectedIndexes()[0]
            context = type('', (object,), {"treat_comparison_name_as_comparison": None})()
        node: TreeItem = index.internalPointer()

        for c in node.child_items:
            child_index = self.model().index(c.row(), 0, index)
            if c.data().path.tags.get(TAG_COMPARISON_NODE) != 'comparison':

                # Check for COMPARISON_NODE_NAME to support legacy comparison experiments
                if context.treat_comparison_name_as_comparison is None and c.data().name == COMPARISON_NODE_NAME:
                    result = QMessageBox.question(self, "Comparison detection",
                                                  f'Treat nodes which are named "{COMPARISON_NODE_NAME}" as comparison '
                                                  'nodes?')
                    if result == QMessageBox.Yes:
                        context.treat_comparison_name_as_comparison = True
                        continue
                    else:
                        context.treat_comparison_name_as_comparison = False
                elif context.treat_comparison_name_as_comparison and c.data().name == COMPARISON_NODE_NAME:
                    continue

                self.expand_recursively_without_comparison(child_index, context)
        self.expand(index)

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

                menu.addMenu(self._create_expand_collapse_menu(model))
                if self._selector_widget.main_widget.developer_mode:
                    menu.addSeparator()
                    menu.addMenu(self._create_developer_menu(model, selectedCallpath))
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
                    lambda: self.copy_model_to_clipboard(selectedModel)
                )
                menu.exec_(self.mapToGlobal(event.pos()))

    def _create_expand_collapse_menu(self, model):
        expand_collapse_submenu = QMenu("Expand / Collapse")
        expandAction2 = expand_collapse_submenu.addAction("Expand all")
        expandAction2.triggered.connect(self.expandAll)
        expandSubtree = expand_collapse_submenu.addAction("Expand subtree")
        expandSubtree.triggered.connect(
            lambda: self.expandRecursively(self.selectedIndexes()[0]))
        expandSubtree = expand_collapse_submenu.addAction("Expand subtree without comparisons")
        expandSubtree.triggered.connect(self.expand_recursively_without_comparison)
        expandLargest = expand_collapse_submenu.addAction("Expand largest")
        expandLargest.triggered.connect(lambda: self.expand_largest(model, self.selectedIndexes()[0]))
        expand_collapse_submenu.addSeparator()  # --------------------------------------------------
        collapseAction = expand_collapse_submenu.addAction("Collapse all")
        collapseAction.triggered.connect(self.collapseAll)
        collapseSubtree = expand_collapse_submenu.addAction("Collapse subtree")
        collapseSubtree.triggered.connect(
            lambda: self.collapseRecursively(self.selectedIndexes()[0]))
        return expand_collapse_submenu

    def _create_developer_menu(self, selectedModel, selectedCallpath):
        submenu = QMenu("Developer tools")
        showInfoAction = submenu.addAction("Show tags")
        showInfoAction.triggered.connect(
            lambda: self.show_info(selectedModel, selectedCallpath.path))
        submenu.addSeparator()
        showInfoAction = submenu.addAction("Delete subtree")
        showInfoAction.triggered.connect(
            lambda: self.delete_subtree(selectedModel))
        return submenu

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
        allComments = '\n'.join(("– " + c.getMessage())
                                for c in model.getComments())
        msg.setInformativeText(allComments)
        msg.setWindowTitle("Model Comments")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    @staticmethod
    def show_info(model, callpath):
        if not model and not callpath:
            return
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        if callpath:
            msg.setText(
                f"Tags for callpath {callpath}:")
            allComments = '\n'.join(f"{tag}: {value}" for tag, value in callpath.tags.items())
            msg.setInformativeText(allComments)
        msg.setWindowTitle("Model Info")
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
        msgBox.exec_()

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
            QTimer.singleShot(0, lambda: self._selector_widget.main_widget.open_experiment(file_name))

    def delete_subtree(self, model):
        if not self.selectedIndexes():
            return
        selectedCallpaths = [model.getValue(i) for i in self.selectedIndexes()]

        for selectedCallpath in selectedCallpaths:
            if not selectedCallpath.path:
                continue
            callpath = selectedCallpath.path
            main_widget: MainWidget = self._selector_widget.main_widget
            experiment = main_widget.getExperiment()
            callpaths_to_delete = [(i, c) for i, c in enumerate(experiment.callpaths) if
                                   c.name.startswith(callpath.name)]

            for callpath_index, callpath_to_delete in reversed(callpaths_to_delete):
                del experiment.callpaths[callpath_index]  # make sure to delete only once
                for metric in experiment.metrics:
                    key = (callpath_to_delete, metric)
                    experiment.measurements.pop(key, None)
                    for modeler in experiment.modelers:
                        modeler.models.pop(key, None)
        self.model().valuesChanged()
