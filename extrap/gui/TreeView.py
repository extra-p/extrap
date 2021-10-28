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


# TODO Expand largest

class TreeView(QTreeView):

    def __init__(self, parent):
        super(TreeView, self).__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAnimated(True)

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
        curr_tuple = tuple((value, node, parent_index))  # tuple with current node

        if not node.child_items:
            return [curr_tuple]

        # call max_path on children, then select list with largest max value, then insert current node in front of list
        node_index = model.index(node.row(), 0, parent_index)  # get index of current node
        children = [self.max_path(model, c, node_index) for c in node.child_items]  # call max_path on children

        children_max = [self.max_value(c) for c in
                        children]  # find the maximum value in each c, where c is a list of tuples

        ret_list = children[children_max.index(max(children_max))]
        ret_list.insert(0, curr_tuple)

        return ret_list

    def max_value(self, l):
        l.sort(key=lambda x: x[0], reverse=True)
        return l[0][0]

    def get_value_from_node(self, tree_model, node, parent_index):
        index = tree_model.index(node.row(), 0, parent_index)
        callpath = tree_model.getValue(index).path
        node_model = tree_model.getSelectedModel(callpath)
        return tree_model.get_comparison_value(node_model)

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
        expandLargest = expand_collapse_submenu.addAction("Expand largest")
        expandLargest.triggered.connect(lambda: self.expand_largest(model, self.selectedIndexes()[0]))
        expand_collapse_submenu.addSeparator()  # --------------------------------------------------
        collapseAction = expand_collapse_submenu.addAction("Collapse all")
        collapseAction.triggered.connect(self.collapseAll)
        collapseSubtree = expand_collapse_submenu.addAction("Collapse subtree")
        collapseSubtree.triggered.connect(
            lambda: self.collapseRecursively(self.selectedIndexes()[0]))
        return expand_collapse_submenu

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
