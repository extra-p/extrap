# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from typing import Optional, TYPE_CHECKING, List

import numpy
from PySide2.QtCore import *  # @UnusedWildImport

from extrap.entities import calltree
from extrap.entities.model import Model
from extrap.gui.Utils import formatFormula
from extrap.gui.Utils import formatNumber

if TYPE_CHECKING:
    from extrap.gui.SelectorWidget import SelectorWidget


class TreeModel(QAbstractItemModel):
    def __init__(self, selectorWidget: SelectorWidget, parent=None):
        super(TreeModel, self).__init__(parent)
        self.main_widget = selectorWidget.main_widget
        self.selector_widget = selectorWidget
        self.root_item = TreeItem(None)
        experiment = self.main_widget.getExperiment()
        if experiment is not None:
            call_tree = experiment.call_tree
            nodes = call_tree.get_nodes()
            self.setupModelData(nodes, self.root_item)
            self.root_item.call_tree_node = call_tree

    def removeRows(self, position=0, count=1, parent=QModelIndex()):
        node = self.nodeFromIndex(parent)
        self.beginRemoveRows(parent, position, position + count - 1)
        node.childItems.pop(position)
        self.endRemoveRows()

    def nodeFromIndex(self, index):
        if index.isValid():
            return index.internalPointer()
        else:
            return self.root_item

    # noinspection PyMethodMayBeStatic
    def getValue(self, index):
        item = index.internalPointer()
        if item is None:
            return None
        return item.data()

    def data(self, index, role):
        if not index.isValid():
            return None

        getDecorationBoxes = (role == Qt.DecorationRole and index.column() == 1)

        if role != Qt.DisplayRole and not getDecorationBoxes:
            return None

        call_tree_node = index.internalPointer().data()
        if call_tree_node is None:
            return "Invalid"

        callpath = call_tree_node.path
        if callpath is None:
            return "Invalid"

        model = self.getSelectedModel(callpath)
        if model is None and index.column() != 0:
            return None

        if getDecorationBoxes:
            delta = self.main_widget.max_value - self.main_widget.min_value
            if delta == 0:
                return None  # can't divide by zero

            # code commented for single-parameter model
            '''
            paramValueList = EXTRAP.ParameterValueList()
            parameters = self.main_widget.experiment.getParameters()
            paramValueList[ parameters[0] ] = self.main_widget.spinBox.value()
            formula = model.hypothesis.function
            value = formula.evaluate( paramValueList )
            '''

            # added for two-parameter models here
            parameters = self.selector_widget.getParameterValues()
            formula = model.hypothesis.function
            previous = numpy.seterr(divide='ignore', invalid='ignore')
            value = formula.evaluate(parameters)
            numpy.seterr(**previous)

            # convert value to relative value between 0 and 1
            relativeValue = max(0.0, (value - self.main_widget.min_value) / delta)

            return self.main_widget.color_widget.getColor(relativeValue)

        if index.column() == 0:
            return call_tree_node.name
        elif index.column() == 2:
            # if len(model.getComments()) > 0:
            #     return len(model.getComments())
            return None
        elif index.column() == 3:
            experiment = self.main_widget.getExperiment()
            formula = model.hypothesis.function
            if self.selector_widget.asymptoticCheckBox.isChecked():
                parameters = tuple(experiment.parameters)
                return formatFormula(formula.to_string(*parameters))
            else:
                parameters = self.selector_widget.getParameterValues()
                previous = numpy.seterr(divide='ignore', invalid='ignore')
                res = formatNumber(str(formula.evaluate(parameters)))
                numpy.seterr(**previous)
                return res
        elif index.column() == 4:
            return formatNumber(str(model.hypothesis.RSS))
        elif index.column() == 5:
            return formatNumber(str(model.hypothesis.AR2))
        elif index.column() == 6:
            return formatNumber(str(model.hypothesis.SMAPE))
        elif index.column() == 7:
            return formatNumber(str(model.hypothesis.RE))
        return None

    def getSelectedModel(self, callpath) -> Optional[Model]:
        model = self.selector_widget.getCurrentModel()
        metric = self.selector_widget.getSelectedMetric()
        if model is None or metric is None:
            return None
        key = (callpath, metric)
        if key in model.models:
            return model.models[key]  # might be None
        else:
            return None

    def on_metric_changed(self):

        model = self.selector_widget.getCurrentModel()
        metric = self.selector_widget.getSelectedMetric()
        if model is None or metric is None:
            return None

        def selection_function(tree_item: TreeItem):
            key = (tree_item.call_tree_node.path, metric)
            return key in model.models

        if self.root_item.call_tree_node is not None:
            if self.root_item.does_selection_change(selection_function):
                self.beginResetModel()
                self.root_item.calculate_selection(selection_function)
                self.endResetModel()
            self.valuesChanged()

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags

        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def columnCount(self, _=None):
        return 8  # This needs to be updated when a new column is added!

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if section == 0:
                # This must have logical column index 0, because the tree structure is always shown there
                return "Callpath"
            elif section == 1:
                # Severity has logical column index 1, but is visually swapped by swapSections(0,1)
                return "Severity"
            elif section == 2:
                return "Comments"
            elif section == 3:
                return "Value"
            elif section == 4:
                return "RSS"
            elif section == 5:
                return "Adj. R²"
            elif section == 6:
                return "SMAPE"
            elif section == 7:
                return "RE"

        return None

    def index(self, row, column, parent):

        # print("In tree model # of rows",self.rowCount(parent))
        if row < 0 or column < 0 or row >= self.rowCount(parent) or column >= self.columnCount(parent):
            return QModelIndex()

        if not parent.isValid():
            parentItem = self.root_item
        else:
            parentItem = parent.internalPointer()

        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        childItem = index.internalPointer()
        if childItem is None:
            return QModelIndex()
        parentItem = childItem.parent()

        if parentItem == self.root_item:
            return QModelIndex()
        try:
            return self.createIndex(parentItem.row(), 0, parentItem)
        except ValueError:
            return QModelIndex()

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self.root_item
        else:
            parentItem = parent.internalPointer()

        return len(parentItem.child_items)

    def setupModelData(self, nodes, root):
        for i in range(0, len(nodes)):
            # region = callpaths[i].getRegion().name
            children = nodes[i].get_childs()
            new_tree_item = TreeItem(nodes[i], root)
            root.appendChild(new_tree_item)
            self.setupModelData(children, new_tree_item)

    def valuesChanged(self):
        self.dataChanged.emit(self.createIndex(0, 0),
                              self.createIndex(len(self.main_widget.getExperiment().callpaths) - 1,
                                               self.columnCount(None) - 1))

    def getRootItem(self):
        return self.root_item


class TreeItem(object):
    def __init__(self, call_tree_node, parent=None):
        self.parent_item = parent
        self.call_tree_node: calltree.Node = call_tree_node
        self.child_items: List[TreeItem] = []
        self._all_child_items: Optional[List[TreeItem]] = None
        self.is_selected = True

    def appendChild(self, item):
        self.child_items.append(item)

    def child(self, row):
        return self.child_items[row]

    def data(self):
        return self.call_tree_node

    def parent(self):
        return self.parent_item

    def row(self):
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        return 0

    def set_selection_recursive(self, value: bool):
        if value:
            self.child_items = self._all_child_items
        else:
            if self._all_child_items is None:
                self._all_child_items = self.child_items
            self.child_items = []
        for c in self.child_items:
            c.set_selection_recursive(value)

    def calculate_selection(self, selection_function) -> bool:
        if self.child_items or self._all_child_items:
            if self._all_child_items is None:
                self._all_child_items = self.child_items
            self.child_items = [c for c in self._all_child_items if c.calculate_selection(selection_function)]
        select = self.child_items or selection_function(self)
        self.is_selected = select
        return select

    def does_selection_change(self, selection_function) -> bool:
        if self.child_items or self._all_child_items:
            if self._all_child_items is None:
                self._all_child_items = self.child_items
            return any(c.does_selection_change(selection_function) for c in self._all_child_items)
        elif selection_function(self) != self.is_selected:
            return True
        else:
            return False
