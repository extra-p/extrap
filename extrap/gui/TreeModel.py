# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import copy
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING, List, Callable

import numpy
from PySide2.QtCore import *  # @UnusedWildImport

from extrap.entities import calltree
from extrap.entities.calltree import CallTree, Node
from extrap.entities.model import Model
from extrap.gui.Utils import formatFormula
from extrap.gui.Utils import formatNumber

import re

if TYPE_CHECKING:
    from extrap.gui.SelectorWidget import SelectorWidget


class TreeModel(QAbstractItemModel):
    def __init__(self, selector_widget: SelectorWidget, parent=None):
        super(TreeModel, self).__init__(parent)
        self.main_widget = selector_widget.main_widget
        self.selector_widget = selector_widget
        self.root_item = TreeItem(None)
        self.item_filter = TreeItemFilterProvider(self)
        experiment = self.main_widget.getExperiment()
        if experiment is not None:
            call_tree = experiment.call_tree
            self.on_metric_changed()
            self.item_filter.setup(call_tree)

    def removeRows(self, position=0, count=1, parent=QModelIndex()):
        node = self._node_from_index(parent)
        self.beginRemoveRows(parent, position, position + count - 1)
        node.childItems.pop(position)
        self.endRemoveRows()

    def _node_from_index(self, index):
        if index.isValid():
            return index.internalPointer()
        else:
            return self.root_item

    # noinspection PyMethodMayBeStatic
    def getValue(self, index) -> Optional[calltree.Node]:
        item = index.internalPointer()
        if item is None:
            return None
        return item.data()

    def data(self, index, role=None):
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
            delta = self.selector_widget.max_value - self.selector_widget.min_value
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
            if self.selector_widget.show_parameters.isChecked():
                return call_tree_node.name
            else:
                return self.remove_method_parameters(call_tree_node.name)
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

    def remove_method_parameters(self, name):
        depth = 0
        start = -1
        re.escape("()")
        for i in range(0, len(name)):
            elem = name[i]
            if elem == "<":
                depth += 1
                if start == -1:
                    start = i
            if elem == ">":
                depth -= 1
                if start != -1 and depth == 0:
                    name = name[0:start:] + "<…>" + self.remove_method_parameters(name[i+1:len(name):])
                    return re.sub(r"\(.*\)", "(…)", name)
        return re.sub(r"\(.*\)", "(…)", name)



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

        def selection_function(tree_item: Node):
            key = (tree_item.path, metric)
            return key in model.models

        self.item_filter.put_condition('metric', selection_function)

        # if self.root_item.call_tree_node is not None:
        #     if self.root_item.does_selection_change(selection_function):
        #         self.beginResetModel()
        #         self.root_item.calculate_selection(selection_function)
        #         self.endResetModel()
        #     self.valuesChanged()

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

    def index(self, row, column, parent=None):

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

    def parent(self, index=...):
        if not index.isValid():
            return QModelIndex()

        childItem: TreeItem = index.internalPointer()
        if childItem is None:
            return QModelIndex()
        parentItem = childItem.parent()

        if parentItem == self.root_item:
            return QModelIndex()
        try:
            return self.createIndex(parentItem.row(), 0, parentItem)
        except ValueError:
            return QModelIndex()

    def rowCount(self, parent=None):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self.root_item
        else:
            parentItem = parent.internalPointer()

        return len(parentItem.child_items)

    def valuesChanged(self):
        if not self.main_widget.getExperiment():
            return
        self.dataChanged.emit(self.createIndex(0, 0),
                              self.createIndex(len(self.main_widget.getExperiment().callpaths) - 1,
                                               self.columnCount(None) - 1))


class TreeItem(object):
    def __init__(self, call_tree_node, parent=None):
        self.parent_item = parent
        self.call_tree_node: calltree.Node = call_tree_node
        self.child_items: List[TreeItem] = []
        self.is_skip_item = False

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


class TreeItemFilterProvider:
    class ConstructionType(Enum):
        INCLUDE = auto()
        EXCLUDE = auto()
        FLAT = auto()
        COMPACT = auto()

    def __init__(self, model: TreeModel):
        self._model = model
        self._view_type = self.ConstructionType.INCLUDE
        self._call_tree: Optional[CallTree] = None
        self.conditions = {}
        self._type_builder = {
            self.ConstructionType.INCLUDE: self._construct_tree_include_child_if_mismatch,
            self.ConstructionType.COMPACT: self._construct_tree_skip_if_mismatch_and_at_most_one_child,
            self.ConstructionType.FLAT: self._construct_tree_flat
        }
        self._tree_builder = self._type_builder[self._view_type]

    def put_condition(self, id, condition: Callable[[Node], bool], inherited_from_child=True):
        self.conditions[id] = condition  # , inherited_from_child, {})
        self.update_tree()

    def remove_condition(self, id):
        if id in self.conditions:
            del self.conditions[id]

    @property
    def view_type(self):
        return self._view_type

    @view_type.setter
    def view_type(self, val):
        self._view_type = val
        self._tree_builder = self._type_builder[val]
        self.update_tree()

    def setup(self, call_tree: CallTree):
        self._call_tree = call_tree
        self.update_tree()

    def update_tree(self):
        if not self._call_tree:
            return
        root = TreeItem(self._call_tree)
        all_conditions = list(self.conditions.values())

        def predicate(node: Node):
            return all(cond(node) for cond in all_conditions)

        for child in self._call_tree:
            self._tree_builder(child, root, predicate)

        if len(self._model.root_item.child_items) == 0 or self.is_tree_changed(self._model.root_item, root):
            self._model.beginResetModel()
            self._model.root_item = root
            self._model.endResetModel()

        self._model.valuesChanged()

    @staticmethod
    def is_tree_changed(old_tree: TreeItem, new_tree: TreeItem):
        if len(old_tree.child_items) != len(new_tree.child_items):
            return True
        diverge = False
        for i in range(len(old_tree.child_items)):
            old_node = old_tree.child_items[i]
            new_node = new_tree.child_items[i]
            if old_node.call_tree_node.name != new_node.call_tree_node.name:
                diverge = True
                break
            else:
                diverge = TreeItemFilterProvider.is_tree_changed(old_node, new_node)
        return diverge

    @staticmethod
    def _construct_tree_exclude_child_if_mismatch(ct_node: Node, parent: TreeItem,
                                                  predicate: Callable[[Node], bool]):
        if predicate(ct_node):
            node = TreeItem(ct_node, parent)
            parent.appendChild(node)
            for ct_child in ct_node:
                TreeItemFilterProvider._construct_tree_exclude_child_if_mismatch(ct_child, node, predicate)

    @staticmethod
    def _construct_tree_include_child_if_mismatch(ct_node: Node, parent: TreeItem,
                                                  predicate: Callable[[Node], bool]):
        node = TreeItem(ct_node, parent)
        for ct_child in ct_node:
            TreeItemFilterProvider._construct_tree_include_child_if_mismatch(ct_child, node, predicate)
        if node.child_items or predicate(ct_node):
            parent.appendChild(node)

    @staticmethod
    def _construct_tree_flat(ct_node: Node, parent: TreeItem,
                             predicate: Callable[[Node], bool]):
        if predicate(ct_node):
            parent.appendChild(TreeItem(ct_node, parent))
        for ct_child in ct_node:
            TreeItemFilterProvider._construct_tree_flat(ct_child, parent, predicate)

    @staticmethod
    def _construct_tree_skip_if_mismatch_and_at_most_one_child(ct_node: Node, parent: TreeItem,
                                                               predicate: Callable[[Node], bool]):
        node = TreeItem(ct_node, parent)
        for ct_child in ct_node:
            TreeItemFilterProvider._construct_tree_skip_if_mismatch_and_at_most_one_child(ct_child, node, predicate)
        if predicate(ct_node) or len(node.child_items) > 1:
            parent.appendChild(node)
        elif len(node.child_items) == 1:
            child = node.child_items[0]
            if child.is_skip_item:
                child.call_tree_node.name = ct_node.name + '->' + child.call_tree_node.name
                child.parent_item = parent
                parent.child_items.append(child)
            else:
                node.is_skip_item = True
                node.call_tree_node = copy.copy(node.call_tree_node)
                parent.appendChild(node)

    @staticmethod
    def _construct_tree_skip_if_mismatch(ct_node: Node, parent: TreeItem,
                                         predicate: Callable[[Node], bool]):
        if predicate(ct_node):
            node = TreeItem(ct_node, parent)
            parent.appendChild(node)
            for ct_child in ct_node:
                TreeItemFilterProvider._construct_tree_skip_if_mismatch(ct_child, node, predicate)
        else:
            for ct_child in ct_node:
                TreeItemFilterProvider._construct_tree_skip_if_mismatch(ct_child, parent, predicate)
