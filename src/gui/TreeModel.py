"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""

try:
    from PyQt4.QtCore import *
except ImportError:
    from PyQt5.QtCore import *  # @UnusedWildImport
from gui.Utils import formatFormula
from gui.Utils import formatNumber


class TreeModel(QAbstractItemModel):
    def __init__(self, selectorWidget, parent=None):
        super(TreeModel, self).__init__(parent)
        self.main_widget = selectorWidget.main_widget
        self.selector_widget = selectorWidget
        self.root_item = TreeItem(None)
        experiment = self.main_widget.getExperiment()
        if not experiment == None:
            call_tree = experiment.get_call_tree()
            nodes = call_tree.get_nodes()
            self.setupModelData(nodes, self.root_item)

    def removeRows(self, position=0, count=1,  parent=QModelIndex()):
        node = self.nodeFromIndex(parent)
        self.beginRemoveRows(parent, position, position + count - 1)
        node.childItems.pop(position)
        self.endRemoveRows()

    def nodeFromIndex(self, index):
        if index.isValid():
            return index.internalPointer()
        else:
            return self.root_item

    def getValue(self, index):
        item = index.internalPointer()
        if item == None:
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
        if getDecorationBoxes:
            delta = self.main_widget.max_value - self.main_widget.min_value
            if delta == 0:
                return None  # can't divide by zero

            model = self.getSelectedModel(callpath)
            if model == None:
                return None

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
            value = formula.evaluate(parameters)

            # convert value to relative value between 0 and 1
            relativeValue = max(0.0, (value - self.main_widget.min_value) / delta)

            return self.main_widget.color_widget.getColor(relativeValue)

        if index.column() == 0:
            return call_tree_node.name
        elif index.column() == 2:
            model = self.getSelectedModel(callpath)
            if model == None:
                return None
            # if len(model.getComments()) > 0:
            #     return len(model.getComments())
            else:
                return None
        elif index.column() == 3:
            experiment = self.main_widget.getExperiment()
            if not experiment == None:
                metric = self.selector_widget.getSelectedMetric()
                model = self.selector_widget.getCurrentModel()
                if model == None:
                    return None
                formula = model.models[(callpath, metric)].hypothesis.function
                if self.selector_widget.asymptoticCheckBox.isChecked():
                    parameters = tuple(experiment.parameters)
                    return formatFormula(formula.to_string(*parameters))
                else:
                    parameters = self.selector_widget.getParameterValues()
                    return formatNumber(str(formula.evaluate(parameters)))
        elif index.column() == 4:
            model = self.getSelectedModel(callpath).hypothesis
            if model == None:
                return None
            return formatNumber(str(model.get_RSS()))
        elif index.column() == 5:
            model = self.getSelectedModel(callpath).hypothesis
            if model == None:
                return None
            return formatNumber(str(model.get_AR2()))
        elif index.column() == 6:
            model = self.getSelectedModel(callpath).hypothesis
            if model == None:
                return None
            return formatNumber(str(model.get_SMAPE()))
        elif index.column() == 7:
            model = self.getSelectedModel(callpath).hypothesis
            if model == None:
                return None
            return formatNumber(str(model.get_RE()))
        return None

    def getSelectedModel(self, callpath):
        model = self.selector_widget.getCurrentModel()
        metric = self.selector_widget.getSelectedMetric()
        if model is None or metric is None:
            return None

        return model.models[(callpath, metric)]  # might be None

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
        if childItem == None:
            return QModelIndex()
        parentItem = childItem.parent()

        if parentItem == self.root_item:
            return QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

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
                              self.createIndex(100, self.columnCount(None) - 1))

    def getRootItem(self):
        return self.root_item


class TreeItem(object):
    def __init__(self, callpath, parent=None):
        self.parent_item = parent
        self.callpath = callpath
        self.child_items = []

    def appendChild(self, item):
        self.child_items.append(item)

    def child(self, row):
        return self.child_items[row]

    def data(self):
        return self.callpath

    def parent(self):
        return self.parent_item

    def row(self):
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        return 0
