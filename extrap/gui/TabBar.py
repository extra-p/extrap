# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2025, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.


from pathlib import Path

from PySide6.QtCore import *  # @UnusedWildImport
from PySide6.QtGui import *  # @UnusedWildImport
from PySide6.QtWidgets import *  # @UnusedWildImport

import extrap
from extrap.util.event import Event
from PySide6.QtCore import *  # @UnusedWildImport
from PySide6.QtGui import *  # @UnusedWildImport
from PySide6.QtWidgets import *  # @UnusedWildImport

class TabBar(QTabBar):
    # Store experiments

    def __init__(self, main_widget: MainWidget, parent):
        super().__init__(parent)
        self.main_widget = main_widget
        self.experiments = []
        self._active_tab_index = 0

    def close_tab(self, tab_index: int):
        if self.count() == 1:
            return
        self.blockSignals(True)
        self.removeTab(tab_index)
        self.blockSignals(False)
        self.experiments.pop(tab_index)
        if self._active_tab_index > tab_index:
            self._active_tab_index -= 1
        elif self._active_tab_index == tab_index:
            self._active_tab_index = self.currentIndex()
        if self.count() > 0:
            index = self.currentIndex()
            if 0 <= index < len(self.experiments):
                self._active_tab_index = index
                experiment, file_name = self.experiments[index][:2]
                self.main_widget.set_experiment(experiment, file_name)
                self._restore_tab_state(index)

    def _save_tab_state(self, tab_index: int):
        """Save the metric index and selected callpath rows for tab_index so they can be restored later."""
        if 0 <= tab_index < len(self.experiments):
            exp, file_name, *_ = self.experiments[tab_index]
            metric_index = self.main_widget.selector_widget.metric_selector.currentIndex()
            tree_filter = self.main_widget.selector_widget.tree_display_select.currentIndex()

            all_selected_nodes = [idx for idx in self.main_widget.selector_widget.tree_view.selectedIndexes() if idx.column() == 0]
            traces = []

            for selected_node in all_selected_nodes:
                #selected_node contains  with parent, row , column but only the first are the important
                #get current row and check for parent
                trace_list = []
                trace_list.append(selected_node.row())
                parent = selected_node.parent()
                while parent.isValid():
                    trace_list.insert(0, parent.row())
                    parent = parent.parent()
                traces.append(trace_list)

            self.experiments[tab_index] = (exp, file_name, metric_index,traces,tree_filter)

    def _restore_tab_state(self, index: int):
        """Restore metric selection and callpath selection for the given tab index."""
        entry = self.experiments[index]
        if len(entry) < 4:
            # No saved state yet — auto-expand and select the first visible node
            self.main_widget.selector_widget.tree_view.expandAll()
            first = self.main_widget.selector_widget.tree_model.index(0, 0)
            if first.isValid():
                self.main_widget.selector_widget.tree_view.setCurrentIndex(first)
            return
        _, _, metric_index,traces,tree_filter= entry

        #self.selector_widget.tree_view = tree_view

        if metric_index >= 0:
            self.main_widget.selector_widget.metric_selector.setCurrentIndex(metric_index)

        sel_model = self.main_widget.selector_widget.tree_view.selectionModel()
        sel_model.clearSelection()

        self.main_widget.selector_widget.tree_display_select.setCurrentIndex(tree_filter)

        for trace in traces:
            parent = None
            for i in range(0,len(trace)):
                if i == 0:
                    index = self.main_widget.selector_widget.tree_model.index(trace[i], 0)
                    if index.isValid():
                        parent = index
                        if i == len(trace) - 1:
                            sel_model.select(index, sel_model.SelectionFlag.Select | sel_model.SelectionFlag.Rows)
                        else:
                            self.main_widget.selector_widget.tree_view.expand(index)
                    continue

                if i == len(trace) - 1:
                    index = self.main_widget.selector_widget.tree_model.index(trace[i], 0,parent)
                    if index.isValid():
                        sel_model.select(index, sel_model.SelectionFlag.Select | sel_model.SelectionFlag.Rows)
                    continue

                index = self.main_widget.selector_widget.tree_model.index(trace[i], 0, parent)
                if index.isValid():
                    self.main_widget.selector_widget.tree_view.expand(index)
                    parent = index


    def on_tab_changed(self, index: int):
        if 0 <= index < len(self.experiments):
            self._save_tab_state(self._active_tab_index)
            self._active_tab_index = index
            experiment, file_name = self.experiments[index][:2]
            self.main_widget.set_experiment(experiment, file_name)
            self._restore_tab_state(index)

