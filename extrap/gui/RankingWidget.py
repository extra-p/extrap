from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING

from PySide6.QtCore import Slot, QModelIndex
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QWidget, QGridLayout, QToolBar, QAbstractItemView, QTreeView, QLabel

from extrap.comparison.entities.comparison_model import ComparisonModel
from extrap.comparison.entities.comparison_model_generator import ComparisonModelGenerator
from extrap.gui.components.BasicTableModel import BasicTableModel
from extrap.util.formatting_helper import format_number_plain_text

if TYPE_CHECKING:
    from extrap.gui.MainWidget import MainWidget


class RankingWidget(QWidget):
    def __init__(self, main_widget: MainWidget):
        super().__init__(main_widget)
        self._layout = QGridLayout()
        self._toolbar = QToolBar(self)
        self._largest = []
        self._smallest = []
        self._headers = ['Value', 'Callpath']
        self.list_length = 5

        self.init_ui()
        self.main_widget = main_widget
        self.main_widget.min_max_value_updated_event += lambda x, y: self.on_parameter_values_changed(
            self.main_widget.selector_widget.getParameterValues())

    def init_ui(self):
        self.setLayout(self._layout)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._largest_label = QLabel("")
        self._layout.addWidget(self._largest_label, 0, 0)
        self._smallest_label = QLabel("")
        self._layout.addWidget(self._smallest_label, 0, 1)

        self._largest_list = QTreeView()
        self._largest_list.setUniformRowHeights(True)
        self._largest_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._largest_model = BasicTableModel(self._headers, self._largest, self)
        self._largest_model.formatters[0] = format_number_plain_text
        self._largest_list.setModel(self._largest_model)
        self._largest_list.doubleClicked.connect(self.on_item_dblclick)
        self._smallest_list = QTreeView()
        self._smallest_list.setUniformRowHeights(True)
        self._smallest_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._smallest_model = BasicTableModel(self._headers, self._smallest)
        self._smallest_model.formatters[0] = format_number_plain_text
        self._smallest_list.setModel(self._smallest_model)
        self._layout.addWidget(self._largest_list, 1, 0)
        self._layout.addWidget(self._smallest_list, 1, 1)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.on_parameter_values_changed(self.main_widget.selector_widget.getParameterValues())

    def on_parameter_values_changed(self, values):
        if not self.isVisible():
            return
        model_gen = self.main_widget.get_current_model_gen()
        if not model_gen:
            return

        selected_metric = self.main_widget.get_selected_metric()
        if isinstance(model_gen, ComparisonModelGenerator):
            self._headers[0] = "Difference"
            names = self.main_widget.getExperiment().experiment_names
            self._smallest_label.setText(f"{names[0]} <= {names[1]}")
            self._largest_label.setText(f"{names[0]} >= {names[1]}")
            result_list = []
            for (callpath, metric), model in model_gen.models.items():
                if not isinstance(model, ComparisonModel):
                    continue
                if metric != selected_metric:
                    continue
                res = model.hypothesis.function.evaluate(values)
                result_list.append((res[1] - res[0], callpath.name))

            result_list.sort(key=itemgetter(0))
            self._largest_model.beginResetModel()
            self._largest.clear()
            self._largest.extend((abs(res[0]), *res[1:]) for res in result_list[:self.list_length] if res[0] <= 0)
            self._largest_model.endResetModel()

            self._smallest_model.beginResetModel()
            self._smallest.clear()
            self._smallest.extend(reversed([res for res in result_list[-self.list_length:] if res[0] >= 0]))
            self._smallest_model.endResetModel()
        else:
            self._headers[0] = "Value"
            self._smallest_label.setText("Smallest")
            self._largest_label.setText("Largest")
            result_list = []
            for (callpath, metric), model in model_gen.models.items():
                if metric != selected_metric:
                    continue
                res = model.hypothesis.function.evaluate(values)
                result_list.append((res, callpath.name))

            result_list.sort(key=itemgetter(0))
            self._largest_model.beginResetModel()
            self._largest.clear()
            self._largest.extend(reversed([res for res in result_list[-self.list_length:]]))
            self._largest_model.endResetModel()

            self._smallest_model.beginResetModel()
            self._smallest.clear()
            self._smallest.extend(res for res in result_list[:self.list_length])
            self._smallest_model.endResetModel()

    @Slot(QModelIndex)
    def on_item_dblclick(self, index: QModelIndex):
        # TODO select matching node in call tree
        pass
